"""
method3_llm_rag.py — LLM entity extraction + SNOMED RAG retrieval + LLM reranking.

Build:  build_index(snomed_rf2_file, output_dir)
Infer:  predict(note_text, rag_index, llm_client) → prediction dicts
"""

import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from Link.constants import CONTEXT_WINDOW, SNOMED_FSN_TYPE

_faiss = None
_sentence_transformers = None


def _print_device_info():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  Device: GPU — {gpu_name} ({vram:.1f} GB VRAM)", flush=True)
        else:
            print("  Device: CPU — no CUDA GPU detected (encoding will be slow)", flush=True)
    except ImportError:
        print("  Device: CPU (torch not installed)", flush=True)


def _lazy_imports():
    global _faiss, _sentence_transformers
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError("Install faiss: pip install faiss-cpu")
    if _sentence_transformers is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformers = SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")


TOP_K_RETRIEVAL = 20
TOP_K_OUTPUT = 5
EMBED_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "llama3.1:8b"
BATCH_SIZE = 64


class OllamaClient:
    def __init__(self, model_name: str = LLM_MODEL,
                 host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")):
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")
        self._client = _ollama.Client(host=host)  # one persistent client, not one per call
        self.model_name = model_name
        self.host = host

    def generate(self, system: str, user: str, max_tokens: int = 4096) -> str:
        response = self._client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"num_predict": max_tokens, "temperature": 0},
        )
        return response["message"]["content"].strip()


def _load_snomed_concepts(rf2_description_file: str) -> Dict[str, List[str]]:
    import csv
    csv.field_size_limit(10_000_000)
    concept_terms: Dict[str, List[str]] = {}
    with open(rf2_description_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("active", "0") != "1":
                continue
            concept_id = row.get("conceptId", "").strip()
            term = row.get("term", "").strip()
            type_id = row.get("typeId", "")
            if not concept_id or not term:
                continue
            if concept_id not in concept_terms:
                concept_terms[concept_id] = []
            if type_id == SNOMED_FSN_TYPE:
                concept_terms[concept_id].insert(0, term)
            else:
                concept_terms[concept_id].append(term)
    return concept_terms


def build_index(
    snomed_rf2_file: str,
    output_dir: str,
    embed_model: str = EMBED_MODEL,
    snomed_rf2_fr: str = None,
) -> None:
    _lazy_imports()
    _print_device_info()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading SNOMED RF2 from {snomed_rf2_file} ...")
    concept_terms = _load_snomed_concepts(snomed_rf2_file)

    if snomed_rf2_fr and Path(snomed_rf2_fr).exists():
        print(f"Loading French RF2 synonyms from {snomed_rf2_fr} ...")
        fr_terms = _load_snomed_concepts(snomed_rf2_fr)
        added = 0
        for cid, terms in fr_terms.items():
            if cid in concept_terms:
                # Append French synonyms after the English ones (cap at 3 FR terms)
                concept_terms[cid].extend(terms[:3])
                added += len(terms[:3])
        print(f"  {added} French synonyms merged into existing concepts")

    concept_ids = list(concept_terms.keys())
    # multilingual-e5 uses "passage: " prefix for documents
    concept_texts = [
        "passage: " + " | ".join(concept_terms[cid][:9])
        for cid in concept_ids
    ]
    print(f"  {len(concept_ids)} concepts")

    print(f"Encoding with {embed_model} ...")
    model = _sentence_transformers(embed_model)
    embeddings = model.encode(
        concept_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    print(f"  Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = _faiss.IndexFlatIP(dim)
    index.add(embeddings)
    del embeddings  # FAISS has its own copy; free the 2.17 GB numpy array now
    del model       # free GPU memory before Ollama needs it
    try:
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
    except ImportError:
        import gc; gc.collect()

    _faiss.write_index(index, str(output_path / "faiss_index.bin"))
    with open(output_path / "concept_ids.pkl", "wb") as f:
        pickle.dump(concept_ids, f)
    with open(output_path / "concept_texts.pkl", "wb") as f:
        # Store clean texts (without "passage: " prefix) for LLM reranking
        clean_texts = [" | ".join(concept_terms[cid][:6]) for cid in concept_ids]
        pickle.dump(clean_texts, f)

    print(f"Index saved to {output_dir}")


class RAGIndex:
    def __init__(self, index_dir: str, embed_model: str = EMBED_MODEL):
        _lazy_imports()
        _print_device_info()
        index_path = Path(index_dir)
        print(f"Loading RAG index from {index_dir} ...")
        self.index = _faiss.read_index(str(index_path / "faiss_index.bin"))
        with open(index_path / "concept_ids.pkl", "rb") as f:
            self.concept_ids: List[str] = pickle.load(f)
        with open(index_path / "concept_texts.pkl", "rb") as f:
            self.concept_texts: List[str] = pickle.load(f)
        # Force CPU so Ollama (llama3.1:8b, ~5 GB VRAM) can use the GPU uncontested.
        # Query encoding is fast on CPU: only 20–30 short spans per note.
        self.embed_model = _sentence_transformers(embed_model, device="cpu")
        print(f"  {self.index.ntotal} concepts loaded")

    def search(self, query_texts: List[str], top_k: int = TOP_K_RETRIEVAL) -> List[List[dict]]:
        # multilingual-e5 uses "query: " prefix for queries
        prefixed = ["query: " + t for t in query_texts]
        query_embs = self.embed_model.encode(
            prefixed,
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embs, top_k)
        return [
            [
                {"concept_id": self.concept_ids[idx], "description": self.concept_texts[idx],
                 "score": float(round(score, 4)), "tier": 3}
                for score, idx in zip(row_scores, row_indices) if idx >= 0
            ]
            for row_scores, row_indices in zip(scores, indices)
        ]


def _recover_offset(text: str, span: str) -> Optional[Tuple[int, int]]:
    idx = text.find(span)
    if idx >= 0:
        return idx, idx + len(span)

    idx = text.lower().find(span.lower())
    if idx >= 0:
        return idx, idx + len(span)

    try:
        from rapidfuzz import fuzz, process
        result = process.extractOne(
            span.lower(),
            [text[i:i+len(span)+10] for i in range(0, max(1, len(text)-len(span)), 5)],
            scorer=fuzz.ratio,
            score_cutoff=80,
        )
        if result:
            matched, score, idx_in_list = result
            char_pos = idx_in_list * 5
            for delta in range(-10, 10):
                pos = char_pos + delta
                if 0 <= pos <= len(text) - len(matched):
                    if fuzz.ratio(matched.lower(), text[pos:pos+len(matched)].lower()) >= 80:
                        return pos, pos + len(span)
    except ImportError:
        pass

    return None


_NER_SYSTEM = (
    "You are a clinical NLP expert. Extract all medical entities from the following "
    "clinical note that could be mapped to SNOMED CT concepts.\n\n"
    "SCOPE: Include ALL diagnoses and findings — both primary ophthalmic findings AND "
    "systemic comorbidities mentioned anywhere in the note (patient history, "
    "background conditions, co-morbidities such as hypertension, diabetes, etc.).\n\n"
    "CRITICAL RULES:\n"
    "- Copy the span EXACTLY as it appears in the note — character for character.\n"
    "- NEVER expand abbreviations. If the note says 'IOL', return 'IOL'. "
    "If it says 'CA profonde', return 'CA profonde'. "
    "If it says 'neuropathie optique bilat', return 'neuropathie optique bilat'.\n"
    "- NEVER paraphrase, translate, or complete partial words.\n"
    "- Short abbreviations (IOL, DMEK, OCT, OD, OG, AV, PIO, HTA, HTO) are valid entities — include them.\n\n"
    "Semantic types: disorder, finding, procedure, body_structure, substance, observable_entity\n\n"
    "Return ONLY a JSON array. Each element: "
    '{"span": "...", "type": "..."}\n'
    "No explanation, no markdown fencing — raw JSON only."
)

_NER_CHUNK_SIZE = 12000
_NER_CHUNK_OVERLAP = 500


def _llm_extract_spans(
    note_text: str,
    client: OllamaClient,
    model: str = LLM_MODEL,
    max_retries: int = 3,
) -> List[dict]:
    chunks: List[str] = []
    if len(note_text) <= _NER_CHUNK_SIZE:
        chunks = [note_text]
    else:
        start = 0
        while start < len(note_text):
            end = min(start + _NER_CHUNK_SIZE, len(note_text))
            if end < len(note_text):
                boundary = note_text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunks.append(note_text[start:end])
            if end >= len(note_text):
                break
            start = end - _NER_CHUNK_OVERLAP

    def _call_llm(text: str) -> List[dict]:
        for attempt in range(max_retries):
            try:
                raw = client.generate(_NER_SYSTEM, text, max_tokens=4096)
                clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.DOTALL)
                clean = re.sub(r"\s*```\s*$", "", clean, flags=re.DOTALL).strip()
                parsed = json.loads(clean)
                if isinstance(parsed, list):
                    return parsed
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
            except (json.JSONDecodeError, IndexError):
                pass
            except Exception as e:
                err = str(e).lower()
                if any(kw in err for kw in ("quota", "rate", "resource_exhausted", "429")):
                    wait = 60 if attempt == 0 else 2 ** attempt
                    print(f"\n  [Method 3] Rate limit hit, waiting {wait}s ...", flush=True)
                    time.sleep(wait)
                else:
                    raise
        return []

    seen: set = set()
    results: List[dict] = []
    for chunk in chunks:
        for item in _call_llm(chunk):
            key = item.get("span", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                results.append(item)
    return results


_RERANK_SYSTEM = (
    "You are a clinical terminology expert. Given a medical span extracted from a clinical "
    "note, select the most appropriate SNOMED CT concept from the candidates provided.\n"
    "Consider the full clinical context.\n\n"
    "Return ONLY a JSON object: "
    '{"selected_concept_id": "...", "confidence": "high|medium|low"}\n'
    "No explanation, no markdown — raw JSON only."
)


def _llm_rerank(
    span_text: str,
    entity_type: str,
    context: str,
    candidates: List[dict],
    client: OllamaClient,
    model: str = LLM_MODEL,
) -> Optional[str]:
    cand_lines = "\n".join(
        f"{i+1}. [{c['concept_id']}] {c['description']}"
        for i, c in enumerate(candidates[:TOP_K_RETRIEVAL])
    )
    user_msg = (
        f"Clinical context:\n{context[:500]}\n\n"
        f"Span: \"{span_text}\"\n"
        f"Semantic type: {entity_type}\n\n"
        f"Candidates:\n{cand_lines}\n"
    )
    try:
        raw = client.generate(_RERANK_SYSTEM, user_msg, max_tokens=256)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        return str(parsed.get("selected_concept_id", ""))
    except Exception:
        return None


def _get_context(text: str, start: int, end: int, window: int = CONTEXT_WINDOW) -> str:
    return text[max(0, start - window):min(len(text), end + window)]


def _resolve_overlaps(predictions: List[dict]) -> List[dict]:
    preds = sorted(
        predictions,
        key=lambda p: (p["start"], -p["candidates"][0]["score"], -(p["end"] - p["start"])),
    )
    kept, last_end = [], -1
    for p in preds:
        if p["start"] >= last_end:
            kept.append(p); last_end = p["end"]
        elif kept and p["candidates"][0]["score"] > kept[-1]["candidates"][0]["score"]:
            kept[-1] = p; last_end = p["end"]
    return kept


def predict(
    note_text: str,
    rag_index: RAGIndex,
    llm_client: OllamaClient,
    use_llm_reranking: bool = True,
    resolve_overlaps: bool = True,
    skip_negated: bool = True,
    top_k: int = TOP_K_OUTPUT,
    llm_model: str = LLM_MODEL,
) -> List[dict]:
    from Link.preprocessor import is_negated

    llm_spans = _llm_extract_spans(note_text, llm_client, model=llm_model)

    resolved_spans: List[Tuple[int, int, str, str]] = []
    for item in llm_spans:
        span_text = item.get("span", "").strip()
        entity_type = item.get("type", "finding")
        if not span_text:
            continue
        offsets = _recover_offset(note_text, span_text)
        if offsets is None:
            continue
        start, end = offsets
        if skip_negated and is_negated(note_text, start):
            continue
        resolved_spans.append((start, end, span_text, entity_type))

    if not resolved_spans:
        return []

    MAX_SPANS_PER_NOTE = 80
    if len(resolved_spans) > MAX_SPANS_PER_NOTE:
        resolved_spans = resolved_spans[:MAX_SPANS_PER_NOTE]

    from Link.constants import expand_abbrevs
    span_texts = [expand_abbrevs(s[2]) for s in resolved_spans]
    batch_candidates = rag_index.search(span_texts, top_k=TOP_K_RETRIEVAL)

    predictions = []
    for (start, end, span_text, entity_type), candidates in zip(resolved_spans, batch_candidates):
        if not candidates:
            continue

        if use_llm_reranking and len(candidates) > 1:
            context = _get_context(note_text, start, end)
            best_id = _llm_rerank(
                span_text, entity_type, context, candidates,
                llm_client, model=llm_model,
            )
            if best_id:
                reranked = [c for c in candidates if c["concept_id"] == best_id]
                reranked += [c for c in candidates if c["concept_id"] != best_id]
                candidates = reranked

        predictions.append({
            "start": start,
            "end": end,
            "span": span_text,
            "candidates": candidates[:top_k],
        })

    if resolve_overlaps:
        predictions = _resolve_overlaps(predictions)

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Method 3: LLM + SNOMED RAG")
    sub = parser.add_subparsers(dest="cmd")

    build_p = sub.add_parser("build", help="Build FAISS index from SNOMED RF2")
    build_p.add_argument("--snomed-rf2", required=True)
    build_p.add_argument("--snomed-rf2-fr", default=None)
    _default_index = str(Path(__file__).parent / "results" / "rag_index")
    build_p.add_argument("--output-dir", default=_default_index)
    build_p.add_argument("--embed-model", default=EMBED_MODEL)

    pred_p = sub.add_parser("predict", help="Predict on a single note text file")
    pred_p.add_argument("--note", required=True)
    pred_p.add_argument("--index-dir", default=_default_index)
    pred_p.add_argument("--no-rerank", action="store_true")

    args = parser.parse_args()

    if args.cmd == "build":
        build_index(args.snomed_rf2, args.output_dir, args.embed_model, args.snomed_rf2_fr)

    elif args.cmd == "predict":
        client = OllamaClient(model_name=LLM_MODEL)
        idx = RAGIndex(args.index_dir)
        with open(args.note, encoding="utf-8") as f:
            text = f.read()
        preds = predict(text, idx, client, use_llm_reranking=not args.no_rerank)
        for p in preds:
            top = p["candidates"][0]
            print(f"  [{p['start']:5d}:{p['end']:5d}] {p['span']!r:40s}  "
                  f"→ {top['concept_id']}  (score={top['score']:.3f})")
    else:
        parser.print_help()
