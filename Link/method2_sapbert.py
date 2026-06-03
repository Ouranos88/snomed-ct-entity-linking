"""
method2_sapbert.py — SapBERT bi-encoder + FAISS index + optional cross-encoder reranking.

Build:  build_index(snomed_rf2_file, output_dir)
Infer:  predict(note_text, sapbert_index) → prediction dicts (same format as Method 1)
"""

import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from Link.constants import SNOMED_FSN_TYPE

_torch = None
_transformers = None
_faiss = None


def _lazy_imports():
    global _torch, _transformers, _faiss
    if _torch is None:
        import torch
        _torch = torch
    if _transformers is None:
        import transformers
        _transformers = transformers
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError("faiss is required: pip install faiss-cpu")


def _print_device_info():
    _lazy_imports()
    if _torch.cuda.is_available():
        gpu = _torch.cuda.get_device_name(0)
        vram = _torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device: GPU — {gpu} ({vram:.1f} GB VRAM)", flush=True)
    else:
        print("  Device: CPU — no CUDA GPU detected (encoding will be slow)", flush=True)


TOP_K = 5
SAPBERT_MODEL = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
BATCH_SIZE = 128
MAX_LENGTH = 64


def _load_snomed_concepts(rf2_file: str) -> Dict[str, List[str]]:
    csv.field_size_limit(10_000_000)
    concepts: Dict[str, List[str]] = {}
    with open(rf2_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("active", "0") != "1":
                continue
            cid  = row.get("conceptId", "").strip()
            term = row.get("term", "").strip()
            if not cid or not term:
                continue
            concepts.setdefault(cid, [])
            if row.get("typeId") == SNOMED_FSN_TYPE:
                concepts[cid].insert(0, term)
            else:
                concepts[cid].append(term)
    return concepts


class _SapBERTEncoder:
    def __init__(self, model_name: str = SAPBERT_MODEL, device: Optional[str] = None):
        _lazy_imports()
        self.tokenizer = _transformers.AutoTokenizer.from_pretrained(model_name)
        self.model     = _transformers.AutoModel.from_pretrained(model_name)
        self.device    = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def encode(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_embs = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
            with _torch.no_grad():
                emb = self.model(**enc).last_hidden_state[:, 0, :]
                emb = _torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
            done = min(i + batch_size, total)
            bar = "#" * int(30 * done / total) + "-" * (30 - int(30 * done / total))
            print(f"  [{bar}] {done:,}/{total:,} concepts encoded", end="\r", flush=True)
        print(f"  [{'#' * 30}] {total:,}/{total:,} concepts encoded    ")
        return np.vstack(all_embs).astype("float32")


def build_index(snomed_rf2_file: str, output_dir: str, model_name: str = SAPBERT_MODEL) -> None:
    _lazy_imports()
    _print_device_info()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading SNOMED RF2 from {snomed_rf2_file} ...")
    concepts = _load_snomed_concepts(snomed_rf2_file)
    print(f"  {len(concepts)} active concepts loaded")

    # One vector per synonym (SapBERT trained on individual entity names)
    concept_ids, concept_texts = [], []
    for cid, terms in concepts.items():
        for term in terms[:8]:
            concept_ids.append(cid)
            concept_texts.append(term)
    print(f"  {len(concept_texts)} descriptions to encode")

    encoder = _SapBERTEncoder(model_name)
    embeddings = encoder.encode(concept_texts)

    index = _faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _faiss.write_index(index, str(out / "faiss_index.bin"))
    with open(out / "concept_ids.pkl", "wb") as f:
        pickle.dump(concept_ids, f)
    with open(out / "concept_texts.pkl", "wb") as f:
        pickle.dump(concept_texts, f)
    print(f"Index build complete → {output_dir}")


class SapBERTIndex:
    def __init__(self, index_dir: str, model_name: str = SAPBERT_MODEL):
        _lazy_imports()
        _print_device_info()
        p = Path(index_dir)
        print(f"Loading FAISS index from {index_dir} ...")
        self.index = _faiss.read_index(str(p / "faiss_index.bin"))
        with open(p / "concept_ids.pkl", "rb") as f:
            self.concept_ids: List[str] = pickle.load(f)
        print(f"  {self.index.ntotal} vectors loaded")
        self.encoder = _SapBERTEncoder(model_name)

    def search(self, query_texts: List[str], top_k: int = TOP_K) -> List[List[dict]]:
        embs = self.encoder.encode(query_texts, batch_size=256)
        scores, indices = self.index.search(embs, top_k)
        return [
            [{"concept_id": self.concept_ids[idx], "score": float(round(s, 4)), "tier": 2}
             for s, idx in zip(row_s, row_i) if idx >= 0]
            for row_s, row_i in zip(scores, indices)
        ]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        _lazy_imports()
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device    = "cuda" if _torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def rerank(self, query: str, candidates: List[dict], concept_texts: Dict[str, str]) -> List[dict]:
        if not candidates:
            return candidates
        pairs = [(query, concept_texts.get(c["concept_id"], c["concept_id"])) for c in candidates]
        enc = self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                             padding=True, truncation=True, max_length=128,
                             return_tensors="pt").to(self.device)
        with _torch.no_grad():
            scores = _torch.sigmoid(self.model(**enc).logits.squeeze(-1)).cpu().numpy()
        return [{**c, "score": float(round(s, 4)), "tier": 2}
                for c, s in sorted(zip(candidates, scores.tolist()), key=lambda x: -x[1])]


def _resolve_overlaps(predictions: List[dict]) -> List[dict]:
    preds = sorted(predictions,
                   key=lambda p: (p["start"], -p["candidates"][0]["score"], -(p["end"] - p["start"])))
    kept, last_end = [], -1
    for p in preds:
        if p["start"] >= last_end:
            kept.append(p); last_end = p["end"]
        elif kept and p["candidates"][0]["score"] > kept[-1]["candidates"][0]["score"]:
            kept[-1] = p; last_end = p["end"]
    return kept


def predict(
    note_text: str,
    sapbert_index: "SapBERTIndex",
    reranker: Optional["CrossEncoderReranker"] = None,
    resolve_overlaps: bool = True,
    skip_negated: bool = True,
    top_k: int = TOP_K,
    max_tokens: int = 8,
    min_score: float = 0.5,
    dictionary: Optional[dict] = None,
) -> List[dict]:
    from Link.preprocessor import extract_candidate_spans, is_negated

    spans = extract_candidate_spans(note_text, max_tokens=max_tokens)
    if skip_negated:
        spans = [s for s in spans if not is_negated(note_text, s[0])]

    # Dict pre-filter: keeps only spans with Tier 1/2 exact match, eliminating
    # false-positive sliding-window spans and dramatically improving precision.
    if dictionary is not None:
        from Link.method1_dictionary import _norm_key
        t1, t2 = dictionary["tier1"], dictionary["tier2"]
        spans = [s for s in spans if _norm_key(s[2]) in t1 or _norm_key(s[2]) in t2]
        min_score = 0.0

    if not spans:
        return []

    batch_candidates = sapbert_index.search([s[2] for s in spans], top_k=top_k)
    predictions = []
    for (start, end, span_text), candidates in zip(spans, batch_candidates):
        candidates = [c for c in candidates if c["score"] >= min_score]
        if not candidates:
            continue
        if reranker is not None:
            concept_texts = {c["concept_id"]: c["concept_id"] for c in candidates}
            candidates = reranker.rerank(span_text, candidates, concept_texts)
        predictions.append({"start": start, "end": end, "span": span_text,
                             "candidates": candidates[:top_k]})

    if resolve_overlaps:
        predictions = _resolve_overlaps(predictions)
    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Method 2: SapBERT + FAISS")
    sub = parser.add_subparsers(dest="cmd")

    _default_index = str(Path(__file__).parent / "results" / "sapbert_index")

    build_p = sub.add_parser("build")
    build_p.add_argument("--snomed-rf2", required=True)
    build_p.add_argument("--output-dir", default=_default_index)
    build_p.add_argument("--model", default=SAPBERT_MODEL)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--note", required=True)
    pred_p.add_argument("--index-dir", default=_default_index)
    pred_p.add_argument("--model", default=SAPBERT_MODEL)
    pred_p.add_argument("--rerank", action="store_true")

    args = parser.parse_args()

    if args.cmd == "build":
        build_index(args.snomed_rf2, args.output_dir, args.model)
    elif args.cmd == "predict":
        idx = SapBERTIndex(args.index_dir, args.model)
        reranker = CrossEncoderReranker() if args.rerank else None
        with open(args.note, encoding="utf-8") as f:
            text = f.read()
        for p in predict(text, idx, reranker=reranker):
            top = p["candidates"][0]
            print(f"  [{p['start']:5d}:{p['end']:5d}] {p['span']!r:40s}  "
                  f"→ {top['concept_id']}  (score={top['score']:.3f})")
    else:
        parser.print_help()
