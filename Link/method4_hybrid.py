"""
method4_hybrid.py — LLM NER + tiered dictionary + FAISS.

No new build step required — reuses Method 1 dictionary and Method 3 FAISS index.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

TOP_K_RETRIEVAL = 20
TOP_K_OUTPUT    = 5
LLM_MODEL       = "llama3.1:8b"
RERANK_BATCH    = 8

# If dict concept is in FAISS top-3, skip LLM reranking (they agree)
TIER2_FAISS_AGREE_RANK = 3

_RERANK_SYSTEM = (
    "You are a clinical terminology expert. For each numbered span below, select the best "
    "SNOMED CT concept from its candidate list. Use the shared clinical context.\n\n"
    "Return ONLY a JSON array (one object per span, same order):\n"
    '[{"span_id": 0, "selected_concept_id": "..."}, {"span_id": 1, "selected_concept_id": "..."}, ...]\n'
    "No explanation, no markdown — raw JSON array only."
)


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


def _llm_rerank_batch(
    items: List[dict],
    client,
    model: str = LLM_MODEL,
) -> List[Optional[str]]:
    parts = []
    for i, item in enumerate(items):
        cand_lines = "\n".join(
            f"  {j+1}. [{c['concept_id']}] {c.get('description', c['concept_id'])}"
            for j, c in enumerate(item["candidates"][:TOP_K_RETRIEVAL])
        )
        parts.append(
            f"[{i}] Span: \"{item['span']}\"\n"
            f"    Context: {item['context'][:200]}\n"
            f"    Candidates:\n{cand_lines}"
        )
    user_msg = "\n\n".join(parts)

    try:
        raw = client.generate(_RERANK_SYSTEM, user_msg, max_tokens=128 * len(items))
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        result = [None] * len(items)
        for entry in parsed:
            sid = entry.get("span_id")
            cid = entry.get("selected_concept_id", "")
            if isinstance(sid, int) and 0 <= sid < len(items) and cid:
                result[sid] = str(cid)
        return result
    except Exception:
        return [None] * len(items)


def _get_context(text: str, start: int, end: int, window: int = 300) -> str:
    return text[max(0, start - window): min(len(text), end + window)]


# Matches SNOMED semantic tags like " (disorder)", " (finding)", etc.
_SEMANTIC_TAG_RE = re.compile(r'\s*\([^)]+\)\s*$')


def _boost_by_string_match(span_text: str, candidates: list, threshold: float = 82.0) -> list:
    """Boost candidates whose SNOMED preferred term closely matches the span.

    FAISS retrieves semantically similar concepts but may rank a neighbouring
    concept above the one whose preferred term actually matches the span text.
    A high string-similarity score between the span and a candidate's
    description is a strong signal that this candidate is correct, so we
    boost its score before the LLM reranker sees the list.
    """
    from rapidfuzz import fuzz
    from Link.method1_dictionary import _norm_key

    norm_span = _norm_key(span_text)
    if not norm_span:
        return candidates

    boosted = []
    for c in candidates:
        desc = c.get("description", "")
        if desc:
            clean_desc = _SEMANTIC_TAG_RE.sub("", desc).strip()
            similarity = fuzz.token_sort_ratio(norm_span, _norm_key(clean_desc))
            if similarity >= threshold:
                boost = (similarity - threshold) / (100.0 - threshold) * 0.15
                c = {**c, "score": min(0.99, round(c["score"] + boost, 4))}
        boosted.append(c)

    return sorted(boosted, key=lambda x: -x["score"])


def predict(
    note_text: str,
    dictionary: dict,
    rag_index,
    llm_client,
    use_llm_reranking: bool = True,
    resolve_overlaps: bool = True,
    skip_negated: bool = True,
    max_tokens: int = 8,
    top_k_output: int = TOP_K_OUTPUT,
    llm_model: str = LLM_MODEL,
) -> List[dict]:
    from Link.preprocessor import extract_candidate_spans, is_negated
    from Link.method1_dictionary import _norm_key, _build_fuzzy_keys, _lookup_fuzzy
    from Link.method3_llm_rag import _llm_extract_spans, _recover_offset
    from Link.constants import expand_abbrevs

    candidate_spans = []
    if llm_client is not None:
        llm_spans = _llm_extract_spans(note_text, llm_client, model=llm_model)
        for item in llm_spans:
            span_text = item.get("span", "").strip()
            if not span_text:
                continue
            offsets = _recover_offset(note_text, span_text)
            if offsets is None:
                continue
            start, end = offsets
            if skip_negated and is_negated(note_text, start):
                continue
            candidate_spans.append((start, end, span_text))

    if not candidate_spans:
        candidate_spans = extract_candidate_spans(note_text, max_tokens=max_tokens)
        if skip_negated:
            candidate_spans = [
                s for s in candidate_spans if not is_negated(note_text, s[0])
            ]

    tier1_dict = dictionary["tier1"]
    tier2_dict = dictionary["tier2"]

    tier1_preds = []
    tier2_spans = []
    tier3_spans = []
    faiss_only_spans = []
    fuzzy_keys = None  # lazy-init (expensive to build)

    for start, end, span_text in candidate_spans:
        key = _norm_key(expand_abbrevs(span_text))

        # Try progressively shorter prefixes if full span misses the dict.
        # "dyspnée d'effort ancienne" → "dyspnée d'effort" → HIT
        # Minimum 2 tokens to avoid single-word over-matching.
        def _find_key_with_trimming(full_key):
            tokens = full_key.split()
            for n in range(len(tokens), 1, -1):
                candidate_key = " ".join(tokens[:n])
                if candidate_key in tier1_dict or candidate_key in tier2_dict:
                    return candidate_key
            return full_key

        if key not in tier1_dict and key not in tier2_dict:
            trimmed = _find_key_with_trimming(key)
            if trimmed != key:
                key = trimmed

        if key in tier1_dict:
            entries = tier1_dict[key]
            total = sum(c for _, c in entries)
            candidates = [
                {"concept_id": cid, "score": round(cnt / total, 4), "tier": 1}
                for cid, cnt in entries[:top_k_output]
            ]
            tier1_preds.append({
                "start": start, "end": end, "span": span_text,
                "candidates": candidates,
            })

        elif key in tier2_dict:
            tier2_spans.append((start, end, span_text, tier2_dict[key]))

        else:
            if fuzzy_keys is None:
                fuzzy_keys = _build_fuzzy_keys(dictionary)
            fuzzy_cands = _lookup_fuzzy(key, dictionary, fuzzy_keys)
            if fuzzy_cands:
                tier3_spans.append((start, end, span_text, fuzzy_cands))
            else:
                faiss_only_spans.append((start, end, span_text, None))

    faiss_spans = tier2_spans + tier3_spans + faiss_only_spans
    faiss_preds = []

    if faiss_spans:
        from Link.constants import expand_abbrevs
        span_texts_for_faiss = [expand_abbrevs(s[2]) for s in faiss_spans]
        batch_candidates = rag_index.search(span_texts_for_faiss, top_k=TOP_K_RETRIEVAL)

        n_tier2 = len(tier2_spans)
        n_tier3 = len(tier3_spans)

        merged_list = []
        needs_rerank = []

        for idx, faiss_cands in enumerate(batch_candidates):
            if idx < n_tier2:
                start, end, span_text, dict_concept_id = tier2_spans[idx]
                is_tier2 = True
                fuzzy_cands = None
            elif idx < n_tier2 + n_tier3:
                start, end, span_text, fuzzy_cands = tier3_spans[idx - n_tier2]
                is_tier2 = False
                dict_concept_id = fuzzy_cands[0]["concept_id"] if fuzzy_cands else None
            else:
                start, end, span_text, _ = faiss_only_spans[idx - n_tier2 - n_tier3]
                is_tier2 = False
                fuzzy_cands = None
                dict_concept_id = None

            if not faiss_cands:
                if is_tier2:
                    faiss_preds.append({
                        "start": start, "end": end, "span": span_text,
                        "candidates": [{"concept_id": dict_concept_id, "score": 0.95, "tier": 2}],
                    })
                elif fuzzy_cands:
                    faiss_preds.append({
                        "start": start, "end": end, "span": span_text,
                        "candidates": fuzzy_cands[:top_k_output],
                    })
                merged_list.append(None)
                continue

            if dict_concept_id:
                faiss_ids = [c["concept_id"] for c in faiss_cands]
                dict_in_faiss_rank = next(
                    (i for i, cid in enumerate(faiss_ids) if cid == dict_concept_id), None
                )
                if dict_in_faiss_rank is not None:
                    boosted = max(
                        faiss_cands[dict_in_faiss_rank]["score"],
                        0.95 if is_tier2 else faiss_cands[dict_in_faiss_rank]["score"],
                    )
                    merged = [{**faiss_cands[dict_in_faiss_rank], "score": round(boosted, 4)}]
                    merged += [c for i, c in enumerate(faiss_cands) if i != dict_in_faiss_rank]
                else:
                    dict_score = 0.95 if is_tier2 else fuzzy_cands[0]["score"]
                    inject = {"concept_id": dict_concept_id, "score": dict_score,
                              "tier": 2 if is_tier2 else 3}
                    merged = [inject] + faiss_cands
            else:
                merged = faiss_cands
                dict_in_faiss_rank = None

            # Step 3 — string-similarity boost: prefer candidates whose
            # preferred term closely matches the span over purely semantic
            # FAISS neighbours with different (but close) SNOMED codes.
            merged = _boost_by_string_match(span_text, merged)

            should_rerank = False
            if use_llm_reranking and llm_client is not None and len(merged) > 1:
                if not is_tier2:
                    should_rerank = True
                elif dict_in_faiss_rank is None or dict_in_faiss_rank >= TIER2_FAISS_AGREE_RANK:
                    should_rerank = True

            merged_list.append({
                "start": start, "end": end, "span": span_text,
                "merged": merged, "should_rerank": should_rerank,
            })
            if should_rerank:
                needs_rerank.append(len(merged_list) - 1)

        if needs_rerank and use_llm_reranking and llm_client is not None:
            for batch_start in range(0, len(needs_rerank), RERANK_BATCH):
                batch_indices = needs_rerank[batch_start: batch_start + RERANK_BATCH]
                items = [
                    {
                        "span": merged_list[i]["span"],
                        "context": _get_context(note_text, merged_list[i]["start"], merged_list[i]["end"]),
                        "candidates": merged_list[i]["merged"],
                    }
                    for i in batch_indices
                ]
                best_ids = _llm_rerank_batch(items, llm_client, model=llm_model)
                for i, best_id in zip(batch_indices, best_ids):
                    if best_id:
                        cands = merged_list[i]["merged"]
                        reranked = [c for c in cands if c["concept_id"] == best_id]
                        reranked += [c for c in cands if c["concept_id"] != best_id]
                        merged_list[i]["merged"] = reranked

        for entry in merged_list:
            if entry is None:
                continue
            faiss_preds.append({
                "start": entry["start"],
                "end": entry["end"],
                "span": entry["span"],
                "candidates": entry["merged"][:top_k_output],
            })

    predictions = tier1_preds + faiss_preds

    if resolve_overlaps:
        predictions = _resolve_overlaps(predictions)

    return predictions


if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Method 4: Tiered dictionary + FAISS + selective LLM reranking")
    parser.add_argument("--note",        required=True, help="Path to plain-text note file")
    parser.add_argument("--dictionary",  default=str(Path(__file__).parent / "results" / "dictionary.json"))
    parser.add_argument("--index-dir",   default=str(Path(__file__).parent / "results" / "rag_index"))
    parser.add_argument("--llm-model",   default=LLM_MODEL)
    parser.add_argument("--no-rerank",   action="store_true")
    args = parser.parse_args()

    from Link.method1_dictionary import load_dictionary
    from Link.method3_llm_rag import RAGIndex, OllamaClient

    dictionary = load_dictionary(args.dictionary)
    rag_index  = RAGIndex(args.index_dir)
    client     = None if args.no_rerank else OllamaClient(model_name=args.llm_model)

    with open(args.note, encoding="utf-8") as f:
        text = f.read()

    preds = predict(text, dictionary, rag_index, client, use_llm_reranking=not args.no_rerank)
    for p in preds:
        top1 = p["candidates"][0] if p["candidates"] else {}
        print(f"[{p['start']:5d}-{p['end']:5d}] {p['span']!r:40s} → {top1.get('concept_id','')} ({top1.get('score',0):.3f})")
