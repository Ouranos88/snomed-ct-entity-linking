"""
method1_dictionary.py — Dictionary + fuzzy matching baseline.

3-tier lookup:
  Tier 1 (score 1.00) : exact match against training frequency dictionary
  Tier 2 (score 0.95) : exact match against SNOMED RF2 synonym dictionary
  Tier 3 (score ≤0.85): rapidfuzz token_sort_ratio fuzzy match against merged dict
"""

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rapidfuzz import fuzz, process as rfprocess
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

from Link.preprocessor import extract_candidate_spans, is_negated, normalize_text

TOP_K = 5
FUZZY_THRESHOLD = 85
FUZZY_SCORE_SCALE = 0.85


def _norm_key(text: str) -> str:
    # Expand French/Latin ligatures so œdème and oedème hash to the same key
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_from_annotations(annotations_csv: str) -> Dict[str, Counter]:
    mapping: Dict[str, Counter] = defaultdict(Counter)
    with open(annotations_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("annotation_type", "train").lower() != "train":
                continue
            span = row.get("span", "").strip()
            cid  = row.get("concept_id", "").strip()
            if span and cid:
                mapping[_norm_key(span)][cid] += 1
    return dict(mapping)


def build_from_snomed_rf2(
    description_file: str,
    language_code: Optional[str] = None,
) -> Dict[str, str]:
    """
    Parse a SNOMED CT RF2 description snapshot (tab-separated).
    language_code: if given (e.g. 'fr'), only load that languageCode.
    """
    csv.field_size_limit(10_000_000)
    mapping: Dict[str, str] = {}
    total_rows = active_rows = 0
    with open(description_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            total_rows += 1
            if total_rows % 200_000 == 0:
                print(f"    {total_rows:,} rows read, {active_rows:,} active, {len(mapping):,} unique keys so far...")
            if row.get("active", "0") != "1":
                continue
            if language_code and row.get("languageCode", "") != language_code:
                continue
            active_rows += 1
            term = row.get("term", "").strip()
            cid  = row.get("conceptId", "").strip()
            if term and cid:
                key = _norm_key(term)
                if key not in mapping:
                    mapping[key] = cid
    print(f"    Done: {total_rows:,} rows read, {active_rows:,} active, {len(mapping):,} unique keys")
    return mapping


def build_dictionary(
    annotations_csv: str,
    snomed_rf2_file: Optional[str] = None,
    snomed_rf2_fr: Optional[str] = None,
) -> dict:
    print("Building Tier 1 from training annotations ...")
    tier1_raw = build_from_annotations(annotations_csv)
    tier1 = {k: sorted(c.items(), key=lambda x: -x[1]) for k, c in tier1_raw.items()}
    print(f"  Tier 1: {len(tier1)} unique normalised spans")

    tier2: Dict[str, str] = {}
    if snomed_rf2_file and Path(snomed_rf2_file).exists():
        print("Building Tier 2 from SNOMED RF2 English descriptions ...")
        tier2 = {k: v for k, v in build_from_snomed_rf2(snomed_rf2_file).items() if k not in tier1}
        print(f"  Tier 2 (EN): {len(tier2)} additional SNOMED synonyms")
    else:
        print("  Tier 2 English skipped (no SNOMED RF2 file provided or file not found)")

    if snomed_rf2_fr and Path(snomed_rf2_fr).exists():
        print("Building Tier 2 French extension ...")
        added = 0
        for k, v in build_from_snomed_rf2(snomed_rf2_fr, language_code="fr").items():
            if k not in tier1 and k not in tier2:
                tier2[k] = v
                added += 1
        print(f"  Tier 2 (FR): {added} additional French SNOMED synonyms added")
    elif snomed_rf2_fr:
        print(f"  Tier 2 French skipped (file not found: {snomed_rf2_fr})")

    return {"tier1": tier1, "tier2": tier2}


def save_dictionary(dictionary: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False)
    print(f"Dictionary saved to {output_path}")


def load_dictionary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _lookup_exact(norm_key: str, dictionary: dict) -> Optional[List[dict]]:
    if norm_key in dictionary["tier1"]:
        entries = dictionary["tier1"][norm_key]
        total = sum(c for _, c in entries)
        return [{"concept_id": cid, "score": round(cnt / total, 4), "tier": 1}
                for cid, cnt in entries[:TOP_K]]
    if norm_key in dictionary["tier2"]:
        return [{"concept_id": dictionary["tier2"][norm_key], "score": 0.95, "tier": 2}]
    return None


def _build_fuzzy_keys(dictionary: dict) -> List[str]:
    # Tier 1 only — Tier 2 has 800k+ entries; fuzzy against it would be too slow
    return list(dictionary["tier1"].keys())


def _lookup_fuzzy(norm_key: str, dictionary: dict, fuzzy_keys: List[str]) -> Optional[List[dict]]:
    if not _RAPIDFUZZ:
        return None
    results = rfprocess.extract(
        norm_key, fuzzy_keys,
        scorer=fuzz.token_sort_ratio, limit=TOP_K, score_cutoff=FUZZY_THRESHOLD,
    )
    if not results:
        return None
    candidates = []
    seen = set()
    for matched_key, score, _idx in results:
        raw_score = score / 100.0 * FUZZY_SCORE_SCALE
        if matched_key in dictionary["tier1"]:
            for cid, _ in dictionary["tier1"][matched_key][:1]:
                if cid not in seen:
                    candidates.append({"concept_id": cid, "score": round(raw_score, 4), "tier": 3})
                    seen.add(cid)
        elif matched_key in dictionary["tier2"]:
            cid = dictionary["tier2"][matched_key]
            if cid not in seen:
                candidates.append({"concept_id": cid, "score": round(raw_score, 4), "tier": 3})
                seen.add(cid)
        if len(candidates) >= TOP_K:
            break
    return candidates if candidates else None


def _resolve_overlaps(predictions: List[dict]) -> List[dict]:
    preds = sorted(predictions,
                   key=lambda p: (p["start"], -p["candidates"][0]["score"], -(p["end"] - p["start"])))
    kept = []
    last_end = -1
    for p in preds:
        if p["start"] >= last_end:
            kept.append(p)
            last_end = p["end"]
        elif kept and p["candidates"][0]["score"] > kept[-1]["candidates"][0]["score"]:
            kept[-1] = p
            last_end = p["end"]
    return kept


def predict(
    note_text: str,
    dictionary: dict,
    resolve_overlaps: bool = True,
    skip_negated: bool = True,
    max_tokens: int = 8,
) -> List[dict]:
    candidate_spans = extract_candidate_spans(note_text, max_tokens=max_tokens)
    fuzzy_keys: Optional[List[str]] = None
    predictions: List[dict] = []

    for start, end, span_text in candidate_spans:
        if skip_negated and is_negated(note_text, start):
            continue
        norm = _norm_key(span_text)
        candidates = _lookup_exact(norm, dictionary)
        if candidates is None:
            if fuzzy_keys is None:
                fuzzy_keys = _build_fuzzy_keys(dictionary)
            candidates = _lookup_fuzzy(norm, dictionary, fuzzy_keys)
        if candidates:
            predictions.append({"start": start, "end": end, "span": span_text, "candidates": candidates})

    if resolve_overlaps:
        predictions = _resolve_overlaps(predictions)
    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Method 1: Dictionary + fuzzy matching")
    sub = parser.add_subparsers(dest="cmd")

    build_p = sub.add_parser("build")
    build_p.add_argument("--annotations", required=True)
    build_p.add_argument("--snomed-rf2", default=None)
    build_p.add_argument("--snomed-rf2-fr", default=None)
    _default_dict = str(Path(__file__).parent / "results" / "dictionary.json")
    build_p.add_argument("--output", default=_default_dict)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--note", required=True)
    pred_p.add_argument("--dictionary", default=_default_dict)

    args = parser.parse_args()

    if args.cmd == "build":
        d = build_dictionary(args.annotations, args.snomed_rf2, args.snomed_rf2_fr)
        save_dictionary(d, args.output)
    elif args.cmd == "predict":
        d = load_dictionary(args.dictionary)
        with open(args.note, encoding="utf-8") as f:
            text = f.read()
        for p in predict(text, d):
            top = p["candidates"][0]
            print(f"  [{p['start']:5d}:{p['end']:5d}] {p['span']!r:40s}  "
                  f"→ {top['concept_id']}  (score={top['score']:.3f}, tier={top['tier']})")
    else:
        parser.print_help()
