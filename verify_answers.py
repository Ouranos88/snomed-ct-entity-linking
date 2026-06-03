"""
verify_answers.py — Verify every SNOMED concept ID in cas_cliniques_answer.txt
against the actual RF2, and rewrite the file with corrected IDs.

For each entry:
  1. Check if the concept ID exists and is active in RF2
  2. If yes, check if the preferred term (FSN) roughly matches the label
  3. If the ID is wrong/missing, search RF2 for the best matching concept
"""

import csv
import re
import unicodedata
from pathlib import Path
from rapidfuzz import fuzz, process

RF2_EN = Path("SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20260301.txt")
ANSWER_FILE = Path("cas_cliniques_answer.txt")
OUT_FILE    = Path("cas_cliniques_answer_verified.txt")

SNOMED_FSN_TYPE = "900000000000003001"


def norm(text: str) -> str:
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae")
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_rf2(path: Path):
    """Returns:
      concept2fsn: {concept_id -> FSN string}
      concept2terms: {concept_id -> [all active terms]}
      term2concepts: {norm(term) -> [concept_id, ...]}
    """
    concept2fsn = {}
    concept2terms = {}
    term2concepts = {}
    csv.field_size_limit(10_000_000)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["active"] != "1":
                continue
            cid  = row["conceptId"]
            term = row["term"]
            tid  = row["typeId"]
            concept2terms.setdefault(cid, []).append(term)
            if tid == SNOMED_FSN_TYPE:
                concept2fsn[cid] = term
            key = norm(term)
            term2concepts.setdefault(key, []).append(cid)
    return concept2fsn, concept2terms, term2concepts


def find_best_concept(label: str, term2concepts, concept2fsn, concept2terms):
    """Search RF2 for the concept whose terms best match the label."""
    key = norm(label)
    # 1. Exact match
    if key in term2concepts:
        cids = term2concepts[key]
        # prefer the one whose FSN contains the label text most closely
        best = sorted(cids, key=lambda c: -fuzz.ratio(norm(concept2fsn.get(c, "")), key))[0]
        return best, concept2fsn.get(best, ""), "exact"

    # 2. Fuzzy match over all keys (top-1)
    keys_list = list(term2concepts.keys())
    result = process.extractOne(key, keys_list, scorer=fuzz.token_sort_ratio, score_cutoff=70)
    if result:
        matched_key, score, _ = result
        cids = term2concepts[matched_key]
        best = sorted(cids, key=lambda c: -fuzz.ratio(norm(concept2fsn.get(c, "")), key))[0]
        return best, concept2fsn.get(best, ""), f"fuzzy({score})"

    return None, None, "not_found"


def parse_and_verify():
    print(f"Loading RF2 from {RF2_EN.name} ...")
    concept2fsn, concept2terms, term2concepts = load_rf2(RF2_EN)
    print(f"  {len(concept2fsn):,} active concepts, {len(term2concepts):,} unique normalised terms")

    text = ANSWER_FILE.read_text(encoding="utf-8")
    lines = text.splitlines()

    ENTITY_RE = re.compile(r'^(.+?)\s*\|\s*(\d+)\s*\|\s*(.+)$')
    changes = []
    out_lines = []

    for line in lines:
        m = ENTITY_RE.match(line.strip())
        if not m:
            out_lines.append(line)
            continue

        label    = m.group(1).strip()
        cid      = m.group(2).strip()
        pref     = m.group(3).strip()

        # Check if concept ID exists and is active
        if cid in concept2fsn:
            fsn = concept2fsn[cid]
            # Check if the FSN roughly matches the label (token sort ratio > 50)
            similarity = fuzz.token_sort_ratio(norm(fsn), norm(label))
            if similarity >= 50:
                # Good — ID is correct, just normalise the preferred term to RF2 FSN
                if pref != fsn:
                    changes.append(f"  UPDATED term: [{cid}] '{pref}' -> '{fsn}'")
                out_lines.append(f"{label:<52}| {cid:<12}| {fsn}")
                continue
            else:
                # ID exists but FSN doesn't match the label — likely wrong ID
                new_cid, new_fsn, method = find_best_concept(label, term2concepts, concept2fsn, concept2terms)
                if new_cid and new_cid != cid:
                    changes.append(f"  CORRECTED: '{label}' | {cid} '{concept2fsn[cid]}' -> {new_cid} '{new_fsn}' ({method})")
                    out_lines.append(f"{label:<52}| {new_cid:<12}| {new_fsn}")
                else:
                    changes.append(f"  KEPT (no better match): '{label}' | {cid} '{fsn}' (sim={similarity})")
                    out_lines.append(f"{label:<52}| {cid:<12}| {fsn}")
        else:
            # Concept ID not found in RF2 — search by label
            new_cid, new_fsn, method = find_best_concept(label, term2concepts, concept2fsn, concept2terms)
            if new_cid:
                changes.append(f"  FIXED missing ID: '{label}' | {cid} -> {new_cid} '{new_fsn}' ({method})")
                out_lines.append(f"{label:<52}| {new_cid:<12}| {new_fsn}")
            else:
                changes.append(f"  NOT FOUND: '{label}' | {cid} '{pref}'")
                out_lines.append(line)

    OUT_FILE.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nChanges ({len(changes)}):")
    for c in changes:
        print(c)
    print(f"\nVerified file written to {OUT_FILE}")


if __name__ == "__main__":
    parse_and_verify()
