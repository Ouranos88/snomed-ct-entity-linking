"""
evaluate_cas_cliniques.py — Compare method results against gold answers.

Matching rule: an expected concept is "found" if its SNOMED ID appears
in the top-5 candidates of ANY predicted span for that case / method.
"""

import json
import re
from pathlib import Path

ANSWER_FILE  = Path(__file__).parent / "cas_cliniques_answer.txt"
RESULTS_FILE = Path(__file__).parent / "cas_cliniques_results.json"

# ── Answer file parser ────────────────────────────────────────────────────────

def _answer_case_id(header: str) -> str | None:
    h = header.strip()
    # OPHTALMOLOGIE (EN) — CASE 84  →  OPHTALMO_EN_84
    m = re.search(r'CASE\s+(\w+)', h, re.IGNORECASE)
    if m and 'EN' in h.upper():
        return f"OPHTALMO_EN_{m.group(1)}"
    # OPHTALMOLOGIE (FR) — CAS CLINIQUE 3  →  OPHTALMO_FR_3
    m = re.search(r'CAS CLINIQUE\s+(\w+)', h, re.IGNORECASE)
    if m and 'OPHTALMOLOGIE' in h.upper():
        return f"OPHTALMO_FR_{m.group(1)}"
    # CARDIOLOGIE — CAS CLINIQUE 5  →  CARDIO_5
    if m and 'CARDIOLOGIE' in h.upper():
        return f"CARDIO_{m.group(1)}"
    return None


def parse_answers(path: Path) -> dict[str, list[dict]]:
    """Returns {case_id: [{'label': ..., 'concept_id': ..., 'preferred': ...}]}"""
    text = path.read_text(encoding="utf-8")
    ANY_SEP = re.compile(r'^[=\-]{20,}$')
    ENTITY_RE = re.compile(r'^(.+?)\s*\|\s*(\d+)\s*\|\s*(.+)$')

    answers: dict[str, list[dict]] = {}
    current_id: str | None = None
    in_case_header = False

    for raw in text.splitlines():
        line = raw.strip()
        if ANY_SEP.match(line):
            in_case_header = not in_case_header
            continue
        if in_case_header:
            cid = _answer_case_id(line)
            if cid:
                current_id = cid
                answers.setdefault(current_id, [])
            continue
        if current_id and not line:
            continue
        if current_id:
            m = ENTITY_RE.match(line)
            if m:
                answers[current_id].append({
                    "label":     m.group(1).strip(),
                    "concept_id": m.group(2).strip(),
                    "preferred": m.group(3).strip(),
                })

    return answers


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(answers: dict, results: dict, top_k: int = 5):
    """
    For each case × method, count how many expected concepts appear in top-k
    candidates of any predicted span.

    Returns a nested dict: {case_id: {method: {'hits': int, 'total': int}}}
    """
    methods = ["M1", "M2", "M3", "M4"]
    out: dict = {}

    for case_id, expected in answers.items():
        if case_id not in results:
            continue
        expected_ids = {e["concept_id"] for e in expected}
        case_res = results[case_id]["methods"]
        out[case_id] = {}

        for method in methods:
            preds = case_res.get(method)
            if preds is None:
                out[case_id][method] = None
                continue

            # Collect all concept IDs that appear in top-k across all spans
            found_ids: set[str] = set()
            for pred in preds:
                for cand in (pred.get("candidates") or [])[:top_k]:
                    found_ids.add(cand["concept_id"])

            hits = len(expected_ids & found_ids)
            out[case_id][method] = {"hits": hits, "total": len(expected_ids),
                                    "found": expected_ids & found_ids,
                                    "missed": expected_ids - found_ids}

    return out


# ── Table rendering ───────────────────────────────────────────────────────────

def print_table(eval_res: dict, answers: dict):
    methods = ["M1", "M2", "M3", "M4"]

    # Per-case table
    print("\n" + "=" * 80)
    print("PERFORMANCE PER CASE  (hits / total expected concepts, top-5 match)")
    print("=" * 80)
    header = f"{'Case':<22} {'Expected':>8}  {'M1':>10}  {'M2':>10}  {'M3':>10}  {'M4':>10}"
    print(header)
    print("-" * 80)

    totals = {m: {"hits": 0, "total": 0} for m in methods}

    for case_id in sorted(eval_res, key=lambda x: (x.split("_")[0], int(re.search(r'\d+$', x).group()))):
        row = eval_res[case_id]
        n_expected = len(answers[case_id])
        cells = []
        for m in methods:
            r = row.get(m)
            if r is None:
                cells.append(f"{'n/a':>10}")
            else:
                pct = r["hits"] / r["total"] * 100 if r["total"] else 0
                cells.append(f"{r['hits']:>3}/{r['total']:<3} {pct:>4.0f}%")
                totals[m]["hits"]  += r["hits"]
                totals[m]["total"] += r["total"]
        print(f"  {case_id:<20} {n_expected:>8}  {'  '.join(cells)}")

    print("-" * 80)
    total_expected = sum(len(v) for v in answers.values())
    summary_cells = []
    for m in methods:
        t = totals[m]
        if t["total"] == 0:
            summary_cells.append(f"{'n/a':>10}")
        else:
            pct = t["hits"] / t["total"] * 100
            summary_cells.append(f"{t['hits']:>3}/{t['total']:<3} {pct:>4.0f}%")
    print(f"  {'TOTAL':<20} {total_expected:>8}  {'  '.join(summary_cells)}")
    print("=" * 80)

    # Section subtotals
    print("\n" + "=" * 80)
    print("SUBTOTALS BY SECTION")
    print("=" * 80)
    print(f"{'Section':<22} {'Expected':>8}  {'M1':>10}  {'M2':>10}  {'M3':>10}  {'M4':>10}")
    print("-" * 80)

    sections = {}
    for case_id, row in eval_res.items():
        section = "OPHTALMO_EN" if case_id.startswith("OPHTALMO_EN") \
             else "OPHTALMO_FR" if case_id.startswith("OPHTALMO_FR") \
             else "CARDIO"
        sections.setdefault(section, {m: {"hits": 0, "total": 0} for m in methods})
        for m in methods:
            r = row.get(m)
            if r:
                sections[section][m]["hits"]  += r["hits"]
                sections[section][m]["total"] += r["total"]

    section_expected = {}
    for case_id, expected in answers.items():
        sec = "OPHTALMO_EN" if case_id.startswith("OPHTALMO_EN") \
         else "OPHTALMO_FR" if case_id.startswith("OPHTALMO_FR") \
         else "CARDIO"
        section_expected[sec] = section_expected.get(sec, 0) + len(expected)

    for sec in ["OPHTALMO_EN", "OPHTALMO_FR", "CARDIO"]:
        if sec not in sections:
            continue
        cells = []
        for m in methods:
            t = sections[sec][m]
            pct = t["hits"] / t["total"] * 100 if t["total"] else 0
            cells.append(f"{t['hits']:>3}/{t['total']:<3} {pct:>4.0f}%")
        print(f"  {sec:<20} {section_expected.get(sec,0):>8}  {'  '.join(cells)}")
    print("=" * 80)

    # Missed concepts analysis for M4 (best LLM method)
    print("\n" + "=" * 80)
    print("COMMONLY MISSED CONCEPTS (concept found by NO method)")
    print("=" * 80)
    missed_by_all: dict[str, int] = {}
    for case_id, row in eval_res.items():
        missed_all = None
        for m in methods:
            r = row.get(m)
            if r is None:
                continue
            if missed_all is None:
                missed_all = set(r["missed"])
            else:
                missed_all &= r["missed"]
        if missed_all:
            for cid in missed_all:
                label = next((e["preferred"] for e in answers[case_id]
                              if e["concept_id"] == cid), cid)
                missed_by_all[label] = missed_by_all.get(label, 0) + 1

    for label, cnt in sorted(missed_by_all.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cnt:2d}x  {label}")
    print("=" * 80)


def main():
    print("Parsing answer file ...")
    answers = parse_answers(ANSWER_FILE)
    print(f"  {len(answers)} cases with gold annotations  "
          f"({sum(len(v) for v in answers.values())} total expected concepts)")

    print("Loading results ...")
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    print(f"  {len(results)} cases in results file")

    print("Evaluating (top-5 concept match) ...\n")
    eval_res = evaluate(answers, results, top_k=5)

    print_table(eval_res, answers)


if __name__ == "__main__":
    main()
