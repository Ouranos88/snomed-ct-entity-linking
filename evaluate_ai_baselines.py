"""
Evaluate claude_sonnet46_results.txt and chatgpt35_results.txt against cas_cliniques_answer.txt.
"""
import re
from pathlib import Path
from evaluate_cas_cliniques import parse_answers

ANSWER = Path("cas_cliniques_answer.txt")
CLAUDE_FILE = Path("claude_sonnet46_results.txt")
GPT_FILE    = Path("chatgpt35_results.txt")

ENTITY_RE = re.compile(r'^.+?\|\s*(\d+)\s*\|')

# ── Parse AI result files ──────────────────────────────────────────────────────

def parse_ai_results(path: Path) -> dict[str, set[str]]:
    """Returns {case_id: set_of_concept_ids}.
    Handles two formats:
      Claude: **CAS CLINIQUE N (Section)** headers, entries inside ``` blocks
      GPT:    'cas clinique N [cardiologie]' plain headers, entries on plain lines
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    CL_HDR = re.compile(r'^\*\*(?:CAS CLINIQUE|CASE)\s+(\w+).*?\((Ophtalmologie|Cardiologie)\)|^\*\*CASE\s+(\w+)', re.IGNORECASE)
    GP_HDR = re.compile(r'^cas clinique\s+(\w+)\s*(cardiologie)?$', re.IGNORECASE)

    results: dict[str, set[str]] = {}
    current: str | None = None
    in_code_block = False

    for raw in lines:
        line = raw.strip()

        # Toggle code block state (Claude format)
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue

        # Claude-style bold header — only outside code blocks
        if not in_code_block and line.startswith("**"):
            m = CL_HDR.match(line)
            if m:
                # Groups: (1=num ophtho/cardio, 2=sect, 3=num case84)
                if m.group(3):  # CASE 84 pattern
                    current = "OPHTALMO_EN_84"
                else:
                    num  = m.group(1)
                    sect = (m.group(2) or "").upper()
                    if "CARDIO" in sect:
                        current = f"CARDIO_{num}"
                    else:
                        current = f"OPHTALMO_FR_{num}"
                results.setdefault(current, set())
                continue

        # GPT-style plain header — only outside code blocks
        if not in_code_block:
            m2 = GP_HDR.match(line)
            if m2:
                num  = m2.group(1)
                sect = (m2.group(2) or "").strip().upper()
                if num == "84":
                    current = "OPHTALMO_EN_84"
                elif "CARDIO" in sect:
                    current = f"CARDIO_{num}"
                else:
                    current = f"OPHTALMO_FR_{num}"
                results.setdefault(current, set())
                continue

        if current is None:
            continue

        m3 = ENTITY_RE.match(line)
        if m3:
            cid = m3.group(1).strip()
            if cid and cid not in ("0", "UNKNOWN"):
                results[current].add(cid)

    return results


def evaluate_ai(answers: dict, ai_results: dict) -> dict:
    out = {}
    for case_id, expected in answers.items():
        expected_ids = {e["concept_id"] for e in expected}
        found = ai_results.get(case_id, set())
        hits = len(expected_ids & found)
        out[case_id] = {"hits": hits, "total": len(expected_ids),
                        "found": expected_ids & found,
                        "missed": expected_ids - found}
    return out


def section_of(case_id: str) -> str:
    if case_id.startswith("OPHTALMO_EN"):  return "EN"
    if case_id.startswith("OPHTALMO_FR"):  return "FR_OPHTALMO"
    return "FR_CARDIO"


answers = parse_answers(ANSWER)
print(f"Gold: {len(answers)} cases, {sum(len(v) for v in answers.values())} annotations\n")

for label, path in [("Claude Sonnet 4.6", CLAUDE_FILE), ("GPT-3.5", GPT_FILE)]:
    ai = parse_ai_results(path)
    ev = evaluate_ai(answers, ai)

    totals = {"EN": [0,0], "FR_OPHTALMO": [0,0], "FR_CARDIO": [0,0], "ALL": [0,0]}
    for case_id, r in ev.items():
        sec = section_of(case_id)
        totals[sec][0] += r["hits"]; totals[sec][1] += r["total"]
        totals["ALL"][0] += r["hits"]; totals["ALL"][1] += r["total"]

    print(f"=== {label} ===")
    for sec, (h, t) in totals.items():
        pct = h/t*100 if t else 0
        print(f"  {sec:<15} {h:3d}/{t:<3d} = {pct:.0f}%")
    print()
