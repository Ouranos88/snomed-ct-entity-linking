"""
run_cas_cliniques.py — Run all 4 methods on every clinical case in cas_cliniques.txt.

Output:
    cas_cliniques_results.json   — full structured results (all candidates)
    cas_cliniques_results.txt    — human-readable summary (top-3 per span per method)
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent / "Link" / "results"
INPUT_FILE  = Path(__file__).parent / "cas_cliniques.txt"
OUT_JSON    = Path(__file__).parent / "cas_cliniques_results.json"
OUT_TXT     = Path(__file__).parent / "cas_cliniques_results.txt"

# Section name → short prefix used in case IDs
_SECTION_MAP = {
    "ophtalmologie (english)": "OPHTALMO_EN",
    "ophtalmologie":           "OPHTALMO_FR",
    "cardiologie":             "CARDIO",
}

_CASE_RE    = re.compile(r'^(?:CAS CLINIQUE|CASE)\s+(\w+)', re.IGNORECASE)
_ANY_SEP    = re.compile(r'^[=\-]{20,}$')


def parse_cases(path: Path) -> dict[str, str]:
    """Parse all sections and cases from the input file.

    Returns an ordered dict {case_id: text}, where case_id is like
    OPHTALMO_EN_84, OPHTALMO_FR_1, CARDIO_5, etc.
    """
    text = path.read_text(encoding="utf-8")

    # Split into blocks separated by any long ===/ --- line.
    # Each block is a chunk of text between two separator lines.
    current_block_lines: list[str] = []
    blocks: list[str] = []
    for raw_line in text.splitlines():
        if _ANY_SEP.match(raw_line.strip()):
            block = "\n".join(current_block_lines).strip()
            if block:
                blocks.append(block)
            current_block_lines = []
        else:
            current_block_lines.append(raw_line)
    tail = "\n".join(current_block_lines).strip()
    if tail:
        blocks.append(tail)

    cases: dict[str, str] = {}
    current_section = "UNKNOWN"
    current_case_id: str | None = None

    for block in blocks:
        non_empty = [l for l in block.split("\n") if l.strip()]
        if not non_empty:
            continue
        first = non_empty[0].strip()

        # ── Section header detection ──────────────────────────────────────
        # A section header block contains "CAS CLINIQUES -" and a specialty
        block_lower = block.lower()
        matched_section = False
        if "cas cliniques" in block_lower or "source" in block_lower:
            for key, prefix in _SECTION_MAP.items():
                if key in block_lower:
                    current_section = prefix
                    current_case_id = None
                    matched_section = True
                    break
        if matched_section:
            continue

        # ── Case header detection ─────────────────────────────────────────
        # A case header block is short (≤ 2 non-empty lines) and its first
        # line matches the case pattern.
        m = _CASE_RE.match(first)
        if m and len(non_empty) <= 2:
            num = m.group(1)
            candidate = f"{current_section}_{num}"
            # Guarantee uniqueness within the same section prefix
            unique = candidate
            idx = 2
            while unique in cases:
                unique = f"{candidate}_{idx}"
                idx += 1
            current_case_id = unique
            continue

        # ── Case content ──────────────────────────────────────────────────
        if current_case_id is not None:
            if current_case_id in cases:
                cases[current_case_id] += "\n\n" + block
            else:
                cases[current_case_id] = block

    return cases


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_m1():
    from Link.method1_dictionary import load_dictionary
    return load_dictionary(str(RESULTS_DIR / "dictionary.json"))

def load_m2_index():
    from Link.method2_sapbert import SapBERTIndex
    return SapBERTIndex(str(RESULTS_DIR / "sapbert_index"))

def load_rag_index():
    from Link.method3_llm_rag import RAGIndex
    return RAGIndex(str(RESULTS_DIR / "rag_index"))

def load_llm():
    from Link.method3_llm_rag import OllamaClient
    return OllamaClient(model_name="llama3.1:8b")


# ── Per-method runners ────────────────────────────────────────────────────────

def run_m1(text, d):
    from Link.method1_dictionary import predict
    return predict(text, d)

def run_m2(text, idx, d):
    from Link.method2_sapbert import predict
    return predict(text, idx, dictionary=d)

def run_m3(text, rag, llm):
    from Link.method3_llm_rag import predict
    return predict(text, rag, llm, use_llm_reranking=True)

def run_m4(text, d, rag, llm):
    from Link.method4_hybrid import predict
    return predict(text, d, rag, llm, use_llm_reranking=True)


# ── Formatting helper ─────────────────────────────────────────────────────────

def fmt_preds(preds, max_spans=25, max_cands=3):
    lines = []
    for i, p in enumerate(preds):
        if i >= max_spans:
            lines.append(f"  ... (+{len(preds) - i} more spans)")
            break
        cands = p.get("candidates", [])[:max_cands]
        cand_str = "  |  ".join(
            f"{c['concept_id']} {(c.get('description') or '')[:40]!r} ({c['score']:.2f})"
            for c in cands
        )
        lines.append(f"  [{p['start']:4d}-{p['end']:4d}] \"{p['span']}\"")
        if cand_str:
            lines.append(f"    {cand_str}")
    return "\n".join(lines) if lines else "  (no predictions)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Parsing {INPUT_FILE.name} ...")
    cases = parse_cases(INPUT_FILE)
    print(f"  {len(cases)} clinical cases found.")

    # Print a summary of what was parsed
    by_section: dict[str, list[str]] = {}
    for cid in cases:
        prefix = cid.rsplit("_", 1)[0] if "_" in cid else cid
        by_section.setdefault(prefix, []).append(cid)
    for section, ids in by_section.items():
        print(f"  {section}: {len(ids)} cases  ({ids[0]} … {ids[-1]})")
    print()

    print("Loading Method 1 dictionary ...")
    d = load_m1()
    print("Loading Method 2 SapBERT index ...")
    m2_idx = load_m2_index()

    rag, llm = None, None
    try:
        print("Loading RAG index (Methods 3 & 4) ...")
        rag = load_rag_index()
        print("Connecting to Ollama (llama3.1:8b) ...")
        llm = load_llm()
        print("  Ollama ready.\n")
    except Exception as e:
        print(f"  Warning: Ollama/RAG not available ({e}). Methods 3 & 4 skipped.\n")

    all_results: dict = {}

    n_cases = len(cases)
    t_start = time.time()

    for idx, (case_id, text) in enumerate(cases.items(), 1):
        filled = int(30 * idx / n_cases)
        bar    = "#" * filled + "-" * (30 - filled)
        elapsed = time.time() - t_start
        eta     = (elapsed / idx) * (n_cases - idx) if idx > 1 else 0
        print(f"\n[{bar}] {idx:2d}/{n_cases}  ETA {eta/60:.0f}m{eta%60:02.0f}s")
        print(f"  {case_id}  ({len(text)} chars)")
        case_res: dict = {"text": text, "methods": {}}

        # M1
        print(f"       M1 ...", end=" ", flush=True)
        t0 = time.time()
        try:
            preds = run_m1(text, d)
            case_res["methods"]["M1"] = preds
            print(f"{len(preds):3d} spans  ({time.time()-t0:.1f}s)")
        except Exception as e:
            case_res["methods"]["M1"] = []
            print(f"ERROR: {e}")

        # M2
        print(f"       M2 ...", end=" ", flush=True)
        t0 = time.time()
        try:
            preds = run_m2(text, m2_idx, d)
            case_res["methods"]["M2"] = preds
            print(f"{len(preds):3d} spans  ({time.time()-t0:.1f}s)")
        except Exception as e:
            case_res["methods"]["M2"] = []
            print(f"ERROR: {e}")

        if rag is not None and llm is not None:
            # M3
            print(f"       M3 ...", end=" ", flush=True)
            t0 = time.time()
            try:
                preds = run_m3(text, rag, llm)
                case_res["methods"]["M3"] = preds
                print(f"{len(preds):3d} spans  ({time.time()-t0:.1f}s)")
            except Exception as e:
                case_res["methods"]["M3"] = []
                print(f"ERROR: {e}")

            # M4
            print(f"       M4 ...", end=" ", flush=True)
            t0 = time.time()
            try:
                preds = run_m4(text, d, rag, llm)
                case_res["methods"]["M4"] = preds
                print(f"{len(preds):3d} spans  ({time.time()-t0:.1f}s)")
            except Exception as e:
                case_res["methods"]["M4"] = []
                print(f"ERROR: {e}")
        else:
            case_res["methods"]["M3"] = None
            case_res["methods"]["M4"] = None

        all_results[case_id] = case_res

    # ── Save JSON ─────────────────────────────────────────────────────────
    print(f"\nSaving {OUT_JSON.name} ...")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Save human-readable TXT ───────────────────────────────────────────
    print(f"Saving {OUT_TXT.name} ...")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for case_id, cr in all_results.items():
            f.write(f"{'='*70}\n{case_id}\n{'='*70}\n")
            f.write(cr["text"] + "\n\n")
            for method, preds in cr["methods"].items():
                f.write(f"--- {method} ---\n")
                if preds is None:
                    f.write("  (not run — Ollama/RAG unavailable)\n")
                elif not preds:
                    f.write("  (no predictions)\n")
                else:
                    f.write(fmt_preds(preds) + "\n")
                f.write("\n")
            f.write("\n")

    print(f"\nDone. {len(all_results)} cases processed.")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_TXT}")


if __name__ == "__main__":
    main()
