"""
annotate_pdf.py — Run any pipeline method on a PDF and display SNOMED CT annotations.

Usage
-----
    python annotate_pdf.py --pdf my_note.pdf --method 1
    python annotate_pdf.py --pdf my_note.pdf --method 4 --no-rerank
    python annotate_pdf.py --pdf my_note.pdf --method 4 --llm-model llama3.1:8b

Output (per detected span)
--------------------------
    [  42-  55] "colon cancer"
      1. 363346000  Malignant neoplastic disease          (score: 1.000)
      2. 109841000  Carcinoma of colon                    (score: 0.950)
      ...
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Link.constants import SNOMED_FSN_TYPE, SNOMED_SYN_TYPE

RESULTS_DIR = Path(__file__).parent / "Link" / "results"

SPAN_NOISE_SCORE_MIN = 0.80
SUMMARY_MIN_SCORE = 0.84
SPAN_OVERLAP_THRESHOLD = 0.7


def _ocr_page(pil_image) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    for lang in ("fra", "fra+eng", "eng"):
        try:
            text = pytesseract.image_to_string(pil_image, lang=lang)
            if text.strip():
                return text
        except Exception:
            continue
    return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed. Run: pip install pdfplumber")
        sys.exit(1)

    ocr_available = False
    try:
        import pytesseract
        from pdf2image import convert_from_path
        pytesseract.get_tesseract_version()
        ocr_available = True
    except Exception:
        pass

    _SKIP_PAGE_MARKERS = [
        "résultat notereader",
        "liste de problèmes encodée",
        "liste de problemes encodee",
    ]

    text_parts = []
    ocr_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""

            if len(page_text.strip()) >= 30:
                if any(marker in page_text.lower() for marker in _SKIP_PAGE_MARKERS):
                    print(f"  [skip] Page {page_num}: administrative/NoteReader page excluded")
                    continue
                text_parts.append(page_text)
                continue

            if ocr_available:
                try:
                    images = convert_from_path(pdf_path, dpi=300,
                                               first_page=page_num,
                                               last_page=page_num)
                    if images:
                        ocr_text = _ocr_page(images[0])
                        if ocr_text.strip():
                            text_parts.append(ocr_text)
                            ocr_pages += 1
                            continue
                except Exception as e:
                    print(f"  [OCR] Page {page_num} failed: {e}")

            if page_text.strip():
                text_parts.append(page_text)

    if ocr_pages:
        print(f"  OCR used on {ocr_pages}/{total_pages} image-based pages")
    elif not ocr_available:
        print("  Note: OCR not available — image-based pages skipped.")
        print("  To enable OCR: pip install pytesseract pdf2image pillow")
        print("  And install Tesseract with French language pack.")

    return "\n\n".join(text_parts)


def load_french_labels(rf2_path: str) -> dict:
    path = Path(rf2_path)
    if not path.exists():
        return {}

    synonyms: dict = {}
    fsns:     dict = {}

    try:
        with open(path, encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                active, lang, type_id, term, concept_id = (
                    parts[2], parts[5], parts[6], parts[7], parts[4]
                )
                if active != "1" or lang != "fr":
                    continue
                if type_id == SNOMED_SYN_TYPE:
                    synonyms.setdefault(concept_id, term)
                elif type_id == SNOMED_FSN_TYPE:
                    fsns.setdefault(concept_id, term)
    except Exception as e:
        print(f"  [warning] Could not load French labels: {e}")
        return {}

    labels = {**fsns, **synonyms}
    print(f"  {len(labels):,} French concept labels loaded")
    return labels


def load_concept_descriptions() -> dict:
    ids_path  = RESULTS_DIR / "rag_index" / "concept_ids.pkl"
    text_path = RESULTS_DIR / "rag_index" / "concept_texts.pkl"
    if not ids_path.exists() or not text_path.exists():
        return {}
    with open(ids_path, "rb") as f:
        concept_ids = pickle.load(f)
    with open(text_path, "rb") as f:
        concept_texts = pickle.load(f)
    lookup = {}
    for cid, text in zip(concept_ids, concept_texts):
        if cid not in lookup:
            lookup[cid] = text
    return lookup


def run_method1(text: str, snomed_rf2_fr: str = None) -> list:
    from Link.method1_dictionary import load_dictionary, predict
    dict_path = RESULTS_DIR / "dictionary.json"
    if not dict_path.exists():
        print("Dictionary not found — cannot run Method 1. Build it first with run_pipeline.py.")
        sys.exit(1)
    return predict(text, load_dictionary(str(dict_path)))


def run_method2(text: str) -> list:
    from Link.method2_sapbert import SapBERTIndex, predict
    from Link.method1_dictionary import load_dictionary
    idx = SapBERTIndex(str(RESULTS_DIR / "sapbert_index"))
    d = load_dictionary(str(RESULTS_DIR / "dictionary.json")) if (RESULTS_DIR / "dictionary.json").exists() else None
    return predict(text, idx, dictionary=d)


def run_method3(text: str, llm_model: str, use_rerank: bool) -> list:
    from Link.method3_llm_rag import RAGIndex, OllamaClient, predict
    idx = RAGIndex(str(RESULTS_DIR / "rag_index"))
    client = OllamaClient(model_name=llm_model)
    return predict(text, idx, client, use_llm_reranking=use_rerank)


def run_method4(text: str, llm_model: str, use_rerank: bool) -> list:
    from Link.method4_hybrid import predict
    from Link.method3_llm_rag import RAGIndex, OllamaClient
    from Link.method1_dictionary import load_dictionary
    d      = load_dictionary(str(RESULTS_DIR / "dictionary.json"))
    idx    = RAGIndex(str(RESULTS_DIR / "rag_index"))
    client = OllamaClient(model_name=llm_model)
    return predict(text, d, idx, client, use_llm_reranking=use_rerank)


_SKIP_TYPES = (
    "(qualifier value)",
    "(physical object)",
    "(environment)",
    "(geographic location)",
    "(attribute)",
    "(organism)",
    "(occupation)",
    "(cell)",
    "(cell structure)",
    "(situation)",
    "(unit of presentation)",
    "(unit)",
)

_TYPE_PRIORITY = {
    "(disorder)": 0,
    "(morphologic abnormality)": 1,
    "(finding)": 2,
    "(observable entity)": 3,
    "(procedure)": 4,
    "(substance)": 5,
}


def _is_clinical(desc: str) -> bool:
    desc_lower = desc.lower()
    for t in _SKIP_TYPES:
        if t in desc_lower:
            return False
    return any(t in desc_lower for t in _TYPE_PRIORITY)


def _summary_sort_key(pred: dict, descriptions: dict, in_priority: bool = False):
    cands = pred.get("candidates", [])
    if not cands:
        return (1, 99, 0, 0)
    top1  = cands[0]
    cid   = top1.get("concept_id", "")
    desc  = (top1.get("description") or descriptions.get(cid, "")).lower()
    score = top1.get("score", 0.0)
    ntok  = len(pred["span"].split())

    type_prio = 99
    for t, p in _TYPE_PRIORITY.items():
        if t in desc:
            type_prio = p
            break

    section_prio = 0 if in_priority else 1
    return (section_prio, type_prio, -ntok, -score)


_PRIORITY_SECTIONS = re.compile(
    r"\b(conclusion|diagnostic|impression|diagnos[ei]|assessment|plan)\b",
    re.IGNORECASE,
)


def _priority_offsets(text: str) -> list[tuple[int,int]]:
    ranges = []
    for m in _PRIORITY_SECTIONS.finditer(text):
        sec_start = m.start()
        rest = text[m.end():]
        gap = re.search(r"\n\s*\n", rest)
        sec_end = m.end() + (gap.start() if gap else len(rest))
        ranges.append((sec_start, sec_end))
    return ranges


def display(predictions: list, descriptions: dict, top: int = 0,
            fr_labels: dict = None, source_text: str = "") -> None:
    if fr_labels is None:
        fr_labels = {}

    priority_ranges = _priority_offsets(source_text) if source_text else []

    def _in_priority_section(start: int) -> bool:
        return any(s <= start <= e for s, e in priority_ranges)

    if not predictions:
        print("No annotations found.")
        return

    # Basic noise filter (OCR artifacts, short low-confidence spans)
    filtered = [
        p for p in predictions
        if not (
            len(p["span"].split()) == 1
            and len(p["span"]) < 6
            and not p["span"].isupper()
            and p.get("candidates")
            and p["candidates"][0].get("score", 1.0) < SPAN_NOISE_SCORE_MIN
        )
    ]

    if top > 0:
        seen_cids: set = set()
        seen_span_tokens: list = []

        def _span_tokens(span: str) -> set:
            return set(re.sub(r"[^a-zàâçéèêëîïôûùüÿæœ]", " ",
                              span.lower()).split())

        def _overlaps_seen(span: str) -> bool:
            toks = _span_tokens(span)
            if not toks:
                return False
            for seen in seen_span_tokens:
                overlap = len(toks & seen) / min(len(toks), len(seen))
                if overlap >= SPAN_OVERLAP_THRESHOLD:
                    return True
            return False

        summary = []
        for pred in sorted(filtered,
                           key=lambda p: _summary_sort_key(
                               p, descriptions,
                               in_priority=_in_priority_section(p["start"])
                           )):
            if not pred.get("candidates"):
                continue
            top1  = pred["candidates"][0]
            cid   = top1.get("concept_id", "")
            score = top1.get("score", 0.0)
            en_desc = top1.get("description") or descriptions.get(cid, "")
            if not _is_clinical(en_desc):
                continue
            if score < SUMMARY_MIN_SCORE:
                continue
            if cid in seen_cids:
                continue
            if _overlaps_seen(pred["span"]):
                continue
            seen_cids.add(cid)
            seen_span_tokens.append(_span_tokens(pred["span"]))
            label = fr_labels.get(cid) or en_desc
            summary.append((pred["span"], cid, label, en_desc, score))
            if len(summary) >= top:
                break

        print(f"\nTop {len(summary)} clinical findings / problems:\n")
        for i, (span, cid, label, en_desc, score) in enumerate(summary, 1):
            if len(label) > 70:
                label = label[:67] + "..."
            sem_tag = ""
            for t in _TYPE_PRIORITY:
                if t in en_desc.lower():
                    sem_tag = f" {t}"
                    break
            if sem_tag.lower() not in label.lower():
                label = label + sem_tag
            print(f"  {i:2d}. [{cid}]  {label}")
            print(f"       ← \"{span}\"  (score: {score:.3f})")
        print()

    else:
        print(f"\nFound {len(filtered)} annotated spans:\n")
        for pred in filtered:
            start      = pred["start"]
            end        = pred["end"]
            span       = pred["span"]
            candidates = pred.get("candidates", [])

            print(f"[{start:5d}-{end:5d}] \"{span}\"")
            for i, c in enumerate(candidates[:5], 1):
                cid     = c.get("concept_id", "?")
                score   = c.get("score", 0.0)
                en_desc = c.get("description") or descriptions.get(cid, "")
                label   = fr_labels.get(cid) or en_desc
                if len(label) > 60:
                    label = label[:57] + "..."
                print(f"  {i}. {cid:<15}  {label:<60}  (score: {score:.3f})")
            print()


def _rebuild_dict_with_french(snomed_rf2_fr: str) -> None:
    from Link.method1_dictionary import (
        build_from_snomed_rf2, load_dictionary, save_dictionary, _norm_key
    )

    dict_path = RESULTS_DIR / "dictionary.json"
    if not dict_path.exists():
        print(f"  Dictionary not found at {dict_path} — skipping French rebuild.")
        return

    if not Path(snomed_rf2_fr).exists():
        print(f"  French RF2 file not found: {snomed_rf2_fr}")
        return

    print(f"Adding French SNOMED synonyms from {snomed_rf2_fr} ...")
    d = load_dictionary(str(dict_path))

    tier2_fr = build_from_snomed_rf2(snomed_rf2_fr, language_code="fr")
    tier1 = d["tier1"]
    tier2 = d["tier2"]

    added = 0
    for k, v in tier2_fr.items():
        if k not in tier1 and k not in tier2:
            tier2[k] = v
            added += 1

    d["tier2"] = tier2
    save_dictionary(d, str(dict_path))
    print(f"  {added} French synonyms added. Dictionary saved.\n")


def main():
    parser = argparse.ArgumentParser(description="Annotate a PDF with SNOMED CT codes")
    parser.add_argument("--pdf",           required=True,  help="Path to PDF file")
    parser.add_argument("--method",        default="4",    choices=["1", "2", "3", "4"],
                        help="Method to use (default: 4)")
    parser.add_argument("--llm-model",     default="llama3.1:8b",
                        help="Ollama model for methods 3 & 4 (default: llama3.1:8b)")
    parser.add_argument("--no-rerank",     dest="rerank", action="store_false", default=True,
                        help="Disable LLM reranking for methods 3 & 4 (faster)")
    parser.add_argument("--top",           type=int, default=0,
                        help="Show a deduplicated summary of top-N clinical findings. 0 = show all spans.")
    parser.add_argument("--snomed-rf2-fr", default=None,
                        help="Path to SNOMED RF2 French description file")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Extracting text from {pdf_path} ...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No text extracted from PDF.")
        sys.exit(1)
    print(f"  Extracted {len(text)} characters\n")

    if args.snomed_rf2_fr:
        _rebuild_dict_with_french(args.snomed_rf2_fr)

    print("Loading concept descriptions ...")
    descriptions = load_concept_descriptions()
    print(f"  {len(descriptions)} concept descriptions loaded")

    fr_labels = {}
    if args.snomed_rf2_fr:
        print("Loading French concept labels ...")
        fr_labels = load_french_labels(args.snomed_rf2_fr)
    print()

    print(f"Running Method {args.method} ...")
    if args.method == "1":
        predictions = run_method1(text)
    elif args.method == "2":
        predictions = run_method2(text)
    elif args.method == "3":
        predictions = run_method3(text, args.llm_model, args.rerank)
    else:
        predictions = run_method4(text, args.llm_model, args.rerank)

    display(predictions, descriptions, top=args.top, fr_labels=fr_labels,
            source_text=text)


if __name__ == "__main__":
    main()
