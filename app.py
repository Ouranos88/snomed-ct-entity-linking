"""
app.py — Streamlit UI for SNOMED CT Entity Linking

Launch:
    streamlit run app.py
"""

import os
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="SNOMED CT Linker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path(__file__).parent / "Link" / "results"

FR_RF2 = (
    Path(__file__).parent
    / "xSnomedCT_BelgianAlphaReleasePackage_Alpha_20200930"
    / "Snapshot" / "Terminology"
    / "xsct2_Description_Snapshot_BelgianAlphaPackage_BE1000173_20200930.txt"
)


@st.cache_resource(show_spinner="Loading SNOMED concept descriptions…")
def _get_descriptions():
    from annotate_pdf import load_concept_descriptions
    return load_concept_descriptions()


@st.cache_resource(show_spinner="Loading French labels…")
def _get_fr_labels():
    from annotate_pdf import load_french_labels
    return load_french_labels(str(FR_RF2)) if FR_RF2.exists() else {}


_SKIP_TYPES = (
    "(qualifier value)", "(physical object)", "(environment)",
    "(geographic location)", "(attribute)", "(organism)",
    "(occupation)", "(cell)", "(cell structure)",
    "(situation)", "(unit of presentation)", "(unit)",
)

_TYPE_PRIORITY = {
    "(disorder)": 0,
    "(morphologic abnormality)": 1,
    "(finding)": 2,
    "(observable entity)": 3,
    "(procedure)": 4,
    "(substance)": 5,
}

_PRIORITY_SECTIONS = re.compile(
    r"\b(conclusion|diagnostic|impression|diagnos[ei]|assessment|plan)\b",
    re.IGNORECASE,
)


def _is_clinical(desc: str) -> bool:
    d = desc.lower()
    # Drop only explicitly non-clinical types; keep anything without a type tag
    if any(t in d for t in _SKIP_TYPES):
        return False
    return True


def _priority_ranges(text: str) -> list:
    ranges = []
    for m in _PRIORITY_SECTIONS.finditer(text):
        rest = text[m.end():]
        gap = re.search(r"\n\s*\n", rest)
        ranges.append((m.start(), m.end() + (gap.start() if gap else len(rest))))
    return ranges


def _in_priority(start: int, ranges: list) -> bool:
    return any(s <= start <= e for s, e in ranges)


def _span_toks(span: str) -> set:
    return set(re.sub(r"[^a-zàâçéèêëîïôûùüÿæœ]", " ", span.lower()).split())


def _overlaps_seen(span: str, seen_tok_sets: list, threshold: float = 0.7) -> bool:
    toks = _span_toks(span)
    if not toks:
        return False
    for s in seen_tok_sets:
        if s and len(toks & s) / min(len(toks), len(s)) >= threshold:
            return True
    return False


def _type_label(en_desc: str) -> str:
    for t in _TYPE_PRIORITY:
        if t in en_desc.lower():
            return t
    return ""


def _score_color(score: float) -> str:
    if score >= 0.95:
        return "green"
    if score >= 0.87:
        return "orange"
    return "red"


def build_summary(
    predictions: list,
    descriptions: dict,
    fr_labels: dict,
    text: str,
    top_n: int,
    min_score: float = 0.84,
    noise_score_min: float = 0.80,
) -> list:
    """Filter, deduplicate, sort and return top-N clinical findings."""
    priority = _priority_ranges(text)

    # Basic noise filter
    filtered = [
        p for p in predictions
        if not (
            len(p["span"].split()) == 1
            and len(p["span"]) < 6
            and not p["span"].isupper()
            and p.get("candidates")
            and p["candidates"][0].get("score", 1.0) < noise_score_min
        )
    ]

    def _sort_key(p):
        cands = p.get("candidates", [])
        if not cands:
            return (1, 99, 0, 0)
        top1   = cands[0]
        cid    = top1.get("concept_id", "")
        desc   = (top1.get("description") or descriptions.get(cid, "")).lower()
        score  = top1.get("score", 0.0)
        ntok   = len(p["span"].split())
        tp     = next((v for k, v in _TYPE_PRIORITY.items() if k in desc), 99)
        sec    = 0 if _in_priority(p["start"], priority) else 1
        return (sec, tp, -ntok, -score)

    seen_cids: set = set()
    seen_toks: list = []
    summary = []

    for pred in sorted(filtered, key=_sort_key):
        if not pred.get("candidates"):
            continue
        top1  = pred["candidates"][0]
        cid   = top1.get("concept_id", "")
        score = top1.get("score", 0.0)
        en    = top1.get("description") or descriptions.get(cid, "")

        if not _is_clinical(en):
            continue
        if score < min_score:
            continue
        if cid in seen_cids:
            continue
        if _overlaps_seen(pred["span"], seen_toks):
            continue

        seen_cids.add(cid)
        seen_toks.append(_span_toks(pred["span"]))
        label = fr_labels.get(cid) or en
        summary.append({
            "rank":           len(summary) + 1,
            "concept_id":     cid,
            "label":          label,
            "en_desc":        en,
            "type":           _type_label(en),
            "span":           pred["span"],
            "score":          score,
            "all_candidates": pred.get("candidates", [])[:5],
        })
        if len(summary) >= top_n:
            break

    return summary


def lookup_keywords(
    keywords: list,
    descriptions: dict,
    fr_labels: dict,
    top_k: int = 5,
) -> list:
    from annotate_pdf import RESULTS_DIR
    dict_path = RESULTS_DIR / "dictionary.json"
    rag_path  = RESULTS_DIR / "rag_index"

    dictionary = None
    if dict_path.exists():
        from Link.method1_dictionary import load_dictionary
        dictionary = load_dictionary(str(dict_path))

    rag_index = None
    if rag_path.exists():
        from Link.method3_llm_rag import RAGIndex
        rag_index = RAGIndex(str(rag_path))

    pinned = []
    seen_cids: set = set()

    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue

        candidates = []

        if dictionary:
            from Link.method1_dictionary import _norm_key
            norm = _norm_key(kw)
            tier1 = dictionary.get("tier1", {})
            tier2 = dictionary.get("tier2", {})
            if norm in tier1:
                for entry in tier1[norm][:top_k]:
                    cid = entry.get("concept_id", "")
                    candidates.append({
                        "concept_id": cid,
                        "score": 1.0,
                        "description": descriptions.get(cid, entry.get("preferred_term", "")),
                    })
            elif norm in tier2:
                for entry in tier2[norm][:top_k]:
                    cid = entry.get("concept_id", "")
                    candidates.append({
                        "concept_id": cid,
                        "score": 0.95,
                        "description": descriptions.get(cid, entry.get("preferred_term", "")),
                    })

        if not candidates and rag_index is not None:
            results = rag_index.search([kw], top_k=top_k)[0]
            for hit in results:
                candidates.append({
                    "concept_id":  hit["concept_id"],
                    "score":       hit["score"],
                    "description": hit["description"],
                })

        if not candidates:
            continue

        top1 = candidates[0]
        cid  = top1["concept_id"]
        if cid in seen_cids:
            continue
        seen_cids.add(cid)

        en    = top1.get("description") or descriptions.get(cid, "")
        label = fr_labels.get(cid) or en
        pinned.append({
            "rank":           f"📌",
            "concept_id":     cid,
            "label":          label,
            "en_desc":        en,
            "type":           _type_label(en),
            "span":           kw,
            "score":          top1["score"],
            "all_candidates": candidates[:top_k],
            "pinned":         True,
        })

    return pinned


with st.sidebar:
    st.markdown("## ⚙️ Settings")

    METHOD_LABELS = {
        1: "1 — Dictionary + Fuzzy",
        2: "2 — SapBERT FAISS",
        3: "3 — LLM + RAG",
        4: "4 — LLM NER + Dict + FAISS",
    }
    METHOD_DESC = {
        1: "Exact dictionary lookup + rapidfuzz fuzzy matching. Fast, no GPU/LLM.",
        2: "SapBERT semantic embeddings + FAISS. GPU recommended.",
        3: "LLM extracts entities → multilingual-e5-large retrieves SNOMED candidates. Best results on French.",
        4: "LLM NER for span detection, then tiered dict + FAISS for linking. **Recommended for French.**",
    }

    method = st.selectbox(
        "Method",
        options=[1, 2, 3, 4],
        index=3,
        format_func=lambda x: METHOD_LABELS[x],
    )
    st.caption(METHOD_DESC[method])

    top_n = st.slider("Top N results", min_value=1, max_value=20, value=10)

    min_score = st.slider(
        "Min confidence score",
        min_value=0.50, max_value=1.00, value=0.84, step=0.01,
        help="Lower this if you get fewer results than expected.",
    )

    st.markdown("---")

    if method in (3, 4):
        llm_model = st.text_input("Ollama model", value="llama3.1:8b",
                                  help="Must be running: ollama serve")
        no_rerank = st.checkbox("Disable LLM reranking (faster)", value=True)
    else:
        llm_model = "llama3.1:8b"
        no_rerank = True

    st.markdown("---")
    st.caption(
        "Built for UCLouvain thesis 2025–2026\n\n"
        "Guillaume Gillet · Supervisors: C. Hemptinne, J. Vanderdonckt"
    )

st.title("🏥 SNOMED CT Entity Linker")
st.caption("Automatic recognition of SNOMED CT codes in clinical notes")

tab_pdf, tab_text = st.tabs(["📄 Upload PDF", "✏️ Paste text"])

pdf_bytes   = None
text_pasted = ""
source_name = ""

with tab_pdf:
    uploaded = st.file_uploader(
        "Upload a clinical note (PDF)", type=["pdf"],
        help="Image-based PDFs are OCR'd with Tesseract (French + English).",
    )
    if uploaded:
        pdf_bytes   = uploaded.read()
        source_name = uploaded.name

with tab_text:
    text_pasted = st.text_area(
        "Paste clinical note text",
        height=220,
        placeholder="Paste note text here (French or English)…",
    )
    if text_pasted.strip():
        source_name = "pasted text"

with st.expander("📌 Priority keywords (optional)", expanded=False):
    st.caption(
        "Enter terms you want to force-lookup regardless of what the method finds. "
        "One keyword per line, or comma-separated. These appear pinned at the top of results."
    )
    keywords_raw = st.text_area(
        "Keywords",
        height=100,
        placeholder="e.g.\ndystrophie de Fuchs\nglaucome\ncataracte sénile",
        label_visibility="collapsed",
    )

run_btn = st.button(
    "🔍 Run SNOMED linking",
    type="primary",
    use_container_width=True,
    disabled=(not pdf_bytes and not text_pasted.strip()),
)

if run_btn:
    note_text = ""

    if pdf_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        with st.spinner("Extracting text from PDF…"):
            from annotate_pdf import extract_text_from_pdf
            note_text = extract_text_from_pdf(tmp_path)
        os.unlink(tmp_path)
        if not note_text.strip():
            st.error("No text could be extracted from the PDF.")
            st.stop()
    elif text_pasted.strip():
        note_text = text_pasted.strip()
    else:
        st.warning("Upload a PDF or paste some text first.")
        st.stop()

    with st.spinner(f"Running Method {method} ({METHOD_LABELS[method]})…"):
        try:
            if method == 1:
                from annotate_pdf import run_method1
                preds = run_method1(note_text)
            elif method == 2:
                from annotate_pdf import run_method2
                preds = run_method2(note_text)
            elif method == 3:
                from annotate_pdf import run_method3
                preds = run_method3(note_text, llm_model, not no_rerank)
            else:
                from annotate_pdf import run_method4
                preds = run_method4(note_text, llm_model, not no_rerank)
        except FileNotFoundError as e:
            st.error(
                f"Artifact not found: {e}\n\n"
                "Run `python -m Link.run_pipeline --method 1 ...` first to build the dictionary and index."
            )
            st.stop()
        except Exception as e:
            st.error(f"Method {method} failed: {e}")
            st.stop()

    keywords = [
        k.strip()
        for k in re.split(r"[,\n]", keywords_raw)
        if k.strip()
    ]

    st.session_state["preds"]          = preds
    st.session_state["note_text"]      = note_text
    st.session_state["source_name"]    = source_name
    st.session_state["used_method"]    = method
    st.session_state["used_top_n"]     = top_n
    st.session_state["used_min_score"] = min_score
    st.session_state["keywords"]       = keywords

if "preds" in st.session_state:
    preds       = st.session_state["preds"]
    note_text   = st.session_state["note_text"]
    source_name = st.session_state["source_name"]
    used_method    = st.session_state["used_method"]
    used_top_n     = st.session_state.get("used_top_n", top_n)
    used_min_score = st.session_state.get("used_min_score", min_score)
    keywords       = st.session_state.get("keywords", [])

    descriptions = _get_descriptions()
    fr_labels    = _get_fr_labels()

    pinned  = lookup_keywords(keywords, descriptions, fr_labels) if keywords else []
    summary = build_summary(
        preds, descriptions, fr_labels, note_text,
        top_n=used_top_n, min_score=used_min_score,
    )

    pinned_cids = {p["concept_id"] for p in pinned}
    summary = [s for s in summary if s["concept_id"] not in pinned_cids]

    all_results = pinned + summary

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source", source_name or "—")
    c2.metric("Method", METHOD_LABELS[used_method])
    c3.metric("Predictions (raw)", len(preds))
    c4.metric("Clinical findings", len(all_results))

    if not all_results:
        st.warning(
            "No clinical findings found above the confidence threshold.\n\n"
            "Try lowering the threshold or switching to Method 3 or 4."
        )
    else:
        st.subheader(f"Top {len(all_results)} SNOMED CT findings")

        import csv, io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["rank", "pinned", "concept_id", "label_fr", "label_en", "type", "detected_span", "score"])
        for i, item in enumerate(all_results, 1):
            w.writerow([
                i, item.get("pinned", False), item["concept_id"], item["label"],
                item["en_desc"], item["type"], item["span"], f"{item['score']:.4f}"
            ])
        st.download_button(
            "⬇️ Export CSV",
            data=buf.getvalue(),
            file_name="snomed_results.csv",
            mime="text/csv",
        )

        st.markdown("---")

        for item in all_results:
            score   = item["score"]
            sem_tag = item["type"]
            label   = item["label"]
            en_desc = item["en_desc"]

            display_label = label if len(label) <= 80 else label[:77] + "…"
            badge     = f":{_score_color(score)}[{score:.3f}]"
            is_pinned = item.get("pinned", False)
            rank_str  = item["rank"] if is_pinned else f"**{item['rank']}.**"
            pin_mark  = " 📌 *keyword*" if is_pinned else ""

            with st.expander(
                f"{rank_str} {display_label} &nbsp; {badge}{pin_mark}",
                expanded=True,
            ):
                col_info, col_score = st.columns([3, 1])

                with col_info:
                    st.markdown(f"**SNOMED code:** `{item['concept_id']}`")
                    st.markdown(f"**Detected span:** *\"{item['span']}\"*")
                    if sem_tag:
                        st.caption(f"Semantic type: {sem_tag}")
                    if label != en_desc and len(en_desc) > 2:
                        st.caption(f"EN: {en_desc[:120]}")

                with col_score:
                    st.metric("Score", f"{score:.3f}")

                if len(item["all_candidates"]) > 1:
                    with st.expander("Other candidates (top 5)"):
                        rows = []
                        for i, c in enumerate(item["all_candidates"], 1):
                            c_cid   = c.get("concept_id", "")
                            c_score = c.get("score", 0.0)
                            c_en    = c.get("description") or descriptions.get(c_cid, "")
                            c_label = fr_labels.get(c_cid) or c_en
                            rows.append({
                                "Rank": i,
                                "Code": c_cid,
                                "Label": c_label[:90],
                                "Score": round(c_score, 4),
                            })
                        import pandas as pd
                        st.dataframe(
                            pd.DataFrame(rows).set_index("Rank"),
                            use_container_width=True,
                        )
