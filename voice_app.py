"""
voice_app.py — Real-time voice transcription + live SNOMED CT entity detection

Launch:
    streamlit run voice_app.py

Transcription: uses the browser's built-in Web Speech API (Chrome / Edge).
Same engine as Google Assistant, Cortana, and every modern voice interface.
No local model, no GPU, instant response.  Works offline for SNOMED detection;
only the speech recognition itself requires internet access to Google's endpoint.
"""

import html
import io
import csv
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as _components

# ── Project root on sys.path ─────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="SNOMED Voice Linker",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Artefact paths ────────────────────────────────────────────────────────────
_DICT_PATH = _SCRIPT_DIR / "Link" / "results" / "dictionary.json"
_RAG_DIR   = _SCRIPT_DIR / "Link" / "results" / "rag_index"
_FR_RF2    = (
    _SCRIPT_DIR
    / "xSnomedCT_BelgianAlphaReleasePackage_Alpha_20200930"
    / "Snapshot" / "Terminology"
    / "xsct2_Description_Snapshot_BelgianAlphaPackage_BE1000173_20200930.txt"
)

# ── Constants ─────────────────────────────────────────────────────────────────
_OVERLAP_CHARS = 80   # character overlap between consecutive detection windows
_MIN_DETECT    = 150  # minimum new characters before triggering detection (~1–2 sentences)

# ── Custom Web Speech API component (real-time interim results) ───────────────
_STT_COMPONENT_DIR = _SCRIPT_DIR / "_stt_component"
_stt_component = _components.declare_component(
    "stt_webapi",
    path=str(_STT_COMPONENT_DIR),
)


# ── @st.cache_resource loaders ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Method 1 dictionary…")
def _load_m1_dict():
    if not _DICT_PATH.exists():
        return None
    from Link.method1_dictionary import load_dictionary
    return load_dictionary(str(_DICT_PATH))


@st.cache_resource(show_spinner="Loading SNOMED vector index (Method 4) — may take ~30 s…")
def _load_rag_index():
    if not _RAG_DIR.exists():
        return None
    try:
        from Link.method3_llm_rag import RAGIndex
        return RAGIndex(str(_RAG_DIR))
    except Exception:
        return None


@st.cache_resource(show_spinner="Connecting to Ollama (llama3.1:8b)…")
def _load_llm_client():
    try:
        from Link.method3_llm_rag import OllamaClient
        return OllamaClient(model_name="llama3.1:8b")
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading French labels from Belgian RF2…")
def _load_fr_labels():
    if not _FR_RF2.exists():
        return {}
    try:
        from annotate_pdf import load_french_labels
        return load_french_labels(str(_FR_RF2))
    except Exception:
        return {}


@st.cache_resource(show_spinner="Loading concept descriptions…")
def _load_descriptions():
    try:
        from annotate_pdf import load_concept_descriptions
        return load_concept_descriptions()
    except Exception:
        return {}


# ── Pre-load heavy artefacts once ─────────────────────────────────────────────
_load_m1_dict()
_load_fr_labels()
_load_descriptions()
if _RAG_DIR.exists():
    _load_rag_index()


# ── Background SNOMED detection worker ────────────────────────────────────────

def _detection_worker(text: str, result_q, char_offset: int) -> None:
    try:
        m1   = _load_m1_dict()
        rag  = _load_rag_index()
        llmc = _load_llm_client()
        if m1 is not None and rag is not None and llmc is not None:
            from Link.method4_hybrid import predict
            preds = predict(text, m1, rag, llmc, use_llm_reranking=False)
        elif m1 is not None:
            from Link.method1_dictionary import predict
            preds = predict(text, m1)
        else:
            preds = []
        result_q.put({"preds": preds, "offset": char_offset})
    except Exception as exc:
        result_q.put({"preds": [], "offset": char_offset, "error": str(exc)})


# ── Label helpers ─────────────────────────────────────────────────────────────

def _get_label(concept_id: str) -> str:
    lbl = _load_fr_labels().get(concept_id)
    if lbl:
        return lbl
    desc = _load_descriptions().get(concept_id)
    if desc:
        return desc
    return concept_id


def _score_color(score: float) -> str:
    if score >= 0.90:
        return "#28a745"
    if score >= 0.70:
        return "#fd7e14"
    return "#dc3545"


# ── Session state ─────────────────────────────────────────────────────────────

_SS: dict = {
    "language":         "fr-BE",
    "threshold":        0.6,
    "transcript":       "",
    "processed_up_to":  0,
    "stt_seen":         "",    # last STT value already appended to transcript
    "stt_key":          0,     # incremented on reset to force component re-init
    "pending":          [],
    "accepted":         [],
    "rejected_keys":    set(),
    "accepted_keys":    set(),
    "pending_keys":     set(),
    "detection_active": False,
    "detection_q":      None,
}

for _k, _v in _SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (set, list, dict)) else _v


def _ensure_detection_q():
    import queue
    if st.session_state.detection_q is None:
        st.session_state.detection_q = queue.Queue(maxsize=60)


# ── Detection trigger ─────────────────────────────────────────────────────────

def _maybe_trigger_detection() -> None:
    """Launch a detection thread if enough new text has accumulated."""
    _ensure_detection_q()
    total = len(st.session_state.transcript)
    # Require both a minimum total length and a minimum amount of NEW text
    # since the last detection, so single short words never trigger alone.
    if (total >= _MIN_DETECT
            and total > st.session_state.processed_up_to + _MIN_DETECT
            and not st.session_state.detection_active):
        overlap_start = max(0, st.session_state.processed_up_to - _OVERLAP_CHARS)
        detect_text   = st.session_state.transcript[overlap_start:]
        st.session_state.processed_up_to = total
        st.session_state.detection_active = True
        threading.Thread(
            target=_detection_worker,
            args=(detect_text, st.session_state.detection_q, overlap_start),
            daemon=True,
        ).start()


def _drain_detection_q() -> None:
    """Collect any finished detection results and build suggestion cards."""
    if st.session_state.detection_q is None:
        return
    while True:
        try:
            result = st.session_state.detection_q.get_nowait()
        except Exception:
            break
        st.session_state.detection_active = False
        new_sugs = _make_suggestions(result.get("preds", []))
        if new_sugs:
            st.session_state.pending = new_sugs + st.session_state.pending


# ── Suggestion builder ────────────────────────────────────────────────────────

def _make_suggestions(preds: list) -> list:
    threshold    = st.session_state.threshold
    rejected     = st.session_state.rejected_keys
    accepted     = st.session_state.accepted_keys
    pending_keys = st.session_state.pending_keys
    ts = datetime.now().strftime("%H:%M:%S")
    new: list = []
    for pred in preds:
        cands = pred.get("candidates", [])
        if not cands:
            continue
        top   = cands[0]
        score = top.get("score", 0.0)
        if score < threshold:
            continue
        span_lo = pred["span"].lower()
        key = f"{span_lo}::{top['concept_id']}"
        if key in rejected or key in accepted or key in pending_keys:
            continue
        new.append({
            "id":         f"{span_lo}_{top['concept_id']}_{int(time.time()*1000)}",
            "span":       pred["span"],
            "candidates": cands[:5],
            "timestamp":  ts,
        })
        pending_keys.add(key)
    return new


# ── Accept / Reject / Reset callbacks ────────────────────────────────────────

def _accept_top(sug_id: str) -> None:
    for i, sug in enumerate(st.session_state.pending):
        if sug["id"] != sug_id:
            continue
        top = sug["candidates"][0]
        cid = top["concept_id"]
        st.session_state.accepted.append({
            "span":       sug["span"],
            "concept_id": cid,
            "label":      _get_label(cid),
            "score":      top["score"],
            "timestamp":  sug["timestamp"],
        })
        key = f"{sug['span'].lower()}::{cid}"
        st.session_state.accepted_keys.add(key)
        st.session_state.pending_keys.discard(key)
        st.session_state.pending.pop(i)
        break


def _accept_alt(sug_id: str, cand_idx: int) -> None:
    for i, sug in enumerate(st.session_state.pending):
        if sug["id"] != sug_id:
            continue
        cand = sug["candidates"][cand_idx]
        cid  = cand["concept_id"]
        st.session_state.accepted.append({
            "span":       sug["span"],
            "concept_id": cid,
            "label":      _get_label(cid),
            "score":      cand["score"],
            "timestamp":  sug["timestamp"],
        })
        key = f"{sug['span'].lower()}::{cid}"
        st.session_state.accepted_keys.add(key)
        st.session_state.pending_keys.discard(key)
        st.session_state.pending.pop(i)
        break


def _reject(sug_id: str) -> None:
    for i, sug in enumerate(st.session_state.pending):
        if sug["id"] != sug_id:
            continue
        top = sug["candidates"][0]
        key = f"{sug['span'].lower()}::{top['concept_id']}"
        st.session_state.rejected_keys.add(key)
        st.session_state.pending_keys.discard(key)
        st.session_state.pending.pop(i)
        break


def _reset_session() -> None:
    st.session_state.transcript       = ""
    st.session_state.processed_up_to  = 0
    st.session_state.stt_seen         = ""
    st.session_state.stt_key         += 1   # new key → component re-inits with default=None
    st.session_state.pending          = []
    st.session_state.accepted         = []
    st.session_state.rejected_keys    = set()
    st.session_state.accepted_keys    = set()
    st.session_state.pending_keys     = set()
    st.session_state.detection_active = False
    st.session_state.detection_q      = None


# ── Text-mode detection (synchronous) ────────────────────────────────────────

def _run_text_detection(text: str) -> None:
    st.session_state.transcript      = text
    st.session_state.processed_up_to = len(text)
    st.session_state.pending         = []
    st.session_state.pending_keys    = set()
    _ensure_detection_q()
    m1   = _load_m1_dict()
    rag  = _load_rag_index()
    llmc = _load_llm_client()
    if m1 is not None and rag is not None and llmc is not None:
        from Link.method4_hybrid import predict
        preds = predict(text, m1, rag, llmc, use_llm_reranking=False)
    elif m1 is not None:
        st.warning("RAG index or Ollama not available — falling back to dictionary only.")
        from Link.method1_dictionary import predict
        preds = predict(text, m1)
    else:
        st.error("Dictionary not found — build it first with run_pipeline.py.")
        return
    st.session_state.pending = _make_suggestions(preds)


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🎙️ SNOMED Voice Linker")
st.caption("Real-time SNOMED CT coding from voice — Method 4 (LLM NER + Dict + FAISS)")

cc = st.columns([3, 3, 2])

with cc[0]:
    st.selectbox(
        "Language",
        options=["fr-BE", "fr-FR", "en-US"],
        format_func=lambda x: {
            "fr-BE": "🇧🇪 Français (Belgique)",
            "fr-FR": "🇫🇷 Français (France)",
            "en-US": "🇬🇧 English",
        }[x],
        key="language",
        help="Language for voice recognition. fr-BE is recommended for Saint-Luc notes.",
    )

with cc[1]:
    st.slider(
        "Confidence threshold",
        min_value=0.0, max_value=1.0,
        value=st.session_state.threshold,
        step=0.05,
        key="threshold",
        help="Only show suggestions above this score.",
    )

with cc[2]:
    st.button(
        "↺ Reset",
        on_click=_reset_session,
        use_container_width=True,
        help="Clear transcript, suggestions, and accepted codes.",
    )

st.divider()

# ── Two-column main layout ────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="medium")

# ─── Left column: Transcript + voice input ───────────────────────────────────
with left_col:
    st.subheader("📝 Transcript")

    # ── Web Speech API recorder (real-time interim results in the widget) ────
    # language is already a full BCP47 tag (fr-BE, fr-FR, en-US)
    # key changes on reset → forces component to re-init with default=None
    stt_result = _stt_component(
        language=st.session_state.language,
        default=None,
        key=f"stt_widget_{st.session_state.stt_key}",
    )

    # stt_result is the cumulative transcript of the current recording session.
    # Each committed phrase fires a Streamlit rerun with a longer stt_result.
    if stt_result:
        seen = st.session_state.stt_seen
        if len(stt_result) < len(seen):
            # New recording session started (JS reset finalText) — clear tracking
            seen = ""
            st.session_state.stt_seen = ""
        new_chunk = stt_result[len(seen):].strip()
        if new_chunk:
            sep = " " if st.session_state.transcript else ""
            st.session_state.transcript += sep + new_chunk
            _maybe_trigger_detection()
        st.session_state.stt_seen = stt_result

    if not st.session_state.transcript:
        st.caption("Press **▶ Start** in the widget above — transcript grows here as you speak.")
    elif st.session_state.detection_active:
        st.caption("⏳ SNOMED detection running…")

    # ── Editable transcript — user can fix misrecognitions directly ───────────
    # key="transcript" binds to session_state.transcript; edits are reflected
    # immediately and will be used by the next SNOMED detection pass.
    st.text_area(
        "Transcript",
        key="transcript",
        height=220,
        label_visibility="collapsed",
        placeholder="Press ▶ Start to dictate, or type / paste text here…",
    )
    _detect_col, _status_col = st.columns([2, 3])
    with _detect_col:
        if st.button("🔍 Detect now", use_container_width=True,
                     help="Run SNOMED detection on the current transcript text."):
            if st.session_state.transcript.strip():
                _run_text_detection(st.session_state.transcript)
    with _status_col:
        if st.session_state.detection_active:
            st.caption("⏳ SNOMED detection running…")

# ─── Right column: Suggestion cards ──────────────────────────────────────────
# Drain any finished detection results before rendering
_drain_detection_q()

with right_col:
    _n_pending   = len(st.session_state.pending)
    _detecting_b = " &nbsp; ⏳" if st.session_state.detection_active else ""
    st.subheader(f"💡 Suggestions ({_n_pending})")
    st.caption(f"Method 4 — LLM NER + Dict + FAISS{_detecting_b}")

    if not st.session_state.pending:
        st.markdown(
            '<p style="color:#888; font-style:italic;">'
            + (
                "No suggestions yet — try lowering the confidence threshold."
                if st.session_state.transcript
                else "Suggestions appear here as SNOMED concepts are detected."
            )
            + "</p>",
            unsafe_allow_html=True,
        )
    else:
        for sug in st.session_state.pending:
            top   = sug["candidates"][0]
            cid   = top["concept_id"]
            score = top["score"]
            lbl   = _get_label(cid)
            color = _score_color(score)

            with st.container(border=True):
                st.markdown(
                    f'<span style="background:#1e3a5f; color:#90caf9; '
                    f'padding:2px 8px; border-radius:4px; font-style:italic; font-size:0.9em;">'
                    f'"{html.escape(sug["span"])}"</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**`{cid}`**  \n"
                    + (lbl[:72] + "…" if len(lbl) > 72 else lbl)
                )
                bar_width = int(score * 100)
                st.markdown(
                    f'<div style="background:#333; border-radius:4px; height:6px; margin:4px 0 2px;">'
                    f'<div style="background:{color}; width:{bar_width}%; '
                    f'height:6px; border-radius:4px;"></div></div>'
                    f'<small style="color:#aaa;">{score:.0%} confidence '
                    f'(Tier {top.get("tier", "?")})</small>',
                    unsafe_allow_html=True,
                )
                _bcols = st.columns(2)
                with _bcols[0]:
                    st.button(
                        "✓ Accept",
                        key=f"acc_{sug['id']}",
                        on_click=_accept_top,
                        args=(sug["id"],),
                        type="primary",
                        use_container_width=True,
                    )
                with _bcols[1]:
                    st.button(
                        "✗ Reject",
                        key=f"rej_{sug['id']}",
                        on_click=_reject,
                        args=(sug["id"],),
                        use_container_width=True,
                    )
                alts = sug["candidates"][1:3]
                if alts:
                    with st.expander("Other candidates"):
                        for _ci, _cand in enumerate(alts, start=1):
                            _c_cid   = _cand["concept_id"]
                            _c_score = _cand["score"]
                            _c_lbl   = _get_label(_c_cid)
                            _ac, _ab = st.columns([5, 1])
                            with _ac:
                                st.markdown(
                                    f"**#{_ci + 1}** `{_c_cid}`  \n"
                                    + (_c_lbl[:55] + "…" if len(_c_lbl) > 55 else _c_lbl)
                                    + f"  *({_c_score:.2f})*"
                                )
                            with _ab:
                                st.button(
                                    "✓",
                                    key=f"acc_{sug['id']}_alt{_ci}",
                                    on_click=_accept_alt,
                                    args=(sug["id"], _ci),
                                    use_container_width=True,
                                    help=f"Accept {_c_cid}",
                                )

# ── Bottom: Accepted codes table ──────────────────────────────────────────────
st.divider()

_n_acc = len(st.session_state.accepted)
st.subheader(f"✅ Accepted Codes ({_n_acc})")

if _n_acc == 0:
    st.caption("Accepted SNOMED codes will accumulate here as you click ✓ Accept.")
else:
    import pandas as pd

    _rows = [
        {
            "Detected span": a["span"],
            "SNOMED code":   a["concept_id"],
            "French label":  a["label"],
            "Score":         round(a["score"], 4),
            "Time":          a["timestamp"],
        }
        for a in st.session_state.accepted
    ]
    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

    _csv_buf = io.StringIO()
    _writer  = csv.DictWriter(
        _csv_buf,
        fieldnames=["Detected span", "SNOMED code", "French label", "Score", "Time"],
    )
    _writer.writeheader()
    _writer.writerows(_rows)

    st.download_button(
        "⬇️ Export accepted codes to CSV",
        data=_csv_buf.getvalue(),
        file_name=f"snomed_voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# ── Auto-rerun while detection is running ────────────────────────────────────
if st.session_state.detection_active:
    time.sleep(0.5)
    st.rerun()
