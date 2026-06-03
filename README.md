# SNOMED CT Entity Linking Pipeline

Automatic recognition and generation of SNOMED CT codes from clinical notes.  
Developed as part of the UCLouvain Master thesis 2025–2026 — *Automatic Recognition and Generation of SNOMED codes in Medical Documents* — Guillaume Gillet.

**Phase 1:** English MIMIC-IV notes (CSV benchmark, quantitative evaluation).  
**Phase 2:** French ophthalmological PDF notes from Cliniques universitaires Saint-Luc (qualitative) + French textbook clinical cases (semi-quantitative).

---

## Repository structure

```
snomed-ct-entity-linking-challenge-1.2.1/
│
├── Link/                              # Pipeline code
│   ├── constants.py                   # Shared constants (SNOMED type IDs, abbrev. expansion)
│   ├── preprocessor.py                # Text normalisation + candidate span extraction
│   ├── method1_dictionary.py          # Method 1: dictionary + fuzzy matching (rapidfuzz)
│   ├── method2_sapbert.py             # Method 2: SapBERT bi-encoder + FAISS
│   ├── method3_llm_rag.py             # Method 3: LLM NER + multilingual-e5-large FAISS
│   ├── method4_hybrid.py              # Method 4: LLM NER + tiered dict + FAISS (recommended)
│   ├── evaluator.py                   # Evaluation: mIoU + ranking table (MRR/NDCG/Hits@k)
│   └── run_pipeline.py                # End-to-end Phase 1 runner (CLI)
│
├── app.py                             # Streamlit UI — PDF/text upload, all 4 methods
├── voice_app.py                       # Streamlit UI — real-time voice + SNOMED detection
├── annotate_pdf.py                    # CLI PDF annotation tool (pdfplumber + OCR)
│
├── run_cas_cliniques.py               # Run all 4 methods on French textbook cases
├── evaluate_cas_cliniques.py          # Evaluate method results vs gold answers
├── evaluate_ai_baselines.py           # Evaluate Claude/GPT baseline results vs gold
├── verify_answers.py                  # Verify/correct SNOMED IDs in gold answer file
├── run_pdf_comparison.sh              # Batch script: run all 4 methods on 6 clinical PDFs
│
├── train_notes.csv                    # MIMIC-IV notes (note_id, text) — 272 notes
├── train_annotations.csv              # Gold annotations (note_id, start, end, span, concept_id)
├── cas_cliniques.txt                  # French textbook clinical cases (36 cases, EN+FR)
├── cas_cliniques_answer.txt           # Gold SNOMED answers for textbook cases (270 annotations)
├── cas_cliniques_results.json         # Method results on textbook cases (all 4 methods)
├── cas_cliniques_results.txt          # Human-readable summary of textbook case results
├── claude_sonnet46_results.txt        # Claude Sonnet 4.6 baseline outputs on textbook cases
├── chatgpt35_results.txt              # GPT-3.5 baseline outputs on textbook cases
│
├── SnomedCT_InternationalRF2_*/       # SNOMED CT International RF2 (download separately)
└── xSnomedCT_BelgianAlpha*/           # Belgian Alpha RF2 with French labels (download separately)
```

> **Note:** SNOMED RF2 files and the FAISS indexes in `Link/results/` are not tracked by git  
> (see `.gitignore`). Build them with the commands below before running the apps.

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows CMD / PowerShell
source .venv/Scripts/activate   # Windows Git Bash
source .venv/bin/activate       # Linux / macOS
```

### 2. Install dependencies

```bash
# Core (all methods)
pip install streamlit rapidfuzz pandas numpy

# Methods 2, 3, 4 (ML models + FAISS)
pip install torch transformers sentence-transformers faiss-cpu
pip install sentencepiece protobuf huggingface_hub

# Methods 3 & 4 (Ollama Python client)
pip install ollama

# PDF extraction + OCR (annotate_pdf.py, app.py, voice_app.py)
pip install pdfplumber pytesseract pdf2image pillow
```

> **Windows — Poppler:** `pdf2image` requires Poppler.  
> Download from https://github.com/oschwartz10612/poppler-windows/releases, extract,  
> and add the `bin/` folder to PATH.

> **Tesseract OCR (optional):** Required for scanned PDFs.  
> Download from https://github.com/UB-Mannheim/tesseract/wiki and install the French language pack.

### 3. LLM for Methods 3 & 4

Install [Ollama](https://ollama.com/) and pull the model:

```bash
ollama pull llama3.1:8b
# Ollama runs as a background service on http://localhost:11434
```

### 4. SNOMED RF2 data

Download separately (free academic registration):

| Resource | URL | Used by |
|----------|-----|---------|
| SNOMED CT International RF2 (EN) | https://www.snomed.org/snomed-ct/get-snomed | All methods |
| Belgian Alpha RF2 (FR labels) | https://www.snomed.org | French PDF annotation |

---

## Building the indexes

Run once before using the apps or evaluating:

```bash
# Method 1 — build dictionary (Tier 1 from training data + Tier 2 from SNOMED RF2)
python -m Link.method1_dictionary build \
    --annotations train_annotations.csv \
    --snomed-rf2 "SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20260301.txt" \
    --snomed-rf2-fr "xSnomedCT_BelgianAlphaReleasePackage_Alpha_20200930/Snapshot/Terminology/xsct2_Description_Snapshot_BelgianAlphaPackage_BE1000173_20200930.txt"

# Method 2 — build SapBERT FAISS index (~20 min on GPU)
python -m Link.method2_sapbert build \
    --snomed-rf2 "SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20260301.txt"

# Methods 3 & 4 — build multilingual-e5-large FAISS index (~40 min on GPU)
python -m Link.method3_llm_rag build \
    --snomed-rf2 "SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20260301.txt"
```

---

## Launching the apps

### Standard UI — `app.py`

Upload a PDF or paste clinical text, run any of the 4 methods, browse and export results.

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Voice UI — `voice_app.py`

Real-time voice dictation with live SNOMED entity detection.

```bash
streamlit run voice_app.py
```

Open http://localhost:8501 in **Chrome or Edge** (Firefox does not support the Web Speech API).

---

## Phase 1 — MIMIC-IV benchmark (English)

### Run full pipeline + evaluation

```bash
# All 4 methods, build + evaluate on the test split:
python -m Link.run_pipeline --method all \
    --annotations train_annotations.csv \
    --notes train_notes.csv \
    --snomed-rf2 "SnomedCT_InternationalRF2_PRODUCTION_20260301T120000Z/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20260301.txt"

# Single method (e.g. Method 4, skip rebuild):
python -m Link.run_pipeline --method 4 --no-rebuild \
    --annotations train_annotations.csv \
    --notes train_notes.csv
```

### Results (68 test notes, MIMIC-IV)

| Method | mIoU | MRR | Hits@1 | Hits@5 | Span Recall |
|--------|------|-----|--------|--------|-------------|
| M1 — Dict + Fuzzy | **0.360** | 0.734 | 0.708 | **0.767** | **0.489** |
| M2 — SapBERT hybrid | 0.102 | 0.260 | 0.199 | 0.359 | 0.459 |
| M3 — LLM + RAG | 0.016 | 0.414 | 0.314 | 0.561 | 0.045 |
| M4 — LLM + Dict + FAISS | 0.040 | **0.680** | **0.648** | 0.730 | 0.052 |

> **Annotation density paradox:** M4 has the best MRR and Hits@1 (concept linking quality)  
> but very low mIoU — the LLM extracts ~35 clinically relevant entities per note  
> while MIMIC-IV has ~340 exhaustive gold annotations per note.  
> mIoU measures benchmark compliance; Hits@5 measures clinical utility.

---

## Phase 2 — French clinical evaluation

### Annotate a single PDF

```bash
# Method 4 (recommended for French):
python annotate_pdf.py \
    --pdf "path/to/note.pdf" \
    --method 4 \
    --top 10 \
    --snomed-rf2-fr "xSnomedCT_BelgianAlphaReleasePackage_Alpha_20200930/Snapshot/Terminology/xsct2_Description_Snapshot_BelgianAlphaPackage_BE1000173_20200930.txt"

# Method 3:
python annotate_pdf.py --pdf "path/to/note.pdf" --method 3 --top 10

# All options:
python annotate_pdf.py --help
```

### Qualitative results — 6 real clinical PDFs (Saint-Luc, Epic ground truth)

| PDF | Diagnoses | M1 | M2 | M3 | M4 | M4* |
|-----|-----------|----|----|----|----|-----|
| PDF 1 — Retina | 3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| PDF 2 — Strabismus | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| PDF 4 — Neuro-ophth. | 5 | 3/5 | 2/5 | 3/5 | 3/5 | **5/5** |
| PDF 5 — Cornea | 1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 |
| PDF 6 — Neuro-ophth. | 6 | 2/6 | 2/6 | 2/6 | 2/6 | 2/6 |
| **Total** | **16** | 37.5% | 31.3% | 43.8% | 43.8% | **56.3%** |

> M4* is M4 with an expanded NER prompt that explicitly includes systemic comorbidities.

### Semi-quantitative results — 36 French textbook cases (270 gold annotations)

Run all methods on the textbook cases, then evaluate:

```bash
# Run all 4 methods (requires Ollama + built indexes):
python run_cas_cliniques.py

# Evaluate results vs gold answers:
python evaluate_cas_cliniques.py

# Evaluate Claude / GPT-3.5 AI baselines:
python evaluate_ai_baselines.py
```

| Section | Gold | M1 | M2 | M3 | M4 | Claude S. | GPT-3.5 |
|---------|------|----|----|----|----|-----------|---------|
| Ophthalmology (EN) | 5 | 80% | 60% | 80% | **80%** | 20% | 40% |
| Ophthalmology (FR) | 82 | 20% | 13% | 32% | **38%** | **38%** | 24% |
| Cardiology (FR) | 183 | 33% | 22% | 27% | **42%** | 31% | 30% |
| **Overall** | **270** | 30% | 20% | 29% | **42%** | 33% | 29% |

> Metric: top-5 hit rate (correct SNOMED concept anywhere in top-5 candidates).  
> M4 reranking does not change top-5 hit rate (it improves Hits@1 / MRR).  
> Claude and GPT-3.5 used the same NER prompt but without SNOMED RF2 retrieval.

### Verify / correct gold answer SNOMED IDs

```bash
# Check gold answers against the active SNOMED CT RF2 release:
python verify_answers.py
# Output: cas_cliniques_answer_verified.txt with corrected IDs
```

### Batch comparison on clinical PDFs

```bash
# Run all 4 methods on all 6 clinical PDFs (requires PDFs in the expected directory):
bash run_pdf_comparison.sh
# Results saved to Link/results/pdf_comparison/
```

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint (Methods 3 & 4) |
| `PYTHONIOENCODING` | system | Set to `utf-8` on Windows to avoid encoding errors |

---

## Data format

**Input:** `train_notes.csv` — columns: `note_id`, `text`  
**Annotations:** `train_annotations.csv` — columns: `annotation_id`, `note_id`, `start`, `end`, `span`, `concept_id`, `annotation_type` (`train` / `test`)

**Output per span:**
```json
{
  "start": 123, "end": 145,
  "span": "diabetic macular oedema",
  "candidates": [
    {"rank": 1, "concept_id": "232020008", "score": 0.95},
    {"rank": 2, "concept_id": "37231002",  "score": 0.87}
  ]
}
```
