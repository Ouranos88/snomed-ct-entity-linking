#!/usr/bin/env bash
# run_pdf_comparison.sh
# Runs all 4 methods on all 6 French ophthalmology PDFs and saves output to
#   Link/results/pdf_comparison/<pdf_name>_method<N>.txt
#
# Usage (from project root, venv activated):
#   bash run_pdf_comparison.sh
#
# Runtime estimate:
#   Methods 1 & 2 : ~1–2 min per PDF   (fast)
#   Methods 3 & 4 : ~3–8 min per PDF   (Ollama calls)
#   Total          : ~1–2 h for all 24 runs

set -euo pipefail

# Force UTF-8 output so Unicode characters (arrows, accents) are written correctly
export PYTHONIOENCODING=utf-8

PDF_DIR="Exemples de lettres et notes v2-pages"
OUT_DIR="Link/results/pdf_comparison"
FR_RF2="xSnomedCT_BelgianAlphaReleasePackage_Alpha_20200930/Snapshot/Terminology/xsct2_Description_Snapshot_BelgianAlphaPackage_BE1000173_20200930.txt"
TOP=10

mkdir -p "$OUT_DIR"

for i in 1 2 3 4 5 6; do
    PDF="${PDF_DIR}/Exemples de lettres et notes v2-pages-${i}.pdf"
    echo "========================================"
    echo "PDF $i / 6 : $PDF"
    echo "========================================"

    for METHOD in 1 2 3 4; do
        OUT="${OUT_DIR}/pdf${i}_method${METHOD}.txt"

        # Skip if already done (re-run with: rm -rf Link/results/pdf_comparison/)
        if [ -f "$OUT" ]; then
            echo "  [Method $METHOD] already done, skipping."
            continue
        fi

        echo "  [Method $METHOD] running..."
        python annotate_pdf.py \
            --pdf "$PDF" \
            --method "$METHOD" \
            --top "$TOP" \
            --snomed-rf2-fr "$FR_RF2" \
            --no-rerank \
            > "$OUT" 2>&1 \
            && echo "  [Method $METHOD] done → $OUT" \
            || echo "  [Method $METHOD] FAILED (see $OUT)"
    done
done

echo ""
echo "All done. Results in $OUT_DIR/"
