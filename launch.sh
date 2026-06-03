#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "Starting SNOMED CT Linker..."
python3 -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
