@echo off
cd /d "%~dp0"
echo Starting SNOMED CT Linker...
"C:\Users\guill\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
