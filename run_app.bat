@echo off
title Plant Mamba Diagnostic App
echo Starting Plant Mamba Web Application...
cd /d "%~dp0core_engine"
"C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
pause
