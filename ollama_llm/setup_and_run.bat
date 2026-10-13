@echo off
title Plant Mamba LLM - Setup and Launch
echo ============================================================
echo      PLANT MAMBA LLM - AUTOMATED SETUP FOR OLLAMA
echo ============================================================
echo.

:: 1. Check if Ollama command is available
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Ollama was not found in your system PATH.
    echo Please install Ollama from: https://ollama.com/download/windows
    echo If already installed, make sure it is added to your PATH or running in your system tray.
    pause
    exit /b 1
)

:: 2. Pull base model
echo [1/3] Pulling base model (llama3.2:3b)...
ollama pull llama3.2:3b
if %errorlevel% neq 0 (
    echo [ERROR] Failed to pull llama3.2:3b. Ensure your internet connection is active.
    pause
    exit /b 1
)

:: 3. Build custom LLM from Modelfile
echo.
echo [2/3] Building custom LLM 'plant-mamba-llm' from Modelfile...
ollama create plant-mamba-llm -f Modelfile
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create custom model from Modelfile.
    pause
    exit /b 1
)

:: 4. Start interactive session
echo.
echo [3/3] Success! Starting Plant Mamba LLM session...
echo ============================================================
ollama run plant-mamba-llm
pause
