# ============================================================
# Plant Mamba LLM - Automated Setup for Ollama (PowerShell)
# ============================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "     PLANT MAMBA LLM - AUTOMATED SETUP FOR OLLAMA          " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if Ollama is accessible
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Host "[ERROR] Ollama command not found in PATH." -ForegroundColor Red
    Write-Host "Please install Ollama from https://ollama.com/download/windows" -ForegroundColor Yellow
    Write-Host "If installed, launch Ollama desktop or restart your terminal."
    Read-Host "Press Enter to exit"
    exit 1
}

# 2. Pull base model
Write-Host "[1/3] Pulling base model (llama3.2:3b)..." -ForegroundColor Yellow
ollama pull llama3.2:3b

# 3. Create model from Modelfile
Write-Host ""
Write-Host "[2/3] Building custom model 'plant-mamba-llm' from Modelfile..." -ForegroundColor Yellow
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
ollama create plant-mamba-llm -f Modelfile

# 4. Launch model
Write-Host ""
Write-Host "[3/3] Launching 'plant-mamba-llm'..." -ForegroundColor Green
Write-Host "Type /bye or press Ctrl+C to exit.`n"
ollama run plant-mamba-llm
