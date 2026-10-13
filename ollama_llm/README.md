# Plant Mamba AI - Custom Ollama LLM Assistant

[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-blue.svg)](https://ollama.com)
[![Base Model](https://img.shields.io/badge/Base-Llama%203.2%203B-green.svg)](https://ollama.com/library/llama3.2)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

An intelligent, local agricultural LLM assistant customized to support the **Plant Mamba** leaf disease diagnostic framework. It provides expert plant pathology insights, treatment regimens (organic and chemical), pathogen biology, and preventive field advice.

---

## 🚀 Features

- **Local & Private**: Runs 100% offline on your machine using Ollama without sending farm data to external clouds.
- **Agricultural Domain Knowledge**: Tailored system instructions designed for crop pathology, disease stages, biological cycles, and chemical safety.
- **Zero-Dependency Python Client**: Integrates directly with existing Python apps, Streamlit dashboards, or command-line scripts via the local REST API (`http://localhost:11434`).
- **One-Click Setup**: Automated scripts (`setup_and_run.bat` / `.ps1`) to pull the base model, build the custom LLM, and launch it instantly.

---

## 📁 Project Structure

```
ollama_llm/
├── Modelfile             # Ollama blueprint (parameters, base weights, system prompt)
├── ollama_client.py      # Python API wrapper (streaming & non-streaming)
├── interactive_chat.py   # CLI terminal chat interface for agronomists/farmers
├── setup_and_run.bat     # Windows 1-click build and launch script
├── setup_and_run.ps1     # PowerShell setup script
└── README.md             # Documentation and usage guide
```

---

## 🛠️ Getting Started

### 1. Prerequisites
Install Ollama for Windows from [ollama.com/download/windows](https://ollama.com/download/windows).

### 2. Build the Custom Model
Run the automated script:
```cmd
cd ollama_llm
setup_and_run.bat
```
*Or manually execute:*
```bash
ollama pull llama3.2:3b
ollama create plant-mamba-llm -f Modelfile
```

### 3. Chat with the Model

**Via Ollama CLI:**
```bash
ollama run plant-mamba-llm
```

**Via Python Interactive Chat:**
```bash
python interactive_chat.py
```

---

## 💻 Python Integration

```python
from ollama_client import PlantMambaLLMClient

# Initialize client
client = PlantMambaLLMClient()

# Generate advice for a detected disease
diagnosis = client.get_diagnostic_advice(
    disease_name="Tomato Early Blight",
    confidence=0.96,
    crop_type="Tomato",
    severity_level="High"
)
print(diagnosis)
```

---

## 🌐 Pushing to GitHub & Ollama Registry

### Push to GitHub
```bash
git add ollama_llm/
git commit -m "feat: add custom Ollama LLM assistant"
git push origin main
```

### (Optional) Push to Ollama Hub
If you want your model to be publicly downloadable via `ollama run your-name/plant-mamba-llm`:
1. Create an account at [ollama.com](https://ollama.com).
2. Copy your key: `cat ~/.ollama/id_ed25519.pub` and add it to your Ollama account settings.
3. Push:
   ```bash
   ollama cp plant-mamba-llm <your-username>/plant-mamba-llm
   ollama push <your-username>/plant-mamba-llm
   ```
