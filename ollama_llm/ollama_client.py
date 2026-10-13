"""
Ollama Client for Plant Mamba LLM
Provides seamless communication with the local Ollama inference server.
Works with zero external dependencies (uses Python standard library urllib/json).
"""

import json
import urllib.error
import urllib.request
from typing import Dict, Generator, List, Optional


class PlantMambaLLMClient:
    """Client for local Ollama LLM specialized for Plant Mamba diagnoses."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "plant-mamba-llm"):
        self.host = host.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        """Check if Ollama service is running locally."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List all models currently installed in Ollama."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name") for m in data.get("models", [])]
        except Exception:
            return []

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a complete text response (non-streaming)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "")

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream chunks of text token-by-token."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break

    def get_diagnostic_advice(
        self,
        disease_name: str,
        confidence: float,
        crop_type: Optional[str] = None,
        severity_level: Optional[str] = None,
        extra_observations: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive clinical agronomist advice for a detected disease.
        """
        prompt = f"""
[DIAGNOSTIC REPORT FROM PLANT MAMBA VISION SYSTEM]
- Identified Condition: {disease_name}
- Prediction Confidence: {confidence:.2%}
- Target Crop: {crop_type or 'Identified from leaf imagery'}
- Assessed Severity: {severity_level or 'Moderate'}
- Field Observations: {extra_observations or 'None provided'}

Please provide:
1. Disease Overview and Pathogen Identification.
2. Immediate Action Plan (First 24-48 hours).
3. Organic and Bio-control Recommendations.
4. Chemical Treatment & Spray Regimen (Active ingredients, dosage, safety precautions).
5. Long-term Field Prevention & Cultural Practices.
"""
        return self.generate(prompt)


# Quick CLI test
if __name__ == "__main__":
    client = PlantMambaLLMClient()
    print("Checking Ollama status...")
    if client.is_available():
        print("[OK] Ollama is active!")
        installed = client.list_models()
        print(f"Available models: {installed}")
        if any("plant-mamba-llm" in m for m in installed):
            print("\nGenerating sample consultation...")
            res = client.get_diagnostic_advice("Tomato Early Blight (Alternaria solani)", 0.94)
            print(res)
        else:
            print("Note: 'plant-mamba-llm' not found yet. Run 'setup_and_run.bat' to build it.")
    else:
        print("[!] Ollama is not running on http://localhost:11434.")
        print("Please start Ollama desktop or run 'ollama serve'.")
