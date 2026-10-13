"""
Interactive CLI for Plant Mamba LLM Assistant
Allows farmers and agronomists to chat directly with the local Ollama LLM.
"""

import sys
from ollama_client import PlantMambaLLMClient


def main():
    print("=" * 65)
    print("      PLANT MAMBA AI - LOCAL AGRONOMIST ADVISOR (OLLAMA)")
    print("=" * 65)
    print("Type your questions or disease symptoms below.")
    print("Commands: 'exit' or 'quit' to close | 'help' for examples\n")

    client = PlantMambaLLMClient(model="plant-mamba-llm")

    if not client.is_available():
        print("[!] Warning: Ollama does not seem to be running on http://localhost:11434")
        print("Please ensure Ollama is installed and running on your system.")
        print("Run 'setup_and_run.bat' to build the model first if you haven't yet.\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye! Happy farming.")
                break

            if user_input.lower() == "help":
                print("\nSample queries:")
                print(" - 'My tomato leaves have concentric brown rings and yellow margins. What is this?'")
                print(" - 'What is the dosage for Mancozeb on potato blight?'")
                print(" - 'Explain how to prevent powdery mildew organically in bell peppers.'")
                continue

            print("\nPlant Mamba AI > ", end="", flush=True)
            for token in client.generate_stream(user_input):
                print(token, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    main()
