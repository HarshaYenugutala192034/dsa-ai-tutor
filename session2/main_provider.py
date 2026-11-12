from dotenv import load_dotenv

from client import LLMClient
from factory import get_llm_provider

load_dotenv()

print("1. Gemini (Hosted LLM)")
print("2. Ollama (Local LLM)")

choice = input("\nSelect your LLM provider (1 or 2): ").strip()

if choice == "1":
    provider_name = "gemini"
elif choice == "2":
    provider_name = "ollama"
else:
    raise ValueError("Invalid selection. Please enter 1 or 2.")

provider = get_llm_provider(provider_name)
client = LLMClient(provider)

prompt = input("Enter your Question ")
answer = client.generate_response(prompt)

print(answer)
