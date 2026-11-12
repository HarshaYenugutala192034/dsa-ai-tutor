import os
from pathlib import Path

from dotenv import load_dotenv

from base import LLMProvider
from gemini_provider import GeminiProvider
from ollama_provider import OllamaProvider

# .env is in the main project folder, one level above session2
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_provider(provider_name: str) -> LLMProvider:
    """Return the provider object that matches the given name."""
    name = provider_name.strip().lower()

    if name == "gemini":
        # Read Gemini API key from the .env file
        api_key = os.getenv("GEMINI_API_KEY")

        # Validate API key
        if not api_key:
            raise ValueError("Gemini API key not found in .env file")

        return GeminiProvider(api_key=api_key)

    if name == "ollama":
        return OllamaProvider()

    raise ValueError(
        f"Unknown provider '{provider_name}'. Choose 'gemini' or 'ollama'."
    )
