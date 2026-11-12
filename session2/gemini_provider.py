import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-3.1-flash-lite"):
        # .env is in the main project folder, one level above session2
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=True)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found in .env file")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text or ""
