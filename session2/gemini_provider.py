from google import genai

from base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text or ""
