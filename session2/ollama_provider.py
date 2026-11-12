import ollama

from base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content or ""
