from base import LLMProvider


class LLMClient:
    """Holds the provider the user chose and sends prompts to it."""

    def __init__(self, provider: LLMProvider):
        # Make sure we received a real provider (Gemini, Ollama, etc.)
        if not isinstance(provider, LLMProvider):
            raise TypeError("provider must be an instance of LLMProvider")

        self.provider = provider

    def generate_response(self, prompt: str) -> str:
        # Do not send an empty prompt to the model
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # The chosen provider's generate_response() does the real work
        return self.provider.generate_response(prompt)
