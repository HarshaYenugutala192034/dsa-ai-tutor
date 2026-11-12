from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for all LLM providers (Gemini, Ollama, etc.).

    Every provider must inherit from this class and implement
    generate_response(), so all providers work the same way.
    """

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Send a prompt to the model and return its response as text."""
        pass