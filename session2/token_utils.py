import tiktoken

# EXAMPLE values only. Check each provider's official pricing and docs pages
# before relying on them, because prices and limits change.
MODEL_INFO = {
    "gemini": {
        "model": "gemini-3.1-flash-lite",
        "context": 1_000_000,
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
    },
    "gpt": {
        "model": "gpt-4o-mini",
        "context": 128_000,
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
    },
    "ollama": {
        "model": "llama3.2",
        "context": 4_096,  # local default window is small unless you raise num_ctx
        "input_per_1m": 0.0,
        "output_per_1m": 0.0,
    },
}

# BPE tokenizer used by recent OpenAI models
_ENCODING = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a text (an estimate for non-OpenAI models)."""
    return len(_ENCODING.encode(text or ""))


def estimate_cost(provider_name: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD = tokens x price per 1 million tokens."""
    info = MODEL_INFO[provider_name]
    return (
        input_tokens * info["input_per_1m"]
        + output_tokens * info["output_per_1m"]
    ) / 1_000_000
