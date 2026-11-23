from pathlib import Path as _P

# Token estimate using the BPE tokenizer trained in session4.
# Falls back to characters / 4 if that file is missing.
_TOKENIZER_FILE = (
    _P(__file__).resolve().parent.parent
    / "session4" / "model" / "tokenization" / "output" / "bpe_tokenizer.json"
)


def estimate_tokens(text):
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(_TOKENIZER_FILE))
        return max(1, len(tokenizer.encode(text).ids))
    except Exception:
        return max(1, len(text) // 4)

import ollama
import time

from monitoring_config import (
    MODEL_NAME,
    CONTEXT_WINDOW_SIZE,
    RESERVED_OUTPUT_TOKENS,
    INPUT_COST_PER_1K_TOKENS,
    OUTPUT_COST_PER_1K_TOKENS
)


def generate_response_with_metrics(prompt):

    # STEP 1: Estimate input tokens before sending request
    estimated_input_tokens = estimate_tokens(prompt)

    available_input_tokens = (
        CONTEXT_WINDOW_SIZE - RESERVED_OUTPUT_TOKENS
    )

    print("\n--- Context Window Check ---")

    print("Context Window Size:", CONTEXT_WINDOW_SIZE)
    print("Estimated Input Tokens:", estimated_input_tokens)
    print("Reserved Output Tokens:", RESERVED_OUTPUT_TOKENS)
    print("Available Input Tokens:", available_input_tokens)

    # STEP 2: Check context window
    if estimated_input_tokens > available_input_tokens:

        raise ValueError(
            "Context window limit exceeded! "
            "Please reduce your prompt."
        )

    print("Context Window Check: PASSED")

    # STEP 3: Generate response using Ollama
    start_time = time.perf_counter()

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "num_ctx": CONTEXT_WINDOW_SIZE,
            "num_predict": RESERVED_OUTPUT_TOKENS
        }
    )

    end_time = time.perf_counter()

    # STEP 4: Extract actual usage metrics
    input_tokens = response.prompt_eval_count
    output_tokens = response.eval_count

    total_tokens = input_tokens + output_tokens

    # STEP 5: Calculate estimated input cost
    input_cost = (
        input_tokens / 1000
    ) * INPUT_COST_PER_1K_TOKENS

    # STEP 6: Calculate estimated output cost
    output_cost = (
        output_tokens / 1000
    ) * OUTPUT_COST_PER_1K_TOKENS

    total_cost = input_cost + output_cost

    # STEP 7: Calculate context utilization
    context_used = input_tokens + output_tokens

    context_usage_percentage = (
        context_used / CONTEXT_WINDOW_SIZE
    ) * 100

    # STEP 8: Store all metrics
    result = {

        "provider": "Ollama",
        "model": MODEL_NAME,

        "prompt": prompt,
        "response": response.response,

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,

        "context_window_size": CONTEXT_WINDOW_SIZE,
        "context_used": context_used,
        "context_usage_percentage": round(
            context_usage_percentage, 2
        ),

        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(total_cost, 4),

        "total_duration_ms": (
            end_time - start_time
        ) * 1000,

        "ollama_total_duration_ms": (
            response.total_duration / 1_000_000
        ),

        "prompt_eval_duration_ms": (
            response.prompt_eval_duration / 1_000_000
        ),

        "generation_duration_ms": (
            response.eval_duration / 1_000_000
        )
    }

    return result


if __name__ == "__main__":

    question = input("Enter your Prompt: ")

    try:

        result = generate_response_with_metrics(question)

        print("\n--- Llama 3.2 Response ---")
        print(result["response"])

        print("\n--- Actual Usage Metrics ---")

        print("Provider:", result["provider"])
        print("Model:", result["model"])

        print("Input Tokens:", result["input_tokens"])
        print("Output Tokens:", result["output_tokens"])
        print("Total Tokens:", result["total_tokens"])

        print("\n--- Context Window Metrics ---")

        print("Context Window Size:",
              result["context_window_size"])

        print("Context Used:",
              result["context_used"])

        print("Context Usage:",
              result["context_usage_percentage"], "%")

        print("\n--- Estimated Demo Cost ---")

        print("Input Cost: ₹", result["input_cost"])
        print("Output Cost: ₹", result["output_cost"])
        print("Total Cost: ₹", result["total_cost"])

        print("\n--- Performance Metrics ---")

        print("Application Latency (ms):",
              round(result["total_duration_ms"], 2))

        print("Ollama Generation Duration (ms):",
              round(result["ollama_total_duration_ms"], 2))

    except Exception as e:

        print("\nLLM Request Failed:", e)
