import time

from dotenv import load_dotenv

from client import LLMClient
from factory import get_llm_provider
from request_logger import RequestLogger
from token_utils import MODEL_INFO, count_tokens, estimate_cost

load_dotenv()

# Used only for the cost forecast before sending the request
ASSUMED_OUTPUT_TOKENS = 500

# Connect to MongoDB (the app still works if MongoDB is down)
try:
    logger = RequestLogger()
except Exception as error:
    print("MongoDB connection failed:", error)
    logger = None

print("\n1. Gemini (Hosted LLM)")
print("2. Ollama (Local LLM)")
print("3. GPT (OpenAI Hosted LLM)")

choice = input("\nSelect your LLM provider (1, 2 or 3): ").strip()

if choice == "1":
    provider_name = "gemini"
elif choice == "2":
    provider_name = "ollama"
elif choice == "3":
    provider_name = "gpt"
else:
    raise ValueError("Invalid selection. Please enter 1, 2 or 3.")

provider = get_llm_provider(provider_name)
client = LLMClient(provider)
info = MODEL_INFO[provider_name]

prompt = input("Enter your Question ")

# ---- Before sending: token count, context check, cost forecast ----
input_tokens = count_tokens(prompt)
forecast = estimate_cost(provider_name, input_tokens, ASSUMED_OUTPUT_TOKENS)

print("\n===== BEFORE SENDING =====")
print(f"Model:                {info['model']}")
print(f"Input tokens (est.):  {input_tokens}")
print(f"Context window:       {info['context']:,} tokens")
print(f"Forecast cost:        ${forecast:.6f} "
      f"(assuming {ASSUMED_OUTPUT_TOKENS} output tokens)")

if input_tokens + ASSUMED_OUTPUT_TOKENS > info["context"]:
    raise SystemExit("\nThis prompt is too long for the model's context window.")

# ---- Send the request ----
start = time.perf_counter()
try:
    answer = client.generate_response(prompt)
    status, error_text = "success", None
except Exception as error:
    answer, status, error_text = "", "failed", str(error)
latency_ms = (time.perf_counter() - start) * 1000

# ---- After the response: output tokens, real cost, context usage ----
output_tokens = count_tokens(answer)
cost = estimate_cost(provider_name, input_tokens, output_tokens) if status == "success" else 0.0
context_used = (input_tokens + output_tokens) / info["context"] * 100

if status == "success":
    print("\n--- Response ---")
    print(answer)
else:
    print("\nRequest failed:", error_text)

print("\n===== AFTER RESPONSE =====")
print(f"Status:               {status}")
print(f"Input tokens (est.):  {input_tokens}")
print(f"Output tokens (est.): {output_tokens}")
print(f"Total tokens:         {input_tokens + output_tokens}")
print(f"Context used:         {context_used:.3f}%")
print(f"Estimated cost:       ${cost:.6f}")
print(f"Latency:              {latency_ms:.2f} ms")

# ---- Save everything in MongoDB ----
if logger is not None:
    try:
        record_id = logger.log(
            provider=provider_name,
            model=getattr(provider, "model_name", info["model"]),
            prompt=prompt,
            response=answer,
            latency_ms=latency_ms,
            status=status,
            error=error_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            context_window=info["context"],
            token_source="estimated (tiktoken o200k_base)",
        )
        print("Request logged. Record ID:", record_id)
    except Exception as error:
        print("MongoDB logging failed:", error)
    finally:
        logger.close()
