import time

from dotenv import load_dotenv

from client import LLMClient
from factory import get_llm_provider
from request_logger import RequestLogger

load_dotenv()

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

prompt = input("Enter your Question ")

# Send the request and measure the time it takes
start = time.perf_counter()
try:
    answer = client.generate_response(prompt)
    status, error_text = "success", None
except Exception as error:
    answer, status, error_text = "", "failed", str(error)
latency_ms = (time.perf_counter() - start) * 1000

if status == "success":
    print("\n--- Response ---")
    print(answer)
else:
    print("\nRequest failed:", error_text)

print(f"\nProvider: {provider_name} | Latency: {latency_ms:.2f} ms")

# Save the request in MongoDB
if logger is not None:
    try:
        record_id = logger.log(
            provider=provider_name,
            model=getattr(provider, "model_name", "unknown"),
            prompt=prompt,
            response=answer,
            latency_ms=latency_ms,
            status=status,
            error=error_text,
        )
        print("Request logged. Record ID:", record_id)
    except Exception as error:
        print("MongoDB logging failed:", error)
    finally:
        logger.close()
