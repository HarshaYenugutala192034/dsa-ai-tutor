from client import LLMClient
from factory import get_provider

TUTOR_PROMPT = (
    "You are a friendly Data Structures and Algorithms tutor. "
    "Explain concepts in simple language with examples and analogies.\n\n"
    "Student question: {question}"
)

PROVIDER_NAMES = {"1": "gemini", "2": "ollama"}

print("\n===== DSA AI TUTOR =====")
choice = input("\nChoose provider (1 = Gemini, 2 = Ollama): ").strip()

if choice not in PROVIDER_NAMES:
    raise SystemExit("\nInvalid choice. Enter 1 or 2.")

try:
    # The factory returns the provider the user chose
    provider = get_provider(PROVIDER_NAMES[choice])

    # The client holds that provider
    client = LLMClient(provider)

    question = input("\nWhat would you like to learn today? ").strip()
    if not question:
        raise SystemExit("\nPlease enter a question and run the program again.")

    answer = client.generate_response(TUTOR_PROMPT.format(question=question))
except SystemExit:
    raise
except Exception as error:
    raise SystemExit(f"\nRequest failed:\n{error}")

print("\n===== DSA AI TUTOR RESPONSE =====\n")
print(answer or "No text was returned. Try rephrasing your question.")
