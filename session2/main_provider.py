from gemini_provider import GeminiProvider
from ollama_provider import OllamaProvider

TUTOR_PROMPT = (
    "You are a friendly Data Structures and Algorithms tutor. "
    "Explain concepts in simple language with examples and analogies.\n\n"
    "Student question: {question}"
)

print("\n===== DSA AI TUTOR =====")
choice = input("\nChoose provider (1 = Gemini, 2 = Ollama): ").strip()

try:
    if choice == "1":
        provider = GeminiProvider()
    elif choice == "2":
        provider = OllamaProvider()
    else:
        raise SystemExit("\nInvalid choice. Enter 1 or 2.")

    question = input("\nWhat would you like to learn today? ").strip()
    if not question:
        raise SystemExit("\nPlease enter a question and run the program again.")

    answer = provider.generate_response(TUTOR_PROMPT.format(question=question))
except SystemExit:
    raise
except Exception as error:
    raise SystemExit(f"\nRequest failed:\n{error}")

print("\n===== DSA AI TUTOR RESPONSE =====\n")
print(answer or "No text was returned. Try rephrasing your question.")
