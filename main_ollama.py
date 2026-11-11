import ollama

# Model downloaded with: ollama pull llama3.2
MODEL_NAME = "llama3.2"

SYSTEM_PROMPT = (
    "You are a friendly Data Structures and Algorithms tutor. "
    "Explain concepts in simple language with examples and analogies."
)

# Display application title
print("\n===== DSA AI TUTOR (OLLAMA) =====")

# Accept student's question
student_question = input("\nWhat would you like to learn today? ").strip()

if not student_question:
    raise SystemExit("\nPlease enter a question and run the program again.")

# Send request to the local Ollama model
try:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": student_question},
        ],
    )
except Exception as error:
    raise SystemExit(
        f"\nRequest to Ollama failed:\n{error}\n\n"
        "Check that Ollama is running and that the model is downloaded "
        f"(ollama pull {MODEL_NAME})."
    )

# Display generated response
print("\n===== DSA AI TUTOR RESPONSE =====\n")
print(response.message.content or "No text was returned. Try rephrasing your question.")

# Display token usage information
print("\n===== USAGE INFORMATION =====")
print("Input tokens:", response.prompt_eval_count or "N/A")
print("Output tokens:", response.eval_count or "N/A")