
from ollama_metrics import generate_response_with_metrics
from mongodb_logger import MongoDBLogger


# STEP 1: Connect to MongoDB
try:

    logger = MongoDBLogger()

except Exception as e:

    print("MongoDB connection failed:", e)
    logger = None


# STEP 2: Get student's question
question = input("\nEnter your DSA question: ")


# STEP 3: Generate LLM response
try:

    result = generate_response_with_metrics(question)

    print("\n--- Llama 3.2 Response ---")
    print(result["response"])

    # Display token metrics
    print("\n--- Usage Metrics ---")

    print("Input Tokens:", result["input_tokens"])
    print("Output Tokens:", result["output_tokens"])
    print("Total Tokens:", result["total_tokens"])

    # Display context metrics
    print("\n--- Context Window Metrics ---")

    print("Context Window Size:",
          result["context_window_size"])

    print("Context Used:",
          result["context_used"])

    print("Context Usage:",
          result["context_usage_percentage"], "%")

    # Display cost metrics
    print("\n--- Estimated Demo Cost ---")

    print("Input Cost: ₹", result["input_cost"])
    print("Output Cost: ₹", result["output_cost"])
    print("Total Cost: ₹", result["total_cost"])

    # Display performance metrics
    print("\n--- Performance Metrics ---")

    print("Application Latency (ms):",
          round(result["total_duration_ms"], 2))

    print("Ollama Duration (ms):",
          round(result["ollama_total_duration_ms"], 2))

    # STEP 4: Store metrics in MongoDB
    if logger is not None:

        try:

            logger.log_request(result)

        except Exception as e:

            print("MongoDB logging failed:", e)

except Exception as e:

    print("\nLLM request failed:", e)


# STEP 5: Close MongoDB connection
if logger is not None:

    logger.client.close()
