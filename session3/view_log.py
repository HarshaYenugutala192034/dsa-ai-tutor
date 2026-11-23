import os
from pathlib import Path as _P

from dotenv import load_dotenv

load_dotenv(_P(__file__).resolve().parent.parent / ".env")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

from pymongo import MongoClient


# Connect to MongoDB
client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000
)

db = client["AIMODEL"]
collection = db["GENAI"]

client.admin.command("ping")

print("Connected to MongoDB successfully!")

# Retrieve records
records = collection.find().sort("_id", -1)

print("\n========== STORED LLM REQUEST LOGS ==========")

for record in records:

    print("\n--------------------------------------")

    print("Record ID:", record["_id"])
    print("Provider:", record["provider"])
    print("Model:", record["model"])

    print("Prompt:", record["prompt"])
    print("Response:", record["response"])

    print("\n--- Token Usage ---")

    print("Input Tokens:", record["input_tokens"])
    print("Output Tokens:", record["output_tokens"])
    print("Total Tokens:", record.get("total_tokens", "N/A"))

    print("\n--- Context Window ---")

    print("Context Window Size:",
          record.get("context_window_size", "N/A"))

    print("Context Used:",
          record.get("context_used", "N/A"))

    print("Context Usage:",
          record.get("context_usage_percentage", "N/A"), "%")

    print("\n--- Estimated Demo Cost ---")

    print("Input Cost: ₹", record.get("input_cost", "N/A"))
    print("Output Cost: ₹", record.get("output_cost", "N/A"))
    print("Total Cost: ₹", record.get("total_cost", "N/A"))

    print("\n--- Performance ---")

    print("Application Latency (ms):",
          round(record["total_duration_ms"], 2))

    print("Status:", record["status"])
    print("Request Time:", record["request_time"])

client.close()
