import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(uri, serverSelectionTimeoutMS=8000)
client.admin.command("ping")

collection = client["AIMODEL"]["PROVIDER_LOGS"]

print("\n========== LATEST 10 REQUESTS ==========")

for record in collection.find().sort("_id", -1).limit(10):
    print("\n--------------------------------------")
    print("Provider:", record["provider"])
    print("Model:", record["model"])
    print("Prompt:", record["prompt"])
    print("Response:", (record["response"] or "")[:200])
    print("Latency (ms):", record["latency_ms"])
    print("Status:", record["status"])
    print("Time:", record["request_time"])

client.close()
