import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# .env is in the main project folder, one level above session2
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class RequestLogger:
    """Saves every LLM request (any provider) in MongoDB Atlas."""

    def __init__(self):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.client = MongoClient(uri, serverSelectionTimeoutMS=8000)

        # Verify MongoDB connection
        self.client.admin.command("ping")

        self.collection = self.client["AIMODEL"]["PROVIDER_LOGS"]
        print("MongoDB Logger connected successfully!")

    def log(self, provider, model, prompt, response, latency_ms,
            status="success", error=None):
        record = {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "response": response,
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "error": error,
            "request_time": datetime.now(timezone.utc),
        }
        return self.collection.insert_one(record).inserted_id

    def close(self):
        self.client.close()
