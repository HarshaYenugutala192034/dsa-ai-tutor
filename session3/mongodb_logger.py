import os
from pathlib import Path as _P

from dotenv import load_dotenv

load_dotenv(_P(__file__).resolve().parent.parent / ".env")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

from pymongo import MongoClient
from datetime import datetime, timezone


class MongoDBLogger:

    def __init__(self):

        self.client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000
        )

        self.db = self.client["AIMODEL"]
        self.collection = self.db["GENAI"]

        self.client.admin.command("ping")

        print("MongoDB Logger connected successfully!")

    def log_request(self, result):

        log_record = {

            "provider": result["provider"],
            "model": result["model"],

            "prompt": result["prompt"],
            "response": result["response"],

            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],

            "context_window_size": result["context_window_size"],
            "context_used": result["context_used"],
            "context_usage_percentage":
                result["context_usage_percentage"],

            "input_cost": result["input_cost"],
            "output_cost": result["output_cost"],
            "total_cost": result["total_cost"],

            "total_duration_ms": result["total_duration_ms"],

            "ollama_total_duration_ms":
                result["ollama_total_duration_ms"],

            "prompt_eval_duration_ms":
                result["prompt_eval_duration_ms"],

            "generation_duration_ms":
                result["generation_duration_ms"],

            "request_time": datetime.now(timezone.utc),

            "status": "success"
        }

        inserted_record = self.collection.insert_one(log_record)

        print("\nRequest logged successfully!")

        print("MongoDB Record ID:", inserted_record.inserted_id)
