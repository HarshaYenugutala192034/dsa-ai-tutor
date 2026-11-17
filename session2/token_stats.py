import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

client = MongoClient(
    os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
    serverSelectionTimeoutMS=8000,
)
client.admin.command("ping")
collection = client["AIMODEL"]["PROVIDER_LOGS"]

pipeline = [
    {
        "$group": {
            "_id": "$provider",
            "requests": {"$sum": 1},
            "failed": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}},
            "avg_latency_ms": {"$avg": "$latency_ms"},
            "input_tokens": {"$sum": {"$ifNull": ["$input_tokens", 0]}},
            "output_tokens": {"$sum": {"$ifNull": ["$output_tokens", 0]}},
            "cost": {"$sum": {"$ifNull": ["$estimated_cost_usd", 0]}},
        }
    },
    {"$sort": {"_id": 1}},
]

print("\n========== USAGE SUMMARY BY PROVIDER ==========")
print(f"{'Provider':<10}{'Requests':>9}{'Fail %':>8}{'Avg ms':>10}"
      f"{'In tok':>9}{'Out tok':>9}{'Cost $':>11}")

for row in collection.aggregate(pipeline):
    fail_pct = row["failed"] / row["requests"] * 100
    print(
        f"{row['_id']:<10}{row['requests']:>9}{fail_pct:>8.1f}"
        f"{row['avg_latency_ms']:>10.0f}{row['input_tokens']:>9}"
        f"{row['output_tokens']:>9}{row['cost']:>11.6f}"
    )

client.close()
