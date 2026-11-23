import os
from pathlib import Path as _P

from dotenv import load_dotenv

load_dotenv(_P(__file__).resolve().parent.parent / ".env")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

from pymongo import MongoClient


# STEP 1: Configuration

NUMBER_OF_USERS = 1000
REQUESTS_PER_USER_PER_DAY = 2
DAYS_PER_MONTH = 30

INPUT_COST_PER_1K_TOKENS = 0.50
OUTPUT_COST_PER_1K_TOKENS = 1.50


# STEP 2: Connect to MongoDB

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000
)

db = client["AIMODEL"]
collection = db["GENAI"]

client.admin.command("ping")

print("Connected to MongoDB successfully!")


# STEP 3: Retrieve existing usage records

records = list(collection.find(
    {
        "input_tokens": {"$exists": True},
        "output_tokens": {"$exists": True}
    }
))

if len(records) == 0:

    print("No usage records found!")
    client.close()
    exit()


# STEP 4: Calculate average token consumption

total_input_tokens = sum(
    record["input_tokens"] for record in records
)

total_output_tokens = sum(
    record["output_tokens"] for record in records
)

total_requests = len(records)

average_input_tokens = (
    total_input_tokens / total_requests
)

average_output_tokens = (
    total_output_tokens / total_requests
)


# STEP 5: Calculate daily usage for 1000 users

daily_requests = (
    NUMBER_OF_USERS * REQUESTS_PER_USER_PER_DAY
)

daily_input_tokens = (
    daily_requests * average_input_tokens
)

daily_output_tokens = (
    daily_requests * average_output_tokens
)


# STEP 6: Calculate simulated daily cost

daily_input_cost = (
    daily_input_tokens / 1000
) * INPUT_COST_PER_1K_TOKENS

daily_output_cost = (
    daily_output_tokens / 1000
) * OUTPUT_COST_PER_1K_TOKENS

daily_total_cost = (
    daily_input_cost + daily_output_cost
)


# STEP 7: Calculate monthly budget

monthly_requests = daily_requests * DAYS_PER_MONTH

monthly_input_cost = daily_input_cost * DAYS_PER_MONTH

monthly_output_cost = daily_output_cost * DAYS_PER_MONTH

monthly_total_cost = daily_total_cost * DAYS_PER_MONTH


# STEP 8: Display budget report

print("\n========== OLLAMA BUDGET REPORT ==========")

print("\n--- Configuration ---")

print("Number of Users:", NUMBER_OF_USERS)
print("Requests Per User Per Day:",
      REQUESTS_PER_USER_PER_DAY)

print("Monthly Days:", DAYS_PER_MONTH)

print("\n--- Historical Usage ---")

print("Total MongoDB Records:", total_requests)

print("Average Input Tokens:",
      round(average_input_tokens, 2))

print("Average Output Tokens:",
      round(average_output_tokens, 2))


print("\n--- Estimated Daily Usage ---")

print("Daily Requests:", daily_requests)

print("Daily Input Tokens:",
      round(daily_input_tokens))

print("Daily Output Tokens:",
      round(daily_output_tokens))


print("\n--- Simulated Daily Cost ---")

print("Input Cost: Rs.", round(daily_input_cost, 2))

print("Output Cost: Rs.", round(daily_output_cost, 2))

print("Total Daily Cost: Rs.",
      round(daily_total_cost, 2))


print("\n--- Simulated Monthly Budget ---")

print("Monthly Requests:", monthly_requests)

print("Monthly Input Cost: Rs.",
      round(monthly_input_cost, 2))

print("Monthly Output Cost: Rs.",
      round(monthly_output_cost, 2))

print("Total Monthly Budget: Rs.",
      round(monthly_total_cost, 2))


print("\n==========================================")

client.close()
