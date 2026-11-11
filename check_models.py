import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file next to this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env file")

client = genai.Client(api_key=api_key)

# List models that support generateContent
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name)