import os

from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available Models:\n")

for model in client.models.list():
    print(f"Model Name: {model.name}")
    print(f"Description: {model.description}")
    print("-" * 40)
