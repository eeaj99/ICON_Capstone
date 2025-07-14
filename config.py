import os
from dotenv import load_dotenv
import json

def load_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def load_document_config(json_path="document_map.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
