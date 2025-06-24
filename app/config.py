
import os
from dotenv import load_dotenv
load_dotenv()
class Settings:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    WHATSAPP_VENDOR = os.getenv("WHATSAPP_VENDOR", "twilio")
    STORE_ID = os.getenv("STORE_ID", "kong-food-store-1")
    MENU_KB_FILE = os.getenv("MENU_KB_FILE", f"app/knowledge_base/{STORE_ID}_menu_kb.json")
settings = Settings()
