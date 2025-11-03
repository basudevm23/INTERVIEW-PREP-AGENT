# setup_env.py
import os
from dotenv import load_dotenv

def setup_environment():
    """
    Loads environment variables from .env file and checks critical configs.
    """
    env_path = os.path.join(os.getcwd(), ".env")

    if not os.path.exists(env_path):
        raise FileNotFoundError(
            f".env file not found at {env_path}. Please create it with GEMINI_API_KEY, GEMINI_ENDPOINT, and LLM_LOCAL_MODEL."
        )

    load_dotenv(dotenv_path=env_path)

    print("✅ Loaded .env successfully.")
    # Check essential variables
    required_vars = ["LLM_LOCAL_MODEL", "LORA_MODEL_DIR"]
    for var in required_vars:
        val = os.getenv(var)
        if not val:
            print(f"⚠️ Warning: {var} is not set in .env (using default).")
        else:
            print(f"• {var} = {val}")

    # Gemini optional variables
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("🔑 Gemini API Key detected.")
    else:
        print("ℹ️  No GEMINI_API_KEY found — app will run in local-only mode.")

    gemini_endpoint = os.getenv("GEMINI_ENDPOINT")
    if gemini_endpoint:
        print(f"🌐 Gemini Endpoint set: {gemini_endpoint[:60]}...")
    else:
        print("ℹ️  No GEMINI_ENDPOINT set — using local fallback generator.")

if __name__ == "__main__":
    setup_environment()
