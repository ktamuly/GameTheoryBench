"""
Utility to load environment variables from .env file.
Import this at the top of main.py to automatically load API keys.
"""
import os
from pathlib import Path


def load_dotenv():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"

    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()


# Auto-load when imported
load_dotenv()
