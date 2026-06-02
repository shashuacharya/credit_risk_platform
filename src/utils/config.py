"""
config.py — Central configuration pulled from environment variables.
"""

import os

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DB_PATH = os.environ.get("DB_PATH", "data/credit_risk.db")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8501"))
