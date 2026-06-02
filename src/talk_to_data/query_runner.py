"""
query_runner.py — Execute SQL queries against a SQLite database loaded from the dataset.
"""

import os
import logging
import sqlite3
import json
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "data/credit_risk.db")


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def run_query(sql: str, max_rows: int = 500) -> tuple[list[dict], str]:
    """
    Execute a SQL query and return results as list of dicts.

    Returns
    -------
    rows   : list of row dicts
    error  : empty string on success
    """
    try:
        conn = get_connection()
        df = pd.read_sql_query(sql, conn)
        conn.close()
        df = df.head(max_rows)
        rows = df.to_dict(orient="records")
        return rows, ""
    except Exception as e:
        logger.error(f"Query execution error: {e}\nSQL: {sql}")
        return [], str(e)


def rows_to_json(rows: list[dict]) -> str:
    return json.dumps(rows, default=str, indent=2)


def initialize_db(csv_path: str | None = None):
    """
    Load application_train.csv (or joined data) into SQLite.
    Called once on startup if DB doesn't exist.
    """
    if os.path.exists(DB_PATH):
        logger.info(f"DB already exists at {DB_PATH}")
        return

    data_dir = os.environ.get("DATA_DIR", "data")
    csv_path = csv_path or os.path.join(data_dir, "application_train.csv")

    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found at {csv_path} — DB not initialized.")
        return

    logger.info(f"Initializing SQLite DB from {csv_path}...")
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("applications", conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"DB initialized with {len(df)} rows at {DB_PATH}")
