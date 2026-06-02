"""
nl_to_sql.py - Convert natural language questions to SQL using Groq.
Includes hallucination guards and SQL validation.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

MAX_TOKENS = 512

ALLOWED_TABLES = ["applications", "bureau", "previous_application", "installments", "credit_card"]

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\b",
    re.IGNORECASE,
)


def _call_llm(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    """Make a call to Groq API."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def validate_sql(sql: str) -> tuple[bool, str]:
    """Basic safety and validity checks on generated SQL."""
    if FORBIDDEN_KEYWORDS.search(sql):
        return False, "Query contains forbidden write operations."
    if not re.search(r"\bSELECT\b", sql, re.IGNORECASE):
        return False, "Query does not contain SELECT."
    if not any(table in sql.lower() for table in ALLOWED_TABLES):
        return False, "Query does not reference any known table."
    return True, ""


def natural_language_to_sql(question: str) -> tuple[str, str]:
    """
    Convert a natural language question to a SQL query.

    Returns
    -------
    sql    : the generated (and validated) SQL string
    reason : empty string on success, error message on failure
    """
    from src.talk_to_data.prompt_templates import NL_TO_SQL_SYSTEM

    try:
        sql = _call_llm(
            system=NL_TO_SQL_SYSTEM,
            user=f"Question: {question}",
        )
        # Strip accidental markdown fences
        sql = re.sub(r"```(?:sql)?", "", sql).strip().rstrip(";") + ";"
        sql = re.sub(r"```", "", sql).strip()

        ok, reason = validate_sql(sql)
        if not ok:
            logger.warning(f"SQL validation failed: {reason}\nSQL: {sql}")
            return "", reason

        return sql, ""
    except Exception as e:
        logger.error(f"NL->SQL error: {e}")
        return "", str(e)


def generate_insight(question: str, result_json: str) -> str:
    """Turn a SQL result into a human-readable business insight."""
    from src.talk_to_data.prompt_templates import INSIGHT_SYSTEM

    try:
        insight = _call_llm(
            system=INSIGHT_SYSTEM,
            user=f"Question: {question}\n\nSQL Result (JSON):\n{result_json}",
            max_tokens=300,
        )
        return insight
    except Exception as e:
        logger.warning(f"Insight generation failed: {e}")
        return "Insight generation unavailable."