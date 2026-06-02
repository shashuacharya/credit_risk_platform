"""
helpers.py — Shared utility functions.
"""

import os
import json
import pandas as pd


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(obj: dict | list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def df_info(df: pd.DataFrame) -> dict:
    """Return a serializable summary of a dataframe."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
    }


def risk_color(band: str) -> str:
    return {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}.get(band, "#95a5a6")
