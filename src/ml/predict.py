"""
predict.py — Inference pipeline: load model, score a single applicant.
Returns risk_score (0-100), risk_band, and SHAP explanation.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")

_model = None
_label_encoders = None
_features = None


def _load_artifacts():
    global _model, _label_encoders, _features
    if _model is not None:
        return

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    enc_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    features_path = os.path.join(MODEL_DIR, "features.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )

    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    with open(enc_path, "rb") as f:
        _label_encoders = pickle.load(f)
    with open(features_path, "r") as f:
        _features = json.load(f)

    logger.info("Model artifacts loaded.")


def _band(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    else:
        return "High"


def predict_single(input_dict: dict) -> dict:
    """
    Score a single applicant.

    Parameters
    ----------
    input_dict : raw feature values keyed by column name

    Returns
    -------
    dict with keys: risk_score, risk_band, default_probability, shap_values
    """
    _load_artifacts()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.data.preprocessor import preprocess

    df = pd.DataFrame([input_dict])
    X, _, _ = preprocess(df, fit=False, label_encoders=_label_encoders)

    # Align columns to training features
    for col in _features:
        if col not in X.columns:
            X[col] = 0
    X = X[_features]

    prob = float(_model.predict_proba(X)[0, 1])
    risk_score = round(prob * 100, 1)
    band = _band(prob)

    # SHAP explanation
    shap_vals = _get_shap(X)

    return {
        "default_probability": round(prob, 4),
        "risk_score": risk_score,
        "risk_band": band,
        "shap_values": shap_vals,
    }


def _get_shap(X: pd.DataFrame) -> list[dict]:
    """Return top-10 SHAP feature contributions."""
    try:
        import shap
        explainer = shap.TreeExplainer(_model)
        sv = explainer.shap_values(X)
        # For binary classifiers, shap_values may return list [neg, pos]
        if isinstance(sv, list):
            sv = sv[1]
        arr = sv[0]
        contributions = [
            {"feature": feat, "shap_value": round(float(val), 5)}
            for feat, val in zip(X.columns, arr)
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions[:10]
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        # Fallback: feature importance
        try:
            importances = _model.feature_importances_
            contributions = [
                {"feature": feat, "shap_value": round(float(imp), 5)}
                for feat, imp in zip(X.columns, importances)
            ]
            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return contributions[:10]
        except Exception:
            return []
