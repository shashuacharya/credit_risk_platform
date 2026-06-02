"""
evaluate.py — Compute and return evaluation metrics for the trained model.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report, confusion_matrix,
)

logger = logging.getLogger(__name__)
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")


def load_metrics() -> dict:
    """Load pre-computed metrics from training."""
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def evaluate_on_data(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Run model on X, y and return full metric dict."""
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y, y_prob)
    prec, rec, _ = precision_recall_curve(y, y_prob)

    metrics = {
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "pr_auc": round(average_precision_score(y, y_prob), 4),
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
    }
    return metrics
