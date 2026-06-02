"""
train.py — Model training pipeline for credit default prediction.

Model: LightGBM (fast, handles missing values, great on tabular data).
Class imbalance: scale_pos_weight + SMOTE optional flag.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def get_model(n_pos: int, n_neg: int):
    """Build LightGBM classifier with class-imbalance handling."""
    try:
        import lightgbm as lgb # type: ignore
        scale = n_neg / max(n_pos, 1)
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        logger.info(f"LightGBM model — scale_pos_weight={scale:.2f}")
        return model, "lightgbm"
    except ImportError:
        logger.warning("LightGBM not available, falling back to RandomForest.")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        return model, "random_forest"


def train(X: pd.DataFrame, y: np.ndarray, label_encoders: dict):
    """Train model, evaluate, and save artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    logger.info(f"Class distribution — Positive (default): {n_pos}, Negative: {n_neg}")

    model, model_type = get_model(n_pos, n_neg)

    # 5-fold CV for robust estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(f"CV ROC-AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    # Final fit on full data
    model.fit(X, y)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model_type": model_type,
        "train_roc_auc": round(roc_auc_score(y, y_prob), 4),
        "train_pr_auc": round(average_precision_score(y, y_prob), 4),
        "cv_roc_auc_mean": round(float(cv_aucs.mean()), 4),
        "cv_roc_auc_std": round(float(cv_aucs.std()), 4),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "features": list(X.columns),
    }

    report = classification_report(y, y_pred, output_dict=True)
    metrics["classification_report"] = report

    logger.info(f"Train ROC-AUC: {metrics['train_roc_auc']}")
    logger.info(f"Train PR-AUC:  {metrics['train_pr_auc']}")

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    enc_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    features_path = os.path.join(MODEL_DIR, "features.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(enc_path, "wb") as f:
        pickle.dump(label_encoders, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(features_path, "w") as f:
        json.dump(list(X.columns), f, indent=2)

    logger.info(f"Model saved to {model_path}")
    return model, metrics


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.data.loader import load_and_join_all
    from src.data.preprocessor import preprocess

    df = load_and_join_all()
    X, y, label_encoders = preprocess(df, fit=True)
    train(X, y, label_encoders)
