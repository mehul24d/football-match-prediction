"""
src/models/evaluate.py
-----------------------
Robust evaluation utilities for match prediction models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Optional
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

from src.models.train import brier_score_multiclass


# ─── Core Metrics ────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Compute evaluation metrics safely."""

    if y_prob.ndim != 2:
        raise ValueError("y_prob must be 2D array (n_samples × n_classes)")

    if len(y_true) != len(y_pred):
        raise ValueError("Mismatch in y_true and y_pred length")

    class_names = class_names or ["Home Win", "Draw", "Away Win"]

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    bs = brier_score_multiclass(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    logger.info(
        f"Accuracy: {acc:.4f} | Log-loss: {ll:.4f} | Brier: {bs:.4f}"
    )

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": bs,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_idx: int = 0,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    -------
    prob_pred : predicted probabilities (x-axis)
    prob_true : actual fraction (y-axis)
    """
    if class_idx >= y_prob.shape[1]:
        raise ValueError("Invalid class index")

    y_binary = (y_true == class_idx).astype(int)

    prob_true, prob_pred = calibration_curve(
        y_binary,
        y_prob[:, class_idx],
        n_bins=n_bins,
        strategy="uniform",
    )

    return prob_pred, prob_true


# ─── SHAP Explainability ─────────────────────────────────────────────────────

def shap_feature_importance(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 15,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Compute SHAP importance safely (fast + robust)."""

    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return pd.DataFrame()

    # Sample data for performance
    if X.shape[0] > 2000:
        idx = np.random.choice(len(X), 2000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        logger.warning("TreeExplainer failed, falling back to KernelExplainer.")
        background = shap.sample(X_sample, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample[:200])

    # Handle multi-class output safely
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
        mean_abs = stacked.mean(axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean_abs": mean_abs,
    }).sort_values("importance_mean_abs", ascending=False).head(max_display)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["feature"], importance_df["importance_mean_abs"])
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_importance.png", dpi=100, bbox_inches="tight")
        plt.close()
        logger.success(f"✅ SHAP plot saved to {output_dir / 'shap_importance.png'}")

    return importance_df


# ─── Model Comparison ────────────────────────────────────────────────────────

def compare_models(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Create sorted comparison table."""

    rows = []
    for name, metrics in results.items():
        row = {"model": name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows)

    return df.sort_values("log_loss").reset_index(drop=True)