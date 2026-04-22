"""
src/models/evaluate.py
-----------------------
Evaluation utilities for match prediction models.

Metrics
-------
* Accuracy
* Log-loss (lower is better)
* Brier score (lower is better)
* Confusion matrix
* Calibration plot data
* SHAP feature importance
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

from src.models.train import brier_score_multiclass


# ─── Core metric bundle ──────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute a full set of evaluation metrics.

    Parameters
    ----------
    y_true       : integer ground-truth labels
    y_pred       : integer predicted labels
    y_prob       : probability matrix (n_samples × n_classes)
    class_names  : list of human-readable class names (e.g. ['Home', 'Draw', 'Away'])

    Returns
    -------
    Dictionary with accuracy, log_loss, brier_score, and classification_report.
    """
    class_names = class_names or ["Home Win", "Draw", "Away Win"]
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    bs = brier_score_multiclass(y_true, y_prob)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    logger.info(
        f"Accuracy: {acc:.4f} | Log-loss: {ll:.4f} | Brier: {bs:.4f}"
    )

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": bs,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_idx: int = 0,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mean_predicted_prob, fraction_of_positives) for a reliability plot.

    Parameters
    ----------
    class_idx : which class (0=Home win, 1=Draw, 2=Away win)
    """
    y_binary = (y_true == class_idx).astype(int)
    prob_true, prob_pred = calibration_curve(
        y_binary, y_prob[:, class_idx], n_bins=n_bins, strategy="uniform"
    )
    return prob_pred, prob_true


# ─── SHAP explainability ─────────────────────────────────────────────────────

def shap_feature_importance(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 15,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Compute SHAP values and return a feature importance DataFrame.

    For tree-based models (RF, XGBoost, LightGBM) uses TreeExplainer;
    falls back to KernelExplainer otherwise.

    Parameters
    ----------
    model         : fitted sklearn-compatible classifier
    X             : feature matrix (numpy array)
    feature_names : list of feature names matching X columns
    max_display   : top-N features to include in the output
    output_dir    : if provided, saves a SHAP summary plot here

    Returns
    -------
    DataFrame with columns [feature, importance_mean_abs]
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return pd.DataFrame()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        logger.warning("TreeExplainer failed, using KernelExplainer (may be slow).")
        background = shap.kmeans(X, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)

    # For multi-class, shap_values is a list; average absolute value across classes
    if isinstance(shap_values, list):
        mean_abs = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance_mean_abs": mean_abs}
    ).sort_values("importance_mean_abs", ascending=False).head(max_display)

    if output_dir:
        import matplotlib.pyplot as plt
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.barh(
            importance_df["feature"][::-1],
            importance_df["importance_mean_abs"][::-1],
        )
        plt.xlabel("Mean |SHAP value|")
        plt.title("Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_importance.png", dpi=150)
        plt.close()
        logger.info(f"SHAP plot saved to {output_dir / 'shap_importance.png'}")

    return importance_df


# ─── Model comparison table ──────────────────────────────────────────────────

def compare_models(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Build a comparison table from a dict of {model_name: metrics_dict}.

    Returns a DataFrame sorted by log_loss ascending.
    """
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "model": name,
                "accuracy": metrics.get("accuracy"),
                "log_loss": metrics.get("log_loss"),
                "brier_score": metrics.get("brier_score"),
                "cv_accuracy": metrics.get("cv_accuracy_mean"),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
