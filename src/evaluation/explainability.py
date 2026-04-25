"""
src/evaluation/explainability.py
----------------------------------
SHAP-based feature importance and per-match explanation utilities.

Supports TreeExplainer (tree-based models) and KernelExplainer (any model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger


def generate_shap_explanation(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 2000,
    output_dir: Optional[str | Path] = None,
    match_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance for a fitted model.

    Tries TreeExplainer first (fast for tree-based models), then falls back
    to KernelExplainer for other model types.

    Parameters
    ----------
    model        : Fitted sklearn-compatible estimator.
    X            : Feature matrix (n_samples, n_features).
    feature_names: Names corresponding to X columns.
    max_samples  : Subsample limit for performance.
    output_dir   : If provided, saves a bar-chart PNG there.
    match_id     : Optional label for the plot title.

    Returns
    -------
    DataFrame with columns ['feature', 'importance_mean_abs'] sorted descending.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return pd.DataFrame(columns=["feature", "importance_mean_abs"])

    # Subsample for speed
    if X.shape[0] > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    shap_values = None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        logger.debug("SHAP: used TreeExplainer.")
    except Exception:
        logger.debug("TreeExplainer failed; using KernelExplainer (slower).")
        try:
            background = shap.sample(X_sample, min(100, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            n_kernel = min(200, len(X_sample))
            shap_values = explainer.shap_values(X_sample[:n_kernel])
        except Exception as exc:
            logger.error(f"SHAP explainability failed: {exc}")
            return pd.DataFrame(columns=["feature", "importance_mean_abs"])

    # Aggregate across classes
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
        mean_abs = stacked.mean(axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance_mean_abs": mean_abs})
        .sort_values("importance_mean_abs", ascending=False)
        .reset_index(drop=True)
    )

    if output_dir is not None:
        _save_shap_plot(importance_df, output_dir, match_id)

    return importance_df


def _save_shap_plot(
    importance_df: pd.DataFrame,
    output_dir: str | Path,
    match_id: Optional[str] = None,
    top_n: int = 15,
) -> None:
    """Save a horizontal bar chart of SHAP feature importance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed – skipping SHAP plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["importance_mean_abs"][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    title = "SHAP Feature Importance"
    if match_id:
        title += f" – {match_id}"
    ax.set_title(title)
    plt.tight_layout()

    fname = f"shap_{match_id or 'importance'}.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    logger.info(f"SHAP plot saved to {output_dir / fname}")
