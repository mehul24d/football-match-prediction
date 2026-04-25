"""
src/evaluation/calibration.py
------------------------------
Calibration metrics and reliability diagrams.

Provides:
  - Multiclass Brier Score
  - Expected Calibration Error (ECE)
  - Reliability diagram data for plotting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ─── Brier Score ─────────────────────────────────────────────────────────────

def brier_score_multiclass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Multiclass Brier score: mean of squared probability errors.

    Perfect predictions → 0.0.  Random (1/3 each) → ≈ 0.667.

    Parameters
    ----------
    y_true : (n_samples,) integer class labels.
    y_prob : (n_samples, n_classes) probability matrix.

    Returns
    -------
    Scalar Brier score.
    """
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


# ─── Expected Calibration Error ──────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 0,
) -> float:
    """
    Expected Calibration Error (ECE) for a single class.

    Measures the average gap between predicted probability and observed
    frequency across equal-width confidence bins.

    Parameters
    ----------
    y_true    : (n_samples,) integer class labels.
    y_prob    : (n_samples, n_classes) probability matrix.
    n_bins    : Number of equal-width bins.
    class_idx : Which class to evaluate (0=Home Win, 1=Draw, 2=Away Win).

    Returns
    -------
    ECE as a float in [0, 1].
    """
    probs = y_prob[:, class_idx]
    labels = (y_true == class_idx).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (bin_size / n) * abs(avg_conf - avg_acc)

    return float(ece)


def mean_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Mean ECE across all classes.

    Returns the average Expected Calibration Error over all output classes.
    """
    n_classes = y_prob.shape[1]
    return float(
        np.mean([
            expected_calibration_error(y_true, y_prob, n_bins=n_bins, class_idx=c)
            for c in range(n_classes)
        ])
    )


# ─── Reliability diagram data ─────────────────────────────────────────────────

def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 0,
) -> pd.DataFrame:
    """
    Compute data for a reliability (calibration) diagram.

    Parameters
    ----------
    y_true    : Integer class labels.
    y_prob    : Probability matrix.
    n_bins    : Number of bins.
    class_idx : Class to evaluate.

    Returns
    -------
    DataFrame with columns: ['bin_centre', 'avg_confidence', 'fraction_positive',
                              'count', 'gap']
    """
    probs = y_prob[:, class_idx]
    labels = (y_true == class_idx).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)

        bin_size = int(mask.sum())
        if bin_size == 0:
            continue

        avg_conf = float(probs[mask].mean())
        frac_pos = float(labels[mask].mean())
        rows.append({
            "bin_centre": (lo + hi) / 2,
            "avg_confidence": avg_conf,
            "fraction_positive": frac_pos,
            "count": bin_size,
            "gap": avg_conf - frac_pos,
        })

    return pd.DataFrame(rows)


# ─── Summary calibration report ──────────────────────────────────────────────

def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
    n_bins: int = 10,
) -> dict:
    """
    Compute a full calibration report: Brier score, ECE per class, mean ECE.

    Parameters
    ----------
    y_true       : Integer class labels.
    y_prob       : Probability matrix.
    class_names  : Optional names for each class.
    n_bins       : Calibration bins.

    Returns
    -------
    Dict with 'brier_score', 'mean_ece', and per-class 'ece_{class}' values.
    """
    class_names = class_names or [f"class_{i}" for i in range(y_prob.shape[1])]
    n_classes = y_prob.shape[1]

    report: dict = {
        "brier_score": brier_score_multiclass(y_true, y_prob),
        "mean_ece": mean_ece(y_true, y_prob, n_bins=n_bins),
    }
    for c in range(n_classes):
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins, class_idx=c)
        report[f"ece_{class_names[c]}"] = ece

    logger.info(
        f"Calibration – Brier: {report['brier_score']:.4f}  "
        f"Mean ECE: {report['mean_ece']:.4f}"
    )
    return report
