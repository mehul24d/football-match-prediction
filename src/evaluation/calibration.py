"""
src/evaluation/calibration.py
--------------------------------
Calibration metrics + diagnostics (PRODUCTION SAFE).

✔ Multiclass Brier Score
✔ Expected Calibration Error (ECE)
✔ Reliability diagram data
✔ Full calibration report
✔ Robust against bad probabilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
# INTERNAL: SAFE PROBABILITY HANDLING
# ─────────────────────────────────────────────
def _sanitize_probs(y_prob: np.ndarray) -> np.ndarray:
    """Ensure probabilities are valid (0–1 and sum to 1)."""
    y_prob = np.asarray(y_prob, dtype=float)

    # Clip to [0,1]
    y_prob = np.clip(y_prob, 1e-12, 1.0)

    # Normalize rows
    row_sums = y_prob.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    return y_prob / row_sums


# ─────────────────────────────────────────────
# BRIER SCORE
# ─────────────────────────────────────────────
def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Multiclass Brier Score.

    Lower is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _sanitize_probs(y_prob)

    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true]

    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


# ─────────────────────────────────────────────
# ECE (FIXED BINNING)
# ─────────────────────────────────────────────
def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 0,
) -> float:
    """
    Expected Calibration Error (ECE) for ONE class.
    """
    y_prob = _sanitize_probs(y_prob)

    probs = y_prob[:, class_idx]
    labels = (y_true == class_idx).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    # FIX: ensure values in last bin
    bin_ids = np.digitize(probs, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    n = len(probs)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        conf = probs[mask].mean()
        acc = labels[mask].mean()

        ece += (mask.sum() / n) * abs(conf - acc)

    return float(ece)


def mean_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Average ECE across all classes.
    """
    y_prob = _sanitize_probs(y_prob)

    n_classes = y_prob.shape[1]

    return float(np.mean([
        expected_calibration_error(y_true, y_prob, n_bins, c)
        for c in range(n_classes)
    ]))


# ─────────────────────────────────────────────
# RELIABILITY DIAGRAM DATA
# ─────────────────────────────────────────────
def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 0,
) -> pd.DataFrame:
    """
    Data for reliability diagrams.
    """
    y_prob = _sanitize_probs(y_prob)

    probs = y_prob[:, class_idx]
    labels = (y_true == class_idx).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        if not np.any(mask):
            continue

        avg_conf = float(probs[mask].mean())
        frac_pos = float(labels[mask].mean())

        rows.append({
            "bin_centre": (lo + hi) / 2,
            "avg_confidence": avg_conf,
            "fraction_positive": frac_pos,
            "count": int(mask.sum()),
            "gap": avg_conf - frac_pos,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# FULL CALIBRATION REPORT
# ─────────────────────────────────────────────
def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
    n_bins: int = 10,
) -> dict:
    """
    Full calibration diagnostics.

    Returns:
      - Brier score
      - Mean ECE
      - Per-class ECE
    """
    y_true = np.asarray(y_true)
    y_prob = _sanitize_probs(y_prob)

    n_classes = y_prob.shape[1]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    report = {
        "brier_score": brier_score_multiclass(y_true, y_prob),
        "mean_ece": mean_ece(y_true, y_prob, n_bins),
    }

    for i in range(n_classes):
        report[f"ece_{class_names[i]}"] = expected_calibration_error(
            y_true, y_prob, n_bins, i
        )

    logger.info(
        f"📊 Calibration → "
        f"Brier: {report['brier_score']:.4f}, "
        f"Mean ECE: {report['mean_ece']:.4f}"
    )

    return report