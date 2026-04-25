"""Evaluation sub-package."""

from src.evaluation.calibration import (
    brier_score_multiclass,
    expected_calibration_error,
    mean_ece,
    reliability_diagram_data,
    calibration_report,
)
from src.evaluation.rolling_backtest import RollingBacktest
from src.evaluation.explainability import generate_shap_explanation

__all__ = [
    "brier_score_multiclass",
    "expected_calibration_error",
    "mean_ece",
    "reliability_diagram_data",
    "calibration_report",
    "RollingBacktest",
    "generate_shap_explanation",
]
