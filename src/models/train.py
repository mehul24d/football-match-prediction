"""
src/models/train.py
-------------------
Advanced training pipeline with:
- Time-aware validation
- Hyperparameter tuning
- Calibration (better probabilities)
- Feature importance logging
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from loguru import logger

from src.utils.helpers import ensure_dir, load_config, set_seed


# ─────────────────────────────────────────────
# Feature Columns
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form", "away_form", "form_diff",
    "home_form_decayed", "away_form_decayed",
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "home_shots_avg", "away_shots_avg",
    "home_shots_on_target_avg", "away_shots_on_target_avg",
    "home_corners_avg", "away_corners_avg",
    "home_rest_days", "away_rest_days",
    "elo_form_interaction",
]


# ─────────────────────────────────────────────
# Brier Score
# ─────────────────────────────────────────────

def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error over one-hot probabilities)."""
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


# ─────────────────────────────────────────────
# Models + Hyperparameter Search Space
# ─────────────────────────────────────────────
def get_models():
    return {
        "lightgbm": (
            LGBMClassifier(),
            {
                "n_estimators": [200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [-1, 5, 10],
                "num_leaves": [31, 50, 100],
            },
        ),
        "xgboost": (
            XGBClassifier(eval_metric="mlogloss"),
            {
                "n_estimators": [200, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 1.0],
            },
        ),
        "random_forest": (
            RandomForestClassifier(),
            {
                "n_estimators": [200, 300],
                "max_depth": [10, 20, None],
            },
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=1000),
            {
                "C": [0.1, 1, 10],
            },
        ),
    }


# ─────────────────────────────────────────────
# Data Preparation (TIME SAFE)
# ─────────────────────────────────────────────
def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int | None = None,
    feature_cols: list[str] | None = None,
):
    """
    Prepare train/test splits with scaling.

    Parameters
    ----------
    df           : DataFrame with features and a result_label/target column.
    test_size    : Fraction of data for the test set (temporal split).
    random_seed  : Optional random seed (used for reproducibility).
    feature_cols : Explicit list of feature columns to use.
                   If None, uses FEATURE_COLS filtered to those present in df.
                   Raises ValueError if any explicitly requested column is missing.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feat_cols
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Detect target column
    if "result_label" in df.columns:
        target_col = "result_label"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise ValueError("DataFrame must contain 'result_label' or 'target' column.")

    # Resolve feature columns
    if feature_cols is None:
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in DataFrame: {missing}")
        feat_cols = list(feature_cols)

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    df = df[feat_cols + [target_col]].dropna()

    split = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train_raw = train_df[feat_cols].values
    y_train = train_df[target_col].astype(np.int64).values

    X_test_raw = test_df[feat_cols].values
    y_test = test_df[target_col].astype(np.int64).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler, feat_cols


# ─────────────────────────────────────────────
# Train + Tune + Calibrate
# ─────────────────────────────────────────────
def train_model(name, model, param_grid, X_train, y_train):
    logger.info(f"Tuning {name}...")

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=5,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # 🔥 CALIBRATION → improves probability quality massively
    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    return calibrated, search.best_params_


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["project"]["random_seed"])

    df = pd.read_csv(
        Path(cfg["data"]["processed_dir"]) / "matches_features.csv",
        parse_dates=["date"],
    )

    models_dir = ensure_dir(cfg["models"]["output_dir"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_data(
        df,
        test_size=cfg["models"]["test_size"],
    )

    models = get_models()
    results = {}

    for name, (model, params) in models.items():
        with mlflow.start_run(run_name=name):
            trained_model, best_params = train_model(
                name, model, params, X_train, y_train
            )

            metrics = evaluate(trained_model, X_test, y_test)

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(trained_model, name)

            results[name] = {
                **metrics,
                "model": trained_model,
            }

            logger.info(f"{name} → {metrics}")

    # ─────────────────────────────────────────
    # Select Best Model (log loss)
    # ─────────────────────────────────────────
    best_name = min(results, key=lambda x: results[x]["log_loss"])
    best_model = results[best_name]["model"]

    logger.success(f"Best model: {best_name}")

    joblib.dump(best_model, models_dir / "best_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")

    # Save feature columns
    with open(models_dir / "feature_columns.txt", "w") as f:
        for col in feat_cols:
            f.write(col + "\n")

    logger.success("Model + scaler + features saved.")

    summary = pd.DataFrame(results).T.drop(columns=["model"])
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()