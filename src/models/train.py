"""
src/models/train.py
-------------------
Train multiple classifiers and log experiments to MLflow.
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
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.helpers import ensure_dir, load_config, set_seed


# ─── Brier Score ─────────────────────────────────────────────────────────────

def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_classes = y_prob.shape[1]
    n_samples = len(y_true)
    total = 0.0
    for k in range(n_classes):
        o_k = (y_true == k).astype(float)
        total += np.sum((y_prob[:, k] - o_k) ** 2)
    return total / n_samples


# ─── Model definitions ───────────────────────────────────────────────────────

def get_models(cfg: dict[str, Any]) -> dict[str, Any]:
    m_cfg = cfg["models"]
    return {
        "logistic_regression": LogisticRegression(
            C=m_cfg["logistic_regression"]["C"],
            max_iter=m_cfg["logistic_regression"]["max_iter"],
            solver=m_cfg["logistic_regression"]["solver"],
            random_state=cfg["project"]["random_seed"],
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=m_cfg["random_forest"]["n_estimators"],
            max_depth=m_cfg["random_forest"]["max_depth"],
            n_jobs=-1,
            random_state=cfg["project"]["random_seed"],
        ),
        "xgboost": XGBClassifier(
            n_estimators=m_cfg["xgboost"]["n_estimators"],
            learning_rate=m_cfg["xgboost"]["learning_rate"],
            max_depth=m_cfg["xgboost"]["max_depth"],
            subsample=m_cfg["xgboost"]["subsample"],
            colsample_bytree=m_cfg["xgboost"]["colsample_bytree"],
            eval_metric="mlogloss",
            random_state=cfg["project"]["random_seed"],
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=m_cfg["lightgbm"]["n_estimators"],
            learning_rate=m_cfg["lightgbm"]["learning_rate"],
            max_depth=m_cfg["lightgbm"]["max_depth"],
            num_leaves=m_cfg["lightgbm"]["num_leaves"],
            random_state=cfg["project"]["random_seed"],
        ),
    }


# ─── Feature columns ─────────────────────────────────────────────────────────

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


# ─── Data preparation (TIME-AWARE) ───────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float,
):
    df = df.sort_values("date").reset_index(drop=True)

    sub = df[feature_cols + ["target"]].dropna()

    split_idx = int(len(sub) * (1 - test_size))

    train_df = sub.iloc[:split_idx]
    test_df = sub.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values.astype(int)

    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ─── Training & evaluation ───────────────────────────────────────────────────

def evaluate_model(
    name: str,
    model: Any,
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    cv_folds,
):
    start = time.time()

    # Use scaled data only for Logistic Regression
    if name == "logistic_regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)
        X_cv = X_train_scaled
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        X_cv = X_train

    elapsed = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    bs = brier_score_multiclass(y_test, y_prob)

    # Time-aware CV
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    cv_results = cross_validate(
        model.__class__(**model.get_params()),
        X_cv,
        y_train,
        cv=tscv,
        scoring=["accuracy", "neg_log_loss"],
    )

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": bs,
        "train_time_s": elapsed,
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_log_loss_mean": float(-np.mean(cv_results["test_neg_log_loss"])),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main(config_path: str | Path = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["random_seed"])

    df = pd.read_csv(
        Path(cfg["data"]["processed_dir"]) / "matches_features.csv",
        parse_dates=["date"],
    )

    models_dir = ensure_dir(cfg["models"]["output_dir"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
        df,
        FEATURE_COLS,
        cfg["models"]["test_size"],
    )

    models = get_models(cfg)
    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")

        with mlflow.start_run(run_name=name):
            metrics = evaluate_model(
                name,
                model,
                X_train, X_test,
                X_train_scaled, X_test_scaled,
                y_train, y_test,
                cfg["models"]["cv_folds"],
            )

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=name)

            results[name] = metrics

    # ── Select best model ────────────────────────────────────────────────────
    best_name = min(results, key=lambda n: results[n]["log_loss"])
    best_model = models[best_name]

    # Retrain on FULL data
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    best_model.fit(X_full, y_full)

    joblib.dump(best_model, models_dir / "best_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")

    logger.success(f"Best model: {best_name} saved.")

    summary = pd.DataFrame(results).T
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()