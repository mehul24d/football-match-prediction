"""
src/models/train.py
-------------------
Train multiple classifiers and log experiments to MLflow.

Models
------
* Logistic Regression  (baseline)
* Random Forest
* XGBoost
* LightGBM

Metrics logged
--------------
* Accuracy
* Log-loss
* Brier score (multi-class macro)
* Cross-validation mean / std for each metric

Run as a module:
    python -m src.models.train
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils.helpers import ensure_dir, load_config, set_seed


# ─── Brier Score ─────────────────────────────────────────────────────────────

def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Multi-class Brier score (macro average across classes).

    BS = (1/N) * sum_i sum_k (p_ik - o_ik)^2
    where o_ik is 1 if sample i belongs to class k else 0.
    """
    n_classes = y_prob.shape[1]
    n_samples = len(y_true)
    total = 0.0
    for k in range(n_classes):
        o_k = (y_true == k).astype(float)
        total += np.sum((y_prob[:, k] - o_k) ** 2)
    return total / n_samples


# ─── Model definitions ───────────────────────────────────────────────────────

def get_models(cfg: dict[str, Any]) -> dict[str, Any]:
    """Instantiate all models from config."""
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
            min_samples_split=m_cfg["random_forest"]["min_samples_split"],
            min_samples_leaf=m_cfg["random_forest"]["min_samples_leaf"],
            n_jobs=m_cfg["random_forest"]["n_jobs"],
            random_state=cfg["project"]["random_seed"],
        ),
        "xgboost": XGBClassifier(
            n_estimators=m_cfg["xgboost"]["n_estimators"],
            learning_rate=m_cfg["xgboost"]["learning_rate"],
            max_depth=m_cfg["xgboost"]["max_depth"],
            subsample=m_cfg["xgboost"]["subsample"],
            colsample_bytree=m_cfg["xgboost"]["colsample_bytree"],
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=cfg["project"]["random_seed"],
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=m_cfg["lightgbm"]["n_estimators"],
            learning_rate=m_cfg["lightgbm"]["learning_rate"],
            max_depth=m_cfg["lightgbm"]["max_depth"],
            num_leaves=m_cfg["lightgbm"]["num_leaves"],
            subsample=m_cfg["lightgbm"]["subsample"],
            colsample_bytree=m_cfg["lightgbm"]["colsample_bytree"],
            verbosity=-1,
            random_state=cfg["project"]["random_seed"],
        ),
    }


# ─── Feature & target extraction ─────────────────────────────────────────────

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form", "away_form", "form_diff",
    "home_form_decayed", "away_form_decayed",
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "home_shots_avg", "away_shots_avg",
    "home_shots_on_target_avg", "away_shots_on_target_avg",
    "home_corners_avg", "away_corners_avg",
]


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "result_label",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Extract features and target, drop NaNs, scale, split.

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_scaler, used_feature_cols
    """
    feature_cols = feature_cols or FEATURE_COLS
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        raise ValueError("No feature columns found in DataFrame.")

    sub = df[available_cols + [target_col]].dropna()
    X = sub[available_cols].values
    y = sub[target_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, available_cols


# ─── Training & evaluation loop ──────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, float]:
    """Train model, evaluate on held-out test set and with cross-validation."""
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    bs = brier_score_multiclass(y_test, y_prob)

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_results = cross_validate(
        model.__class__(**model.get_params()),
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "neg_log_loss"],
        return_train_score=False,
    )

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": bs,
        "train_time_s": elapsed,
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_log_loss_mean": float(-np.mean(cv_results["test_neg_log_loss"])),
        "cv_log_loss_std": float(np.std(cv_results["test_neg_log_loss"])),
    }


# ─── Main entry point ─────────────────────────────────────────────────────────

def main(config_path: str | Path = "configs/config.yaml") -> None:
    """Train all models, compare, save the best one."""
    cfg = load_config(config_path)
    set_seed(cfg["project"]["random_seed"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    features_path = processed_dir / "matches_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}. "
            "Run the data pipeline first: python -m src.data.pipeline"
        )

    df = pd.read_csv(features_path, parse_dates=["date"])
    logger.info(f"Loaded feature dataset: {len(df)} rows")

    models_dir = ensure_dir(cfg["models"]["output_dir"])

    # MLflow setup
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_data(
        df,
        test_size=cfg["models"]["test_size"],
        random_seed=cfg["project"]["random_seed"],
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feat_cols)}")

    models = get_models(cfg)
    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        logger.info(f"Training {name}…")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(model.get_params())
            metrics = evaluate_model(
                model,
                X_train, X_test, y_train, y_test,
                cv_folds=cfg["models"]["cv_folds"],
                random_seed=cfg["project"]["random_seed"],
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=name)
            results[name] = metrics
            logger.info(
                f"  {name}: acc={metrics['accuracy']:.4f}, "
                f"log_loss={metrics['log_loss']:.4f}, "
                f"brier={metrics['brier_score']:.4f}"
            )

    # ── Save the best model (lowest log-loss on test set) ────────────────────
    best_name = min(results, key=lambda n: results[n]["log_loss"])
    best_model = models[best_name]

    # Refit on full training data
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, models_dir / "best_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")

    le = LabelEncoder().fit(["H", "D", "A"])
    joblib.dump(le, models_dir / "label_encoder.joblib")

    # Save feature column list
    with open(models_dir / "feature_columns.txt", "w") as fh:
        fh.write("\n".join(feat_cols))

    logger.success(
        f"Best model: {best_name} (log_loss={results[best_name]['log_loss']:.4f}). "
        f"Saved to {models_dir}."
    )

    # Print comparison table
    summary = pd.DataFrame(results).T[
        ["accuracy", "log_loss", "brier_score", "cv_accuracy_mean"]
    ]
    logger.info(f"\nModel Comparison:\n{summary.to_string()}")


if __name__ == "__main__":
    main()
