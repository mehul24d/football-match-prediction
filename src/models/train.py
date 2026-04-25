"""
src/models/train.py
-------------------
FINAL Training Pipeline (v3 - FIXED):

✔ Uses advanced + temporal + pressure features
✔ SAFE numeric-only feature selection (no string bugs)
✔ Stacking Ensemble
✔ Time-aware split
✔ Calibration metrics (Brier + ECE)
✔ MLflow tracking
"""

from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Optional models
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.models.ensemble import StackingEnsemble
from src.evaluation.calibration import calibration_report
from src.utils.helpers import ensure_dir, load_config, set_seed


# ─────────────────────────────────────────────
# Feature Selection (SAFE)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Feature Selection (SAFE + CLEAN)
# ─────────────────────────────────────────────
def get_feature_columns(df: pd.DataFrame):
    ignore = [
        # IDs
        "date", "home_team", "away_team", "match_id",

        # Targets
        "result", "result_label", "target",

        # 🚫 LEAKAGE (VERY IMPORTANT)
        "prob_home", "prob_draw", "prob_away",
        "PSH", "PSD", "PSA",

        # Optional leakage (post-match info)
        "home_goals", "away_goals", "goal_diff"
    ]

    # ✅ Only numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    feature_cols = [c for c in numeric_cols if c not in ignore]

    if not feature_cols:
        raise ValueError("❌ No numeric features found!")

    logger.info(f"Using {len(feature_cols)} features")

    return feature_cols

    # ✅ Only numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    feature_cols = [c for c in numeric_cols if c not in ignore]

    if not feature_cols:
        raise ValueError("❌ No numeric features found!")

    logger.info(f"Using {len(feature_cols)} features")

    return feature_cols


# ─────────────────────────────────────────────
# Prepare Data (TIME SAFE)
# ─────────────────────────────────────────────
def prepare_data(df: pd.DataFrame, test_size=0.2):
    df = df.sort_values("date").reset_index(drop=True)

    target_col = "result_label" if "result_label" in df.columns else "target"

    feature_cols = get_feature_columns(df)

    df = df[feature_cols + [target_col]].dropna()

    if df.empty:
        raise ValueError("❌ No data left after dropping NA")

    split = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    # ✅ KEEP AS DATAFRAME (important)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape : {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ─────────────────────────────────────────────
# Build Base Models
# ─────────────────────────────────────────────
def get_base_models():
    models = [
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42
        ),
        LogisticRegression(max_iter=1000),
    ]

    if HAS_XGB:
        models.append(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            )
        )

    if HAS_LGB:
        models.append(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbosity=-1,
            )
        )

    logger.info(f"Using {len(models)} base models")

    return models


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    report = calibration_report(y_test, probs)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs),
        **report
    }


# ─────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────
def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["project"]["random_seed"])

    # Load features
    df = pd.read_csv(
        Path(cfg["data"]["processed_dir"]) / "matches_features.csv",
        parse_dates=["date"],
    )

    logger.info(f"Loaded dataset: {df.shape}")

    # Debug (optional)
    logger.debug(df.dtypes)

    models_dir = ensure_dir(cfg["models"]["output_dir"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ─────────────────────────────────────────
    # Prepare Data
    # ─────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(
        df,
        test_size=cfg["models"]["test_size"],
    )

    # ─────────────────────────────────────────
    # Ensemble
    # ─────────────────────────────────────────
    base_models = get_base_models()
    ensemble = StackingEnsemble(base_models)

    # ─────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────
    with mlflow.start_run(run_name="stacking_ensemble_v3"):
        logger.info("🚀 Training Stacking Ensemble...")

        ensemble.fit(X_train, y_train)

        metrics = evaluate(ensemble, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(ensemble, "ensemble_model")

        logger.success(f"📊 Results: {metrics}")

    # ─────────────────────────────────────────
    # Save Artifacts
    # ─────────────────────────────────────────
    joblib.dump(ensemble, models_dir / "ensemble_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")

    with open(models_dir / "feature_columns.txt", "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    logger.success("✅ Model, scaler, and features saved.")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()