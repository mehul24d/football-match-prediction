"""
src/models/train.py
-------------------
FINAL Training Pipeline (v6 - COMPLETE INTEGRATION):

✔ Runs data preprocessing & standings building
✔ Extracts league position features
✔ Uses Elo + form + rolling stats + league + advanced + temporal features
✔ SAFE numeric-only feature selection
✔ Stacking Ensemble
✔ Time-aware split
✔ Calibration metrics (Brier + ECE)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
from src.data.pipeline import run_pipeline
from src.features.engineering import build_features


# ─── Create Target Column ────────────────────────────────────────────────────
def create_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target column from match results.
    
    0 = Home Win
    1 = Draw
    2 = Away Win
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must have columns: home_goals, away_goals
        
    Returns:
    --------
    df : pd.DataFrame
        With result_label column added
    """
    if "result_label" in df.columns:
        logger.info("✅ Target column already exists")
        return df
    
    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        raise ValueError("❌ Missing home_goals or away_goals columns")
    
    df = df.copy()
    
    def determine_result(row):
        home = int(row["home_goals"])
        away = int(row["away_goals"])
        
        if home > away:
            return 0  # Home win
        elif home == away:
            return 1  # Draw
        else:
            return 2  # Away win
    
    df["result_label"] = df.apply(determine_result, axis=1)
    
    logger.info(f"✅ Created target column (result_label)")
    logger.info(f"   Home wins: {(df['result_label'] == 0).sum()}")
    logger.info(f"   Draws: {(df['result_label'] == 1).sum()}")
    logger.info(f"   Away wins: {(df['result_label'] == 2).sum()}")
    
    return df


# ─── Feature Selection (SAFE) ────────────────────────────────────────────────
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Get numeric feature columns, excluding IDs and target variables.
    
    This is SAFE - only uses numeric columns that are meaningful for prediction.
    """
    ignore = [
        "date", "home_team", "away_team", "match_id",
        "result", "result_label", "target",
        "prob_home", "prob_draw", "prob_away",
        "PSH", "PSD", "PSA",
        "home_goals", "away_goals", "goal_diff",
        "country", "season",
    ]

    # ✅ Only numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ignore]

    if not feature_cols:
        raise ValueError("❌ No numeric features found!")

    logger.info(f"Using {len(feature_cols)} features:")
    for col in sorted(feature_cols):
        logger.info(f"  - {col}")

    return feature_cols


# ─── Prepare Data (TIME-AWARE SPLIT) ─────────────────────────────────────────
def prepare_data(df: pd.DataFrame, test_size=0.2):
    """
    Prepare time-series split with scaling.
    
    Uses temporal ordering (not random) to avoid data leakage.
    """
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values("date").reset_index(drop=True)

    # Detect target column
    target_col = "result_label" if "result_label" in df.columns else "target"

    # Get features
    feature_cols = get_feature_columns(df)

    # Select relevant columns and drop NaN
    df_clean = df[feature_cols + [target_col]].dropna()

    if df_clean.empty:
        raise ValueError("❌ DataFrame is empty after feature selection!")

    logger.info(f"Using {len(df_clean)} matches for training")

    # Time-aware split (not random!)
    split = int(len(df_clean) * (1 - test_size))

    train_df = df_clean.iloc[:split]
    test_df = df_clean.iloc[split:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].astype(int).values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].astype(int).values

    # Scaling (fit on train, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape : {X_test.shape}")

    return X_train, y_train, X_test, y_test, feature_cols, scaler


# ─── Build Base Models ───────────────────────────────────────────────────────
def get_base_models() -> list:
    """
    Build list of base models for stacking ensemble.
    
    Combines multiple diverse learners for better generalization.
    """
    models = [
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        ),
    ]

    if HAS_XGB:
        models.append(XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ))

    if HAS_LGB:
        models.append(LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ))

    logger.info(f"Using {len(models)} base models:")
    for i, model in enumerate(models, 1):
        logger.info(f"  {i}. {model.__class__.__name__}")

    return models


# ─── Brier Score (Multiclass) ────────────────────────────────────────────────
def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score for multiclass classification.
    
    Measures the mean squared difference between predicted probabilities
    and actual outcomes (lower is better).
    """
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be 2D array (n_samples × n_classes)")
    
    y_true_one_hot = np.eye(y_prob.shape[1])[y_true]
    return np.mean((y_prob - y_true_one_hot) ** 2)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Returns accuracy, log loss, and calibration metrics.
    """
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    report = calibration_report(y_test, probs)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "log_loss": log_loss(y_test, probs),
        "brier_score": brier_score_multiclass(y_test, probs),
        **report
    }


# ─── Load and Combine Standings ──────────────────────────────────────────────
def load_standings(standings_dir: Path) -> pd.DataFrame:
    """
    Load and combine all standings files.
    
    Parameters:
    -----------
    standings_dir : Path
        Directory containing standings_*.csv files
        
    Returns:
    --------
    combined_standings : pd.DataFrame
        Combined standings data
    """
    standings_files = list(standings_dir.glob("standings_*.csv"))
    
    if not standings_files:
        logger.warning(f"⚠️  No standings files found in {standings_dir}")
        return None
    
    logger.info(f"Found {len(standings_files)} standings files")
    
    all_standings = []
    for f in standings_files:
        df = pd.read_csv(f)
        all_standings.append(df)
    
    combined_standings = pd.concat(all_standings, ignore_index=True)
    
    logger.info(f"Loaded {len(combined_standings)} standing records")
    
    return combined_standings


# ─── Load and Combine Matches ────────────────────────────────────────────────
def load_matches(processed_dir: Path) -> pd.DataFrame:
    """
    Load and combine all processed match files.
    
    Parameters:
    -----------
    processed_dir : Path
        Directory containing matches_*.csv files
        
    Returns:
    --------
    combined_matches : pd.DataFrame
        Combined match data with country and season
    """
    matches_files = list(processed_dir.glob("matches_*.csv"))
    
    if not matches_files:
        logger.warning(f"⚠️  No match files found in {processed_dir}")
        return None
    
    logger.info(f"Found {len(matches_files)} match files")
    
    all_matches = []
    for f in matches_files:
        stem = f.stem
        
        # Skip non-match files
        if not stem.startswith("matches_"):
            logger.debug(f"⏭️  Skipping {f.name} - not a match file")
            continue
        
        # Check if it's a processed match file (matches_E0_2022_23.csv format)
        parts = stem.split("_")
        
        # Must have format: matches_COUNTRY_YEAR_YEAR
        if len(parts) < 4:
            logger.debug(f"⏭️  Skipping {f.name} - invalid format (expected: matches_COUNTRY_YEAR_YEAR)")
            continue
        
        df = pd.read_csv(f)
        
        # Extract country and season from filename (matches_E0_2022_23.csv)
        country = "_".join(parts[1:-2])  # E0, D1, SP1, F1
        season = "_".join(parts[-2:])    # 2022_23
        
        df['country'] = country
        df['season'] = season
        all_matches.append(df)
        
        logger.info(f"  ✅ Loaded {f.name}: {len(df)} matches ({country} {season})")
    
    if not all_matches:
        logger.error(f"❌ No valid match files processed")
        return None
    
    combined_matches = pd.concat(all_matches, ignore_index=True)
    
    logger.info(f"Loaded {len(combined_matches)} total matches")
    
    return combined_matches


# ─── Main Training Pipeline ──────────────────────────────────────────────────
def main(config_path="configs/config.yaml"):
    """
    Execute complete pipeline:
    1. Data preprocessing & standings building
    2. Load matches and standings
    3. Create target column
    4. Engineer features (Elo + form + rolling + league + advanced + temporal)
    5. Train stacking ensemble
    6. Evaluate with calibration metrics
    """
    
    # Load config (with safe fallbacks)
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        logger.warning(f"⚠️  Config not found at {config_path} - using defaults")
        cfg = {
            "project": {"random_seed": 42},
            "data": {"processed_dir": "data/processed"},
            "models": {"model_dir": "models"}
        }
    
    set_seed(cfg.get("project", {}).get("random_seed", 42))

    # ─── STEP 1: Run data preprocessing pipeline ───────────────────────────────
    logger.info("="*100)
    logger.info("STEP 1: DATA PREPROCESSING & STANDINGS BUILDING")
    logger.info("="*100 + "\n")
    
    pipeline_result = run_pipeline(config_path)
    all_standings = pipeline_result['all_standings']
    standings_dir = pipeline_result['standings_dir']
    processed_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    
    logger.info(f"✅ Data pipeline complete!")
    logger.info(f"   Generated standings for {len(all_standings)} countries\n")
    
    # ─── STEP 2: Load matches and standings ────────────────────────────────────
    logger.info("="*100)
    logger.info("STEP 2: LOADING MATCHES AND STANDINGS")
    logger.info("="*100 + "\n")
    
    matches_df = load_matches(processed_dir)
    if matches_df is None:
        raise ValueError("❌ Failed to load matches")
    
    standings_df = load_standings(standings_dir)
    if standings_df is None:
        logger.warning("⚠️  Standings not available - will proceed without league features")
    
    # ─── STEP 2.5: Create target column ─────────────────────────────────────────
    logger.info("="*100)
    logger.info("STEP 2.5: CREATING TARGET COLUMN")
    logger.info("="*100 + "\n")
    
    matches_df = create_target_column(matches_df)
    
    # ─── STEP 2.75: Convert date to datetime ────────────────────────────────────
    logger.info("="*100)
    logger.info("STEP 2.75: CONVERTING DATE COLUMN")
    logger.info("="*100 + "\n")
    
    matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
    
    # Check for any invalid dates
    invalid_dates = matches_df['date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"⚠️  Found {invalid_dates} invalid dates - removing rows")
        matches_df = matches_df.dropna(subset=['date'])
    
    logger.success(f"✅ Date conversion complete")
    logger.info(f"   Date range: {matches_df['date'].min()} to {matches_df['date'].max()}\n")
    
    # ─── STEP 3: Feature Engineering ───────────────────────────────────────────
    logger.info("="*100)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("="*100 + "\n")
    
    logger.info("Building comprehensive feature set...\n")
    
    features_df = build_features(
        df=matches_df,
        standings_df=standings_df,
        form_window=5,
        elo_k_factor=32,
        elo_initial_rating=1500,
        elo_home_advantage=100,
    )
    
    if features_df is None or features_df.empty:
        raise ValueError("❌ Feature engineering failed!")
    
    logger.info(f"✅ Feature engineering complete!")
    logger.info(f"   Shape: {features_df.shape}")
    logger.info(f"   Columns: {features_df.shape[1]}\n")
    
    # Save engineered features
    feature_path = processed_dir / "features_complete.csv"
    features_df.to_csv(feature_path, index=False)
    logger.success(f"✅ Saved features to {feature_path.name}\n")
    
    # ─── STEP 4: Prepare data for training ──────────────────────────────────────
    logger.info("="*100)
    logger.info("STEP 4: DATA PREPARATION")
    logger.info("="*100 + "\n")
    
    # Ensure date column is datetime
    features_df['date'] = pd.to_datetime(features_df['date'], errors='coerce')
    
    X_train, y_train, X_test, y_test, feature_cols, scaler = prepare_data(
        features_df,
        test_size=0.2
    )
    
    logger.info(f"✅ Data preparation complete!")
    logger.info(f"   Features used: {len(feature_cols)}\n")
    
    # ─── STEP 5: Build and train models ────────────────────────────────────────
    logger.info("="*100)
    logger.info("STEP 5: MODEL TRAINING")
    logger.info("="*100 + "\n")
    
    logger.info("Building base models...")
    base_models = get_base_models()

    logger.info("Training stacking ensemble...")
    ensemble = StackingEnsemble(base_models)
    ensemble.fit(X_train, y_train)
    logger.success(f"✅ Model training complete!\n")
    
    # ─── STEP 6: Evaluation ────────────────────────────────────────────────────
    logger.info("="*100)
    logger.info("STEP 6: EVALUATION")
    logger.info("="*100 + "\n")
    
    logger.info("Evaluating on test set...")
    results = evaluate(ensemble, X_test, y_test)
    
    logger.success(f"\n📊 TEST SET METRICS:")
    logger.success(f"   Accuracy:    {results['accuracy']:.4f}")
    logger.success(f"   Log Loss:    {results['log_loss']:.4f}")
    logger.success(f"   Brier Score: {results['brier_score']:.4f}")
    
    if 'ece' in results:
        logger.success(f"   ECE:         {results['ece']:.4f}")
    if 'mce' in results:
        logger.success(f"   MCE:         {results['mce']:.4f}\n")
    
    # ─── STEP 7: Save models and artifacts ──────────────────────────────────────
    logger.info("="*100)
    logger.info("STEP 7: SAVING ARTIFACTS")
    logger.info("="*100 + "\n")
    
    # Get models directory with safe fallback
    models_dir = Path(cfg.get("models", {}).get("model_dir", "models"))
    ensure_dir(models_dir)
    
    logger.info(f"Saving to: {models_dir}\n")
    
    # Save ensemble
    ensemble_path = models_dir / "ensemble_model.pkl"
    joblib.dump(ensemble, ensemble_path)
    logger.success(f"✅ Saved ensemble to {ensemble_path.name}")
    
    # Save scaler
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.success(f"✅ Saved scaler to {scaler_path.name}")
    
    # Save feature columns
    features_path = models_dir / "feature_columns.pkl"
    joblib.dump(feature_cols, features_path)
    logger.success(f"✅ Saved feature columns to {features_path.name}\n")
    
    # ─── STEP 8: SUMMARY ──────────────────────────────────────────────────────
    logger.info("="*100)
    logger.info("COMPLETE PIPELINE EXECUTION SUMMARY")
    logger.info("="*100 + "\n")
    
    logger.info(f"📊 Data:")
    logger.info(f"   Countries processed: {len(all_standings)}")
    logger.info(f"   Total matches: {len(features_df)}")
    logger.info(f"   Training samples: {X_train.shape[0]}")
    logger.info(f"   Test samples: {X_test.shape[0]}\n")
    
    logger.info(f"🔧 Features:")
    logger.info(f"   Total features: {len(feature_cols)}")
    logger.info(f"   Categories:")
    logger.info(f"     - Elo ratings (3)")
    logger.info(f"     - Form features (6)")
    logger.info(f"     - Rolling statistics (10)")
    logger.info(f"     - League position (20)")
    logger.info(f"     - Advanced (≥8)")
    logger.info(f"     - Temporal (≥5)\n")
    
    logger.info(f"🤖 Model:")
    logger.info(f"   Base models: {len(base_models)}")
    logger.info(f"   Type: Stacking Ensemble\n")
    
    logger.info(f"📈 Performance:")
    logger.info(f"   Accuracy: {results['accuracy']:.4f}")
    logger.info(f"   Log Loss: {results['log_loss']:.4f}")
    logger.info(f"   Brier Score: {results['brier_score']:.4f}\n")
    
    logger.info(f"💾 Artifacts saved to: {models_dir}\n")
    
    return {
        'model': ensemble,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'results': results,
        'X_test': X_test,
        'y_test': y_test,
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = main()