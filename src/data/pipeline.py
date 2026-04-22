"""
src/data/pipeline.py
--------------------
End-to-end data pipeline:
  download → preprocess → feature engineering → validation → save

Run:
    python -m src.data.pipeline
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.data.ingestion import download_all, load_all_raw
from src.data.preprocessing import clean_matches, save_processed
from src.features.engineering import build_features
from src.utils.helpers import load_config, ensure_dir, set_seed, setup_logging


# ─────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────

def _validate_features(df):
    required = [
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

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"❌ Missing engineered features: {missing}")

    logger.success("✅ Feature validation passed")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    config_path: str | Path = "configs/config.yaml",
    download: bool = True,
    overwrite_download: bool = False,
) -> None:
    cfg = load_config(config_path)

    setup_logging(
        log_level=cfg.get("logging", {}).get("level", "INFO"),
        log_dir=cfg.get("logging", {}).get("log_dir"),
    )

    set_seed(cfg["project"]["random_seed"])

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])

    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    clean_path = processed_dir / "matches_clean.csv"
    feature_path = processed_dir / "matches_features.csv"

    # ─────────────────────────────────────────
    # Step 1: Download
    # ─────────────────────────────────────────
    if download:
        logger.info("=== Step 1/3: Downloading raw data ===")
        try:
            download_all(config_path=config_path, overwrite=overwrite_download)
        except Exception as exc:
            logger.warning(f"Download failed → using cached data: {exc}")
    else:
        logger.info("=== Step 1/3: Skipped download ===")

    # ─────────────────────────────────────────
    # Step 2: Preprocessing
    # ─────────────────────────────────────────
    logger.info("=== Step 2/3: Loading & preprocessing ===")

    raw_df = load_all_raw(raw_dir)

    if raw_df is None or raw_df.empty:
        logger.error("❌ No raw data found. Pipeline aborted.")
        return

    clean_df = clean_matches(raw_df)

    save_processed(clean_df, clean_path)

    logger.info(f"Clean data shape: {clean_df.shape}")

    # ─────────────────────────────────────────
    # Step 3: Feature Engineering
    # ─────────────────────────────────────────
    logger.info("=== Step 3/3: Feature engineering ===")

    features_cfg = cfg["features"]

    feature_df = build_features(
        df=clean_df,
        form_window=features_cfg["form_window"],
        elo_k_factor=features_cfg["elo_k_factor"],
        elo_initial_rating=features_cfg["elo_initial_rating"],
        elo_home_advantage=features_cfg["elo_home_advantage"],
    )

    # ─────────────────────────────────────────
    # Validation (CRITICAL)
    # ─────────────────────────────────────────
    _validate_features(feature_df)

    save_processed(feature_df, feature_path)

    logger.info(f"Feature data shape: {feature_df.shape}")

    # ─────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────
    logger.success(
        f"""
🎯 PIPELINE COMPLETE

Rows        : {len(feature_df)}
Columns     : {len(feature_df.columns)}

Files:
✔ Clean     → {clean_path}
✔ Features  → {feature_path}
"""
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()