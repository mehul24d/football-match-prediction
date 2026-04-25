"""
src/data/pipeline.py
--------------------
FULL v2 Pipeline:
download → preprocess → base → pressure (LIVE) → advanced → temporal → validation → save
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.data.ingestion import download_all, load_all_raw
from src.data.preprocessing import clean_matches, save_processed

from src.features.engineering import build_features
from src.features.match_importance import add_pressure_features
from src.features.advanced_features import add_advanced_features
from src.features.temporal_features import add_temporal_features

# ✅ NEW (LIVE STANDINGS API)
from src.data.live_standings import LiveStandings

from src.utils.helpers import load_config, ensure_dir, set_seed, setup_logging


# ─────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────

def _validate_features(df):
    required = [
        # Base
        "home_elo", "away_elo", "elo_diff",
        "home_form", "away_form", "form_diff",

        # Rolling stats
        "home_goals_scored_avg", "away_goals_scored_avg",
        "home_shots_avg", "away_shots_avg",

        # Interaction
        "elo_form_interaction",

        # Advanced
        "home_attack_vs_def",
        "away_attack_vs_def",
        "tempo_diff",
        "control_diff",

        # Temporal
        "home_form_lag_1",
        "away_form_lag_1",

        # Pressure
        "pressure_index_home",
        "pressure_index_away",
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
        logger.info("=== Step 1/5: Downloading raw data ===")
        try:
            download_all(config_path=config_path, overwrite=overwrite_download)
        except Exception as exc:
            logger.warning(f"Download failed → using cached data: {exc}")
    else:
        logger.info("=== Step 1/5: Skipped download ===")

    # ─────────────────────────────────────────
    # Step 2: Preprocessing
    # ─────────────────────────────────────────
    logger.info("=== Step 2/5: Loading & preprocessing ===")

    raw_df = load_all_raw(raw_dir)

    if raw_df is None or raw_df.empty:
        logger.error("❌ No raw data found. Pipeline aborted.")
        return

    clean_df = clean_matches(raw_df)
    save_processed(clean_df, clean_path)

    logger.info(f"Clean data shape: {clean_df.shape}")

    # ─────────────────────────────────────────
    # Step 3: Base Features
    # ─────────────────────────────────────────
    logger.info("=== Step 3/5: Base features ===")

    features_cfg = cfg["features"]

    feature_df = build_features(
        df=clean_df,
        form_window=features_cfg["form_window"],
        elo_k_factor=features_cfg["elo_k_factor"],
        elo_initial_rating=features_cfg["elo_initial_rating"],
        elo_home_advantage=features_cfg["elo_home_advantage"],
    )

    # ─────────────────────────────────────────
    # Step 4: Advanced Layers
    # ─────────────────────────────────────────
    logger.info("=== Step 4/5: Advanced features ===")

    # 🔥 LIVE PRESSURE INDEX
    try:
        logger.info("Fetching LIVE standings...")

        standings_api = LiveStandings()
        standings_df = standings_api.get_standings()

        if "week" in feature_df.columns:
            current_week = int(feature_df["week"].max())
        else:
            current_week = 38  # fallback

        feature_df = add_pressure_features(
            feature_df,
            standings_by_week={current_week: standings_df},
            season_weeks=38,
        )

        logger.success("✅ Pressure index added using LIVE standings")

    except Exception as exc:
        logger.warning(f"⚠️ Live standings failed → fallback pressure: {exc}")
        feature_df["pressure_index_home"] = 0.5
        feature_df["pressure_index_away"] = 0.5

    # 🔥 Advanced features
    feature_df = add_advanced_features(feature_df)

    # 🔥 Temporal features
    feature_df = add_temporal_features(feature_df)

    # ─────────────────────────────────────────
    # Step 5: Validation + Save
    # ─────────────────────────────────────────
    logger.info("=== Step 5/5: Validation & saving ===")

    _validate_features(feature_df)

    feature_df = feature_df.fillna(0)

    save_processed(feature_df, feature_path)

    logger.success(
        f"""
🎯 PIPELINE COMPLETE (v2 - LIVE)

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