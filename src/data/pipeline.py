"""
src/data/pipeline.py
--------------------
End-to-end data pipeline:
  download → preprocess → feature engineering → save features

Run as a module:
    python -m src.data.pipeline
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.data.ingestion import download_all, load_all_raw
from src.data.preprocessing import clean_matches, save_processed
from src.features.engineering import build_features
from src.utils.helpers import load_config, ensure_dir, set_seed, setup_logging


def run_pipeline(
    config_path: str | Path = "configs/config.yaml",
    download: bool = True,
    overwrite_download: bool = False,
) -> None:
    """
    Execute the full data pipeline.

    Parameters
    ----------
    config_path         : path to the YAML config
    download            : if True, attempt to download raw data first
    overwrite_download  : if True, re-download even when files exist
    """
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

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if download:
        logger.info("=== Step 1/3: Downloading raw data ===")
        try:
            download_all(config_path=config_path, overwrite=overwrite_download)
        except Exception as exc:
            logger.warning(f"Download step failed (continuing with existing files): {exc}")
    else:
        logger.info("=== Step 1/3: Skipping download (download=False) ===")

    # ── Step 2: Load & preprocess ─────────────────────────────────────────────
    logger.info("=== Step 2/3: Loading and preprocessing raw data ===")
    raw_df = load_all_raw(raw_dir)
    if raw_df is None or raw_df.empty:
        logger.warning("No raw data found. Pipeline aborted.")
        return

    clean_df = clean_matches(raw_df)
    save_processed(clean_df, processed_dir / "matches_clean.csv")

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    logger.info("=== Step 3/3: Feature engineering ===")
    features_cfg = cfg["features"]
    feature_df = build_features(
        df=clean_df,
        form_window=features_cfg["form_window"],
        elo_k_factor=features_cfg["elo_k_factor"],
        elo_initial_rating=features_cfg["elo_initial_rating"],
        elo_home_advantage=features_cfg["elo_home_advantage"],
    )
    save_processed(feature_df, processed_dir / "matches_features.csv")

    logger.success(
        f"Pipeline complete. Feature dataset: {len(feature_df)} rows, "
        f"{len(feature_df.columns)} columns."
    )


if __name__ == "__main__":
    run_pipeline()
