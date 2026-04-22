"""
src/data/preprocessing.py
--------------------------
Clean and standardise raw match DataFrames before feature engineering.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger


# ─── Column renaming map ─────────────────────────────────────────────────────

COLUMN_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}

NUMERIC_COLS = [
    "home_goals", "away_goals",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",
]

ODDS_COLS = ["odds_home", "odds_draw", "odds_away"]


# ─── Public API ──────────────────────────────────────────────────────────────

def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for raw football match data.
    """
    df = df.copy()
    initial_rows = len(df)

    # 1. Rename columns
    df = _rename_columns(df)

    # 2. Validate required columns
    required_cols = ["home_team", "away_team", "date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3. Normalize team names
    df = _normalize_team_names(df)

    # 4. Convert date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] <= pd.Timestamp.now()].copy()

    # 5. Convert numeric columns
    for col in NUMERIC_COLS + ODDS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Handle missing values
    before_na = len(df)
    df = df.dropna(subset=["home_team", "away_team", "result"])
    logger.info(f"Dropped {before_na - len(df)} rows due to missing teams/results")

    # Fill numeric stats
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Drop rows where odds missing
    df = df.dropna(subset=[c for c in ODDS_COLS if c in df.columns])

    # 7. Remove invalid scores
    for col in ["home_goals", "away_goals"]:
        if col in df.columns:
            df = df[df[col] >= 0]

    # 8. Encode target variable
    df = _encode_result(df)

    # 🔥 IMPORTANT FIX: ensure compatibility with training.py
    df["result_label"] = df["target"]

    # 9. Add safe features
    df["season"] = df["date"].dt.year

    df["match_id"] = (
        df["date"].astype(str)
        + "_"
        + df["home_team"]
        + "_"
        + df["away_team"]
    )

    # 10. Convert odds → probabilities
    df = _convert_odds_to_probabilities(df)

    # 11. Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before_dup - len(df)} duplicate rows")

    # 12. Sort
    df = df.sort_values("date").reset_index(drop=True)

    logger.success(
        f"Preprocessing complete – {len(df)} rows retained (from {initial_rows})"
    )

    return df


# ─── Helper functions ─────────────────────────────────────────────────────────

def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    return df.rename(columns=rename)


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df["home_team"] = df["home_team"].str.strip().str.lower()
    df["away_team"] = df["away_team"].str.strip().str.lower()
    return df


def _encode_result(df: pd.DataFrame) -> pd.DataFrame:
    label_map = {"H": 0, "D": 1, "A": 2}
    df = df.copy()

    df["target"] = df["result"].map(label_map)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    return df


def _convert_odds_to_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    if not all(col in df.columns for col in ODDS_COLS):
        return df

    odds_home = df["odds_home"]
    odds_draw = df["odds_draw"]
    odds_away = df["odds_away"]

    inv_sum = (1 / odds_home) + (1 / odds_draw) + (1 / odds_away)

    df["prob_home"] = (1 / odds_home) / inv_sum
    df["prob_draw"] = (1 / odds_draw) / inv_sum
    df["prob_away"] = (1 / odds_away) / inv_sum

    return df


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Processed data saved to {output_path} ({len(df)} rows)")


def load_processed(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Processed file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["date"])
    logger.info(f"Loaded processed data from {filepath} ({len(df)} rows)")
    return df