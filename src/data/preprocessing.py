"""
src/data/preprocessing.py
--------------------------
Clean and standardise raw match DataFrames before feature engineering.

Steps
-----
1. Rename columns to a consistent internal schema.
2. Cast numeric columns; fill small gaps via interpolation.
3. Encode the target label (FTR: H/D/A → 0/1/2).
4. Remove obvious data issues (future dates, impossible scores, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder


# ─── Column renaming map ─────────────────────────────────────────────────────

COLUMN_MAP = {
    "Date":     "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG":     "home_goals",
    "FTAG":     "away_goals",
    "FTR":      "result",
    "HS":       "home_shots",
    "AS":       "away_shots",
    "HST":      "home_shots_on_target",
    "AST":      "away_shots_on_target",
    "HC":       "home_corners",
    "AC":       "away_corners",
    "HF":       "home_fouls",
    "AF":       "away_fouls",
    "B365H":    "odds_home",
    "B365D":    "odds_draw",
    "B365A":    "odds_away",
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
    Full preprocessing pipeline for a raw matches DataFrame.

    Returns a clean DataFrame with standardised column names, numeric types
    and an integer ``result`` label (0 = Home win, 1 = Draw, 2 = Away win).
    """
    df = df.copy()

    # 1. Rename
    df = _rename_columns(df)

    # 2. Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["date"] <= pd.Timestamp.now()].copy()

    # 3. Cast numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Drop rows with missing essential columns
    essential = ["home_team", "away_team", "result", "home_goals", "away_goals"]
    present_essential = [c for c in essential if c in df.columns]
    df = df.dropna(subset=present_essential)

    # 5. Validate scores (non-negative integers)
    for col in ["home_goals", "away_goals"]:
        if col in df.columns:
            df = df[df[col] >= 0]

    # 6. Encode result: H → 0, D → 1, A → 2
    df = _encode_result(df)

    # 7. Derive goal difference and total goals
    if "home_goals" in df.columns and "away_goals" in df.columns:
        df["goal_diff"] = df["home_goals"] - df["away_goals"]
        df["total_goals"] = df["home_goals"] + df["away_goals"]

    # 8. Convert odds to implied probabilities
    for col, prob_col in [
        ("odds_home", "implied_prob_home"),
        ("odds_draw", "implied_prob_draw"),
        ("odds_away", "implied_prob_away"),
    ]:
        if col in df.columns:
            df[prob_col] = 1.0 / df[col].replace(0, np.nan)

    # 9. Sort by date
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"Preprocessing complete – {len(df)} rows retained")
    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard column renaming map (only for columns that exist)."""
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    return df.rename(columns=rename)


def _encode_result(df: pd.DataFrame) -> pd.DataFrame:
    """Map FTR string labels to integers and add a ``result_label`` column."""
    if "result" not in df.columns:
        return df
    label_map = {"H": 0, "D": 1, "A": 2}
    df = df.copy()
    df["result_label"] = df["result"].map(label_map)
    df = df.dropna(subset=["result_label"])
    df["result_label"] = df["result_label"].astype(int)
    return df


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the processed DataFrame to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Processed data saved to {output_path} ({len(df)} rows)")


def load_processed(filepath: str | Path) -> pd.DataFrame:
    """Load a previously saved processed CSV."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Processed file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["date"])
    logger.info(f"Loaded processed data from {filepath} ({len(df)} rows)")
    return df
