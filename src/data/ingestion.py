"""
src/data/ingestion.py
---------------------
Download and cache raw match data from Football-Data.co.uk and
optionally from StatsBomb open-data.

Football-Data.co.uk CSV columns we rely on:
  Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR (full-time result H/D/A),
  HS, AS (shots), HST, AST (shots on target), HC, AC (corners),
  HF, AF (fouls), B365H, B365D, B365A (bookmaker odds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from src.utils.helpers import ensure_dir, load_config


# ─── Football-Data.co.uk ─────────────────────────────────────────────────────

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# Columns we want to keep from the raw CSV
REQUIRED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST",
    "HC", "AC", "HF", "AF",
]

OPTIONAL_COLUMNS = [
    "B365H", "B365D", "B365A",   # Bet365 odds
    "PSH", "PSD", "PSA",          # Pinnacle odds
    "xG", "xGA",                  # Expected goals (available in some leagues/seasons)
]


def _season_code(season: str) -> str:
    """Convert '2023-24' → '2324' for the URL."""
    parts = season.split("-")
    if len(parts) != 2:
        raise ValueError(f"Season must be in format 'YYYY-YY', got: {season}")
    return parts[0][2:] + parts[1]


def download_league_season(
    league_code: str,
    season: str,
    raw_dir: str | Path = "data/raw",
    overwrite: bool = False,
) -> Path:
    """
    Download one league-season CSV from Football-Data.co.uk.

    Parameters
    ----------
    league_code : str   e.g. 'E0' (Premier League)
    season      : str   e.g. '2023-24'
    raw_dir     : Path  destination directory
    overwrite   : bool  re-download even if file exists

    Returns
    -------
    Path to the saved CSV file.
    """
    ensure_dir(raw_dir)
    season_code = _season_code(season)
    url = FOOTBALL_DATA_BASE.format(season=season_code, code=league_code)
    filename = Path(raw_dir) / f"{league_code}_{season.replace('-', '_')}.csv"

    if filename.exists() and not overwrite:
        logger.info(f"File already exists, skipping download: {filename}")
        return filename

    logger.info(f"Downloading {league_code} {season} from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(filename, "wb") as fh:
        fh.write(response.content)
    logger.success(f"Saved to {filename}")
    return filename


def download_all(
    config_path: str | Path = "configs/config.yaml",
    overwrite: bool = False,
) -> list[Path]:
    """Download all league-seasons defined in the config file."""
    cfg = load_config(config_path)
    raw_dir = cfg["data"]["raw_dir"]
    leagues = cfg["data"]["leagues"]
    seasons = cfg["data"]["seasons"]

    downloaded: list[Path] = []
    for league in leagues:
        for season in seasons:
            try:
                path = download_league_season(
                    league_code=league["code"],
                    season=season,
                    raw_dir=raw_dir,
                    overwrite=overwrite,
                )
                downloaded.append(path)
            except Exception as exc:
                logger.warning(f"Failed to download {league['code']} {season}: {exc}")
    return downloaded


# ─── Loading raw CSVs ────────────────────────────────────────────────────────

def load_raw_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load a Football-Data.co.uk CSV and return a tidy DataFrame.

    Only the columns present in REQUIRED_COLUMNS are guaranteed;
    optional ones are included when available.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

    # Keep only columns that actually exist
    keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
    optional_keep = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep + optional_keep].copy()

    # Drop rows with no result (blank lines at end of file)
    df = df.dropna(subset=["FTR"])

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} rows from {filepath.name}")
    return df


def load_all_raw(raw_dir: str | Path = "data/raw") -> Optional[pd.DataFrame]:
    """Load and concatenate all raw CSV files found in *raw_dir*."""
    raw_dir = Path(raw_dir)
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {raw_dir}")
        return None

    frames: list[pd.DataFrame] = []
    for csv_file in sorted(csv_files):
        try:
            df = load_raw_csv(csv_file)
            # Infer league/season from filename  e.g. E0_2023_24.csv
            stem = csv_file.stem
            parts = stem.split("_", 1)
            df["league_code"] = parts[0] if len(parts) > 0 else "unknown"
            df["season"] = parts[1].replace("_", "-") if len(parts) > 1 else "unknown"
            frames.append(df)
        except Exception as exc:
            logger.warning(f"Skipping {csv_file.name}: {exc}")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined {len(frames)} files → {len(combined)} total rows")
    return combined


# ─── StatsBomb helper (open data) ────────────────────────────────────────────

STATSBOMB_COMPETITIONS_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
)


def list_statsbomb_competitions() -> pd.DataFrame:
    """Fetch the list of available StatsBomb open-data competitions."""
    logger.info("Fetching StatsBomb open-data competition list…")
    response = requests.get(STATSBOMB_COMPETITIONS_URL, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json())
