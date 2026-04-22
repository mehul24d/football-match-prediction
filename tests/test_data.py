"""
tests/test_data.py
------------------
Unit tests for data ingestion and preprocessing modules.
"""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import load_raw_csv, _season_code
from src.data.preprocessing import (
    COLUMN_MAP,
    clean_matches,
    _encode_result,
    _rename_columns,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

RAW_CSV_CONTENT = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HF,AF,B365H,B365D,B365A
12/08/2023,Arsenal,Chelsea,2,1,H,14,8,6,3,5,4,12,11,1.80,3.50,4.50
19/08/2023,Liverpool,ManCity,1,1,D,10,12,4,5,6,7,10,9,2.40,3.20,3.10
26/08/2023,Tottenham,Brighton,0,2,A,8,11,2,6,3,8,14,10,2.20,3.30,3.40
"""

INVALID_CSV_CONTENT = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
,Arsenal,Chelsea,,1,H
"""


def _make_raw_df() -> pd.DataFrame:
    return pd.read_csv(StringIO(RAW_CSV_CONTENT))


# ─── Season code ─────────────────────────────────────────────────────────────

def test_season_code_valid():
    assert _season_code("2023-24") == "2324"
    assert _season_code("2021-22") == "2122"


def test_season_code_invalid():
    with pytest.raises(ValueError):
        _season_code("2023")


# ─── Column renaming ─────────────────────────────────────────────────────────

def test_rename_columns():
    df = _make_raw_df()
    renamed = _rename_columns(df)
    assert "home_team" in renamed.columns
    assert "away_team" in renamed.columns
    assert "home_goals" in renamed.columns
    assert "result" in renamed.columns


def test_rename_columns_partial():
    """Renaming should work even if some raw columns are absent."""
    df = pd.DataFrame({"Date": ["12/08/2023"], "HomeTeam": ["Arsenal"]})
    renamed = _rename_columns(df)
    assert "date" in renamed.columns
    assert "home_team" in renamed.columns


# ─── Result encoding ─────────────────────────────────────────────────────────

def test_encode_result_values():
    df = pd.DataFrame({"result": ["H", "D", "A", "H"]})
    encoded = _encode_result(df)
    assert list(encoded["result_label"]) == [0, 1, 2, 0]


def test_encode_result_invalid_dropped():
    df = pd.DataFrame({"result": ["H", "X", "D"]})
    encoded = _encode_result(df)
    # 'X' has no mapping → NaN → dropped
    assert len(encoded) == 2


# ─── clean_matches ────────────────────────────────────────────────────────────

def test_clean_matches_returns_dataframe():
    df = _make_raw_df()
    clean = clean_matches(df)
    assert isinstance(clean, pd.DataFrame)


def test_clean_matches_column_names():
    df = _make_raw_df()
    clean = clean_matches(df)
    for col in ["home_team", "away_team", "home_goals", "away_goals", "result_label"]:
        assert col in clean.columns, f"Missing column: {col}"


def test_clean_matches_result_label_range():
    df = _make_raw_df()
    clean = clean_matches(df)
    assert clean["result_label"].isin([0, 1, 2]).all()


def test_clean_matches_goal_diff():
    df = _make_raw_df()
    clean = clean_matches(df)
    assert "goal_diff" in clean.columns
    # Arsenal 2-1 → goal_diff = 1
    arsenal_row = clean[clean["home_team"] == "Arsenal"].iloc[0]
    assert arsenal_row["goal_diff"] == 1


def test_clean_matches_drops_invalid_rows():
    """Rows with missing date or result should be dropped."""
    raw_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
12/08/2023,Arsenal,Chelsea,2,1,H
,Liverpool,ManCity,1,1,D
"""
    df = pd.read_csv(StringIO(raw_content))
    clean = clean_matches(df)
    assert len(clean) == 1


def test_clean_matches_negative_goals_removed():
    df = pd.DataFrame({
        "Date": ["12/08/2023"],
        "HomeTeam": ["Arsenal"],
        "AwayTeam": ["Chelsea"],
        "FTHG": [-1],
        "FTAG": [1],
        "FTR": ["H"],
    })
    clean = clean_matches(df)
    assert len(clean) == 0


def test_clean_matches_implied_probabilities():
    df = _make_raw_df()
    clean = clean_matches(df)
    if "implied_prob_home" in clean.columns:
        # All implied probabilities should be between 0 and 1
        assert (clean["implied_prob_home"].dropna() > 0).all()
        assert (clean["implied_prob_home"].dropna() < 1).all()


def test_clean_matches_sorted_by_date():
    raw_content = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
26/08/2023,Tottenham,Brighton,0,2,A
12/08/2023,Arsenal,Chelsea,2,1,H
"""
    df = pd.read_csv(StringIO(raw_content))
    clean = clean_matches(df)
    dates = clean["date"].tolist()
    assert dates == sorted(dates)
