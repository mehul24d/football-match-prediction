"""
tests/test_features.py
-----------------------
Unit tests for feature engineering (Elo, form, rolling stats).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    EloRatingSystem,
    build_features,
    _compute_form,
    _rolling_team_stats,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_matches(n: int = 10) -> pd.DataFrame:
    """Return a minimal preprocessed-style DataFrame with *n* sequential matches."""
    np.random.seed(0)
    teams = ["Arsenal", "Chelsea", "Liverpool", "ManCity"]
    records = []
    for i in range(n):
        home_team = teams[i % len(teams)]
        away_team = teams[(i + 1) % len(teams)]
        hg = int(np.random.randint(0, 4))
        ag = int(np.random.randint(0, 4))
        if hg > ag:
            result = 0
            ftr = "H"
        elif hg < ag:
            result = 2
            ftr = "A"
        else:
            result = 1
            ftr = "D"
        records.append({
            "date": pd.Timestamp("2023-08-01") + pd.Timedelta(days=i * 7),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": hg,
            "away_goals": ag,
            "result": ftr,
            "result_label": result,
            "home_shots": np.random.randint(5, 20),
            "away_shots": np.random.randint(5, 20),
            "home_shots_on_target": np.random.randint(1, 8),
            "away_shots_on_target": np.random.randint(1, 8),
            "home_corners": np.random.randint(2, 10),
            "away_corners": np.random.randint(2, 10),
        })
    return pd.DataFrame(records)


# ─── EloRatingSystem ─────────────────────────────────────────────────────────

class TestEloRatingSystem:
    def test_initial_rating(self):
        elo = EloRatingSystem(initial_rating=1500)
        assert elo.get_rating("Arsenal") == 1500

    def test_home_win_increases_home_rating(self):
        elo = EloRatingSystem(k_factor=32, initial_rating=1500, home_advantage=0)
        r_home_before, r_away_before = elo.update("Arsenal", "Chelsea", result=0)
        assert elo.get_rating("Arsenal") > r_home_before
        assert elo.get_rating("Chelsea") < r_away_before

    def test_away_win_decreases_home_rating(self):
        elo = EloRatingSystem(k_factor=32, initial_rating=1500, home_advantage=0)
        r_home_before, _ = elo.update("Arsenal", "Chelsea", result=2)
        assert elo.get_rating("Arsenal") < r_home_before

    def test_draw_from_equal_ratings_preserves_ratings(self):
        """For equal teams a draw should barely change ratings."""
        elo = EloRatingSystem(k_factor=32, initial_rating=1500, home_advantage=0)
        elo.update("Arsenal", "Chelsea", result=1)
        # Should be very close to 1500
        assert abs(elo.get_rating("Arsenal") - 1500) < 1.0
        assert abs(elo.get_rating("Chelsea") - 1500) < 1.0

    def test_rating_sum_conserved(self):
        """Total Elo points should be approximately conserved."""
        elo = EloRatingSystem(k_factor=32, initial_rating=1500, home_advantage=0)
        for result in [0, 2, 1]:
            elo.update("Arsenal", "Chelsea", result=result)
        total = elo.get_rating("Arsenal") + elo.get_rating("Chelsea")
        assert abs(total - 3000) < 1e-9

    def test_expected_score_sums_to_one(self):
        elo = EloRatingSystem()
        e_a = elo.expected_score(1600, 1400)
        e_b = elo.expected_score(1400, 1600)
        assert abs(e_a + e_b - 1.0) < 1e-9

    def test_expected_score_range(self):
        elo = EloRatingSystem()
        for ra, rb in [(1000, 2000), (1500, 1500), (2000, 1000)]:
            e = elo.expected_score(ra, rb)
            assert 0 < e < 1


# ─── Rolling team stats ──────────────────────────────────────────────────────

class TestRollingTeamStats:
    def test_output_shape(self):
        df = _make_matches(10)
        result = _rolling_team_stats(
            df, "home_team", ["home_shots"], window=3, prefix="home"
        )
        assert len(result) == len(df)

    def test_first_row_is_nan(self):
        """First appearance of a team has no history → NaN."""
        df = _make_matches(6)
        result = _rolling_team_stats(
            df, "home_team", ["home_shots"], window=3, prefix="home"
        )
        team = df["home_team"].iloc[0]
        first_idx = df[df["home_team"] == team].index[0]
        assert np.isnan(result.loc[first_idx, "home_home_shots_avg"])

    def test_rolling_average_correct(self):
        """Manual check: with window=2 the average of the previous 2 matches."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=4, freq="W"),
            "home_team": ["Arsenal"] * 4,
            "away_team": ["Chelsea"] * 4,
            "home_shots": [10, 20, 30, 40],
            "result_label": [0, 0, 0, 0],
        })
        result = _rolling_team_stats(df, "home_team", ["home_shots"], window=2, prefix="h")
        # row 0: no history → NaN
        assert np.isnan(result["h_home_shots_avg"].iloc[0])
        # row 1: history = [10] → mean = 10
        assert result["h_home_shots_avg"].iloc[1] == pytest.approx(10.0)
        # row 2: history = [10, 20] → mean = 15
        assert result["h_home_shots_avg"].iloc[2] == pytest.approx(15.0)
        # row 3: history last 2 = [20, 30] → mean = 25
        assert result["h_home_shots_avg"].iloc[3] == pytest.approx(25.0)


# ─── Form calculation ────────────────────────────────────────────────────────

class TestComputeForm:
    def test_form_series_length(self):
        df = _make_matches(8)
        hf, af = _compute_form(df, window=3)
        assert len(hf) == len(df)
        assert len(af) == len(df)

    def test_first_appearance_is_nan(self):
        df = _make_matches(6)
        hf, _ = _compute_form(df, window=3)
        # The first match for each team has no prior history
        first_home = df["home_team"].iloc[0]
        first_idx = df[df["home_team"] == first_home].index[0]
        assert np.isnan(hf.iloc[first_idx])

    def test_form_values_in_range(self):
        df = _make_matches(20)
        hf, af = _compute_form(df, window=5)
        valid_home = hf.dropna()
        valid_away = af.dropna()
        assert (valid_home >= 0).all() and (valid_home <= 3).all()
        assert (valid_away >= 0).all() and (valid_away <= 3).all()


# ─── build_features ──────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = _make_matches(15)
        result = build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_elo_columns_present(self):
        df = _make_matches(10)
        result = build_features(df)
        for col in ["home_elo", "away_elo", "elo_diff"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_form_columns_present(self):
        df = _make_matches(10)
        result = build_features(df)
        for col in ["home_form", "away_form", "form_diff"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self):
        df = _make_matches(12)
        result = build_features(df)
        assert len(result) == len(df)

    def test_elo_initial_rating_used(self):
        """First match Elo should equal the initial rating."""
        df = _make_matches(5)
        result = build_features(df, elo_initial_rating=1700)
        # First row's Elo values come from the initial rating
        assert result["home_elo"].iloc[0] == pytest.approx(1700.0)
        assert result["away_elo"].iloc[0] == pytest.approx(1700.0)

    def test_elo_diff_is_correct(self):
        df = _make_matches(10)
        result = build_features(df)
        diff = result["home_elo"] - result["away_elo"]
        pd.testing.assert_series_equal(
            diff.round(6),
            result["elo_diff"].round(6),
            check_names=False,
        )
