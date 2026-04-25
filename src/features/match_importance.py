"""
src/features/match_importance.py
---------------------------------
LIVE Match importance / pressure index engine.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

# 🔥 NEW: Live API
try:
    from src.data.live_standings import LiveStandings
except ImportError:
    LiveStandings = None


DEFAULT_THRESHOLDS = {
    "title_gap": 5,
    "top4_gap": 3,
    "relegation_gap": 3,
    "n_teams": 20,
    "top4_spots": 4,
    "relegation_spots": 3,
}


class MatchImportanceCalculator:
    def __init__(
        self,
        standings: Optional[pd.DataFrame] = None,
        season_weeks: int = 38,
        thresholds: Optional[dict] = None,
        use_live_api: bool = True,
    ):
        self.cfg = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.season_weeks = season_weeks

        # 🔥 LIVE STANDINGS FETCH
        if standings is None and use_live_api and LiveStandings:
            try:
                standings = LiveStandings().get_standings()
                logger.info("✅ Live standings fetched")
            except Exception as e:
                logger.warning(f"Live standings failed: {e}")
                standings = None

        if standings is None:
            raise ValueError("No standings available (API + input both failed)")

        self.standings = self._prepare_standings(standings)

    # ─────────────────────────────────────────────
    # PREP
    # ─────────────────────────────────────────────
    def _prepare_standings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "position" not in df.columns:
            df = df.sort_values("points", ascending=False)
            df["position"] = range(1, len(df) + 1)

        if "team" in df.columns:
            df = df.set_index("team")

        return df

    # ─────────────────────────────────────────────
    # MAIN API
    # ─────────────────────────────────────────────
    def calculate(self, home_team: str, away_team: str, current_week: int):
        remaining = max(self.season_weeks - current_week, 0)
        stage_weight = self._season_stage_weight(current_week)

        hp = self._team_pressure(home_team, remaining, stage_weight)
        ap = self._team_pressure(away_team, remaining, stage_weight)

        return float(hp), float(ap)

    # ─────────────────────────────────────────────
    # CORE LOGIC
    # ─────────────────────────────────────────────
    def _team_pressure(self, team, remaining, stage_weight):
        if team not in self.standings.index:
            logger.warning(f"{team} not in standings → default 0.5")
            return 0.5

        row = self.standings.loc[team]
        position = int(row["position"])
        points = float(row["points"])

        max_pts = remaining * 3

        # TITLE
        leader_pts = self.standings["points"].max()
        title_gap = leader_pts - points
        title_p = self._gap_to_pressure(title_gap, self.cfg["title_gap"], max_pts)

        # TOP 4
        cutoff_pts = self.standings.sort_values("points", ascending=False)["points"].iloc[self.cfg["top4_spots"] - 1]
        top4_gap = abs(points - cutoff_pts)
        top4_p = self._gap_to_pressure(top4_gap, self.cfg["top4_gap"], max_pts)

        # RELEGATION
        safety_pos = len(self.standings) - self.cfg["relegation_spots"]
        safety_pts = self.standings.sort_values("points", ascending=False)["points"].iloc[safety_pos - 1]
        rel_gap = abs(points - safety_pts)
        rel_p = self._gap_to_pressure(rel_gap, self.cfg["relegation_gap"], max_pts)

        raw = max(title_p, top4_p, rel_p)

        return float(np.clip(raw * stage_weight, 0, 1))

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    @staticmethod
    def _gap_to_pressure(gap, threshold, max_pts):
        if max_pts <= 0:
            return 1.0
        if gap > max_pts:
            return 0.0
        norm = gap / max(threshold, 1)
        return float(np.exp(-0.5 * norm ** 2))

    @staticmethod
    def _season_stage_weight(week):
        if week <= 5:
            return 0.4
        if week <= 10:
            return 0.5 + 0.05 * (week - 5)
        return min(1.0, 0.75 + 0.01 * (week - 10))


# ─────────────────────────────────────────────
# 🔥 UPDATED: NO MORE standings_by_week
# ─────────────────────────────────────────────
def add_pressure_features(
    df: pd.DataFrame,
    season_weeks: int = 38,
) -> pd.DataFrame:
    """
    Adds pressure features using LIVE standings.
    """
    calc = MatchImportanceCalculator(season_weeks=season_weeks)

    home_p, away_p = [], []

    for _, row in df.iterrows():
        week = int(row.get("week", 1))

        hp, ap = calc.calculate(
            row["home_team"],
            row["away_team"],
            week
        )

        home_p.append(hp)
        away_p.append(ap)

    df["pressure_index_home"] = home_p
    df["pressure_index_away"] = away_p

    return df