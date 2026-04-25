"""
src/features/match_importance.py
---------------------------------
Match importance / pressure index calculation engine.

Computes a Pressure Index (0–1) for each team in a match based on:
  - Distance to key league thresholds (title, top-4, relegation)
  - Remaining matches in the season
  - Points impact of win/draw/loss scenarios
  - Season-stage weighting (early season = lower weight)
  - Recent form trajectory modifier
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─── Default threshold configuration ─────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "title_gap": 5,          # Within N points of 1st place → title contention
    "top4_gap": 3,           # Within N points of top-4 cutoff
    "relegation_gap": 3,     # Within N points above relegation zone
    "n_teams": 20,           # Teams in the league
    "top4_spots": 4,         # Number of European/promotion spots
    "relegation_spots": 3,   # Number of relegation spots
}


class MatchImportanceCalculator:
    """
    Calculate a pressure index for each team in an upcoming match.

    Parameters
    ----------
    standings : pd.DataFrame
        Current league standings with columns:
        ['team', 'points', 'position', 'matches_played']
    season_weeks : int
        Total number of matchdays in the season (e.g. 38 for 20-team league).
    thresholds : dict, optional
        Override for default threshold configuration.
    """

    def __init__(
        self,
        standings: pd.DataFrame,
        season_weeks: int = 38,
        thresholds: Optional[dict] = None,
    ):
        self.standings = standings.copy()
        self.season_weeks = season_weeks
        self.cfg = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        # Sort by position for easy threshold lookups
        if "position" not in self.standings.columns:
            self.standings = self.standings.sort_values("points", ascending=False)
            self.standings["position"] = range(1, len(self.standings) + 1)

        self.standings = self.standings.set_index("team") if "team" in self.standings.columns else self.standings

    # ── Public API ────────────────────────────────────────────────────────────

    def calculate(
        self,
        home_team: str,
        away_team: str,
        current_week: int,
    ) -> tuple[float, float]:
        """
        Compute pressure indices for both teams.

        Parameters
        ----------
        home_team    : str   Name of the home team.
        away_team    : str   Name of the away team.
        current_week : int   Current matchday (1-indexed).

        Returns
        -------
        (pressure_home, pressure_away) : tuple[float, float]
            Each value is in [0, 1].  Higher = more important match.
        """
        remaining = max(self.season_weeks - current_week, 0)
        stage_weight = self._season_stage_weight(current_week)

        pressure_home = self._team_pressure(home_team, remaining, stage_weight)
        pressure_away = self._team_pressure(away_team, remaining, stage_weight)

        logger.debug(
            f"Pressure – {home_team}: {pressure_home:.3f}  "
            f"{away_team}: {pressure_away:.3f}"
        )
        return float(pressure_home), float(pressure_away)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _team_pressure(
        self, team: str, remaining_matches: int, stage_weight: float
    ) -> float:
        """Compute raw pressure score for a single team."""
        if team not in self.standings.index:
            logger.warning(f"Team '{team}' not found in standings; returning 0.5")
            return 0.5

        row = self.standings.loc[team]
        position = int(row["position"])
        points = float(row["points"])

        # Points available in remaining matches
        max_future_pts = remaining_matches * 3

        # ── Title contention ──────────────────────────────────────────────
        leader_pts = float(self.standings["points"].max())
        title_gap = leader_pts - points
        title_pressure = self._gap_to_pressure(
            title_gap, self.cfg["title_gap"], max_future_pts
        ) if position <= self.cfg["top4_spots"] + 2 else 0.0

        # ── Top-4 / European qualification ───────────────────────────────
        cutoff_pos = self.cfg["top4_spots"]
        if len(self.standings) >= cutoff_pos:
            top4_cutoff_pts = float(
                self.standings.sort_values("points", ascending=False)["points"].iloc[cutoff_pos - 1]
            )
        else:
            top4_cutoff_pts = points
        top4_gap = abs(points - top4_cutoff_pts)
        top4_pressure = self._gap_to_pressure(
            top4_gap, self.cfg["top4_gap"], max_future_pts
        ) if position <= cutoff_pos + 3 else 0.0

        # ── Relegation battle ─────────────────────────────────────────────
        n_teams = len(self.standings)
        safety_pos = n_teams - self.cfg["relegation_spots"]
        if len(self.standings) >= safety_pos:
            safety_pts = float(
                self.standings.sort_values("points", ascending=False)["points"].iloc[safety_pos - 1]
            )
        else:
            safety_pts = points
        relegation_gap = points - safety_pts  # negative if in danger
        relegation_pressure = self._gap_to_pressure(
            abs(relegation_gap), self.cfg["relegation_gap"], max_future_pts
        ) if position >= safety_pos - 2 else 0.0

        # ── Combine pressures ─────────────────────────────────────────────
        raw_pressure = max(title_pressure, top4_pressure, relegation_pressure)

        # Apply season-stage multiplier
        pressure = raw_pressure * stage_weight

        return float(np.clip(pressure, 0.0, 1.0))

    @staticmethod
    def _gap_to_pressure(gap: float, threshold: float, max_pts: float) -> float:
        """
        Convert a points gap to a 0-1 pressure score.

        Teams very close to a threshold (small gap) get high pressure.
        If the gap is larger than max achievable points, pressure is 0.
        """
        if max_pts <= 0:
            return 1.0  # Final matchday – everything is high pressure
        if gap > max_pts:
            return 0.0  # Can't catch up even with all wins
        # Sigmoid-like decay: small gap → high pressure
        normalised = gap / max(threshold, 1)
        return float(np.exp(-0.5 * normalised ** 2))

    @staticmethod
    def _season_stage_weight(current_week: int) -> float:
        """
        Weight based on how far into the season we are.

        Early season (week ≤ 10): low weight (0.4–0.6)
        Mid season: medium weight
        Late season (final 5 weeks): maximum weight (1.0)
        """
        if current_week <= 5:
            return 0.4
        if current_week <= 10:
            return 0.5 + 0.05 * (current_week - 5)
        # Linear ramp from 0.75 at week 10 to 1.0 at week 33+
        return min(1.0, 0.75 + 0.01 * (current_week - 10))


# ─── Convenience: add pressure features to a match DataFrame ─────────────────

def add_pressure_features(
    matches_df: pd.DataFrame,
    standings_by_week: dict[int, pd.DataFrame],
    season_weeks: int = 38,
) -> pd.DataFrame:
    """
    Add ``pressure_index_home`` and ``pressure_index_away`` columns to a
    DataFrame of matches.

    Parameters
    ----------
    matches_df        : DataFrame with columns ['home_team', 'away_team', 'week']
    standings_by_week : {week_number: standings_DataFrame} mapping
    season_weeks      : Total matchdays in the season

    Returns
    -------
    DataFrame with two new columns appended.
    """
    df = matches_df.copy()
    home_pressures, away_pressures = [], []

    for _, row in df.iterrows():
        week = int(row.get("week", 1))
        standings = standings_by_week.get(week)

        if standings is None:
            home_pressures.append(0.5)
            away_pressures.append(0.5)
            continue

        calc = MatchImportanceCalculator(standings, season_weeks=season_weeks)
        hp, ap = calc.calculate(row["home_team"], row["away_team"], week)
        home_pressures.append(hp)
        away_pressures.append(ap)

    df["pressure_index_home"] = home_pressures
    df["pressure_index_away"] = away_pressures
    return df
