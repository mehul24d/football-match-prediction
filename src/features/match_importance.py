"""
src/features/match_importance.py
---------------------------------
Rolling Match Importance / Pressure Index (NO LEAKAGE)

Changes vs v1:
  - _get_table() now walks back to nearest available week (no silent 0.5 fallback)
  - _team_pressure() uses the live API "form" column when present to apply a
    momentum multiplier on top of the gap-based pressure score
  - add_pressure_features() replaced iterrows() with itertuples() for speed
  - Extra API columns (team_id, crest) are silently ignored — no code changes
    needed there since we only access named columns explicitly
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


DEFAULT_THRESHOLDS = {
    "title_gap":        5,
    "top4_gap":         3,
    "relegation_gap":   3,
    "top4_spots":       4,
    "relegation_spots": 3,
}

# How much the recent-form string (from live API) can scale pressure up/down.
# A team on 5 wins gets a +FORM_SCALE boost; 5 losses gets a -FORM_SCALE cut.
FORM_SCALE = 0.15


class MatchImportanceCalculator:
    def __init__(
        self,
        standings_by_week: dict[int, pd.DataFrame],
        season_weeks: int = 38,
        thresholds: Optional[dict] = None,
    ):
        self.cfg           = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.season_weeks  = season_weeks
        self.standings_by_week = standings_by_week

        # Pre-sort the available week keys once for fast nearest-week lookup
        self._sorted_weeks = sorted(standings_by_week.keys())

    # ─────────────────────────────────────────────────────────────────────
    # MAIN API
    # ─────────────────────────────────────────────────────────────────────

    def calculate(
        self,
        home_team: str,
        away_team: str,
        week: int,
    ) -> tuple[float, float]:
        table = self._get_table(week)

        if table is None:
            return 0.5, 0.5

        remaining    = max(self.season_weeks - week, 0)
        stage_weight = self._season_stage_weight(week)

        hp = self._team_pressure(home_team, table, remaining, stage_weight)
        ap = self._team_pressure(away_team, table, remaining, stage_weight)

        return float(hp), float(ap)

    # ─────────────────────────────────────────────────────────────────────
    # INTERNALS
    # ─────────────────────────────────────────────────────────────────────

    def _get_table(self, week: int) -> pd.DataFrame | None:
        """
        Return the standings snapshot for `week`.

        If the exact week is missing (e.g. gaps in the schedule or the live
        API keyed by currentMatchday), walk back to the most recent available
        week that is ≤ `week`.  Returns None only if no week at all is found.
        """
        if week in self.standings_by_week:
            return self.standings_by_week[week]

        # Find the largest available week that doesn't exceed `week`
        earlier = [w for w in self._sorted_weeks if w <= week]
        if earlier:
            fallback = earlier[-1]
            logger.debug(
                f"Week {week} not in standings — using week {fallback} as fallback."
            )
            return self.standings_by_week[fallback]

        logger.warning(f"No standings available for week ≤ {week}.")
        return None

    def _team_pressure(
        self,
        team: str,
        table: pd.DataFrame,
        remaining: int,
        stage_weight: float,
    ) -> float:
        if team not in table["team"].values:
            return 0.5

        row    = table[table["team"] == team].iloc[0]
        points = float(row["points"])
        max_pts = remaining * 3

        # ── Gap-based pressure components ─────────────────────────────────

        # Title race
        leader_pts = table["points"].max()
        title_gap  = leader_pts - points
        title_p    = self._gap_to_pressure(title_gap, self.cfg["title_gap"], max_pts)

        # Top-4 race
        sorted_pts  = table.sort_values("points", ascending=False)["points"].values
        cutoff_pts  = sorted_pts[self.cfg["top4_spots"] - 1]
        top4_gap    = abs(points - cutoff_pts)
        top4_p      = self._gap_to_pressure(top4_gap, self.cfg["top4_gap"], max_pts)

        # Relegation battle
        safety_idx  = len(table) - self.cfg["relegation_spots"] - 1
        safety_pts  = sorted_pts[safety_idx]
        rel_gap     = abs(points - safety_pts)
        rel_p       = self._gap_to_pressure(rel_gap, self.cfg["relegation_gap"], max_pts)

        raw = max(title_p, top4_p, rel_p)

        # ── Form multiplier (only when live API provides the "form" column) ──
        # The API returns a string like "W,D,W,W,W" (newest result first).
        # We convert it to a score in [-1, +1] and nudge pressure accordingly.
        if "form" in table.columns:
            form_str = row.get("form", "")
            form_multiplier = self._form_multiplier(form_str)
            raw = raw + form_multiplier * FORM_SCALE

        raw = raw * stage_weight

        return float(np.clip(raw, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _gap_to_pressure(gap: float, threshold: float, max_pts: float) -> float:
        if max_pts <= 0:
            return 1.0
        if gap > max_pts:
            return 0.0
        norm = gap / max(threshold, 1)
        return float(np.exp(-0.5 * norm ** 2))

    @staticmethod
    def _season_stage_weight(week: int) -> float:
        if week <= 5:
            return 0.4
        if week <= 10:
            return 0.5 + 0.05 * (week - 5)
        return min(1.0, 0.75 + 0.01 * (week - 10))

    @staticmethod
    def _form_multiplier(form_str: str) -> float:
        """
        Convert the API form string to a scalar in [-1, +1].

        "W,D,W,W,W" → score each result: W=+1, D=0, L=-1
        → mean of the scores → normalised to [-1, +1]

        If the form string is empty or unparseable, returns 0 (no effect).
        """
        if not form_str:
            return 0.0

        score_map = {"W": 1, "D": 0, "L": -1}
        results   = [r.strip() for r in form_str.split(",") if r.strip()]
        scores    = [score_map[r] for r in results if r in score_map]

        if not scores:
            return 0.0

        # Mean is already in [-1, +1] since each score ∈ {-1, 0, +1}
        return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# ✅ PUBLIC FEATURE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def add_pressure_features(
    df: pd.DataFrame,
    standings_by_week: dict[int, pd.DataFrame],
    season_weeks: int = 38,
) -> pd.DataFrame:
    """
    Append pressure_index_home and pressure_index_away columns to df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: home_team, away_team, week.
    standings_by_week : dict[int, pd.DataFrame]
        Output of build_rolling_standings().  The latest-week entry may be a
        live API DataFrame (with extra columns form, team_id, crest) — these
        are handled gracefully.
    season_weeks : int
        Total weeks in the season (default 38 for Premier League).
    """
    logger.info("🔥 Adding rolling pressure features (NO LEAKAGE)")

    calc = MatchImportanceCalculator(
        standings_by_week=standings_by_week,
        season_weeks=season_weeks,
    )

    home_p: list[float] = []
    away_p: list[float] = []

    # itertuples is ~10-50x faster than iterrows for large DataFrames
    for row in df.itertuples(index=False):
        week = int(getattr(row, "week", 1))

        hp, ap = calc.calculate(
            row.home_team,
            row.away_team,
            week,
        )

        home_p.append(hp)
        away_p.append(ap)

    df = df.copy()
    df["pressure_index_home"] = home_p
    df["pressure_index_away"] = away_p

    logger.success("✅ Pressure features added (rolling)")
    return df