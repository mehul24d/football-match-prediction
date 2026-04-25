"""
src/features/live_features.py
------------------------------
Extract comprehensive features from processed match data + rolling standings.
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

from src.utils.helpers import load_config


class LiveFeatureBuilder:
    """Build comprehensive features for match prediction."""

    def __init__(self, config_path: str | Path = "configs/config.yaml"):
        """Initialize feature builder with processed data."""
        self.config = load_config(config_path)
        self.processed_df = None
        self.standings_by_week = None
        self._load_data()

    def _load_data(self):
        """Load processed data and rolling standings."""
        try:
            from src.data.preprocessing import load_processed
            from src.features.rolling_standings import build_rolling_standings

            processed_path = (
                Path(self.config["data"]["processed_dir"]) / "matches_features.csv"
            )

            if processed_path.exists():
                self.processed_df = load_processed(processed_path)
                self.standings_by_week = build_rolling_standings(
                    self.processed_df,
                    patch_current_week=False,
                )
                logger.success(
                    f"✅ Loaded {len(self.processed_df)} matches, "
                    f"{len(self.standings_by_week)} weeks"
                )
            else:
                logger.warning(f"⚠️  Processed data not found: {processed_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")

    def build_match_features(
        self,
        home_team: str,
        away_team: str,
        week: int | None = None,
    ) -> dict[str, float]:
        """
        Build comprehensive features for a match.
        
        Extracts from:
        - Team historical stats
        - Form (last 5 matches)
        - League position + points
        - Head-to-head
        - Advanced metrics
        """
        try:
            if self.processed_df is None:
                logger.warning("No processed data available")
                return self._get_default_features()

            # Get team stats
            home_stats = self._extract_team_stats(home_team, week)
            away_stats = self._extract_team_stats(away_team, week)

            # Get league position
            home_pos, away_pos = self._get_standings_position(
                home_team, away_team, week
            )

            # Head-to-head
            h2h_home_win_rate = self._get_h2h(home_team, away_team)

            # Build feature dict
            features = {
                # ─── HOME TEAM BASIC ──────────────────────
                "home_elo": home_stats.get("elo", 1500),
                "home_form": home_stats.get("form", 0.5),
                "home_goals_scored_avg": home_stats.get("goals_scored_avg", 1.5),
                "home_goals_conceded_avg": home_stats.get("goals_conceded_avg", 1.5),
                "home_shots_avg": home_stats.get("shots_avg", 15),
                "home_shots_on_target_avg": home_stats.get("shots_on_target_avg", 5),
                "home_corners_avg": home_stats.get("corners_avg", 5),
                "home_fouls": home_stats.get("fouls_avg", 12),
                "home_attack_vs_def": home_stats.get("attack_vs_def", 0.0),

                # ─── AWAY TEAM BASIC ──────────────────────
                "away_elo": away_stats.get("elo", 1500),
                "away_form": away_stats.get("form", 0.5),
                "away_goals_scored_avg": away_stats.get("goals_scored_avg", 1.5),
                "away_goals_conceded_avg": away_stats.get("goals_conceded_avg", 1.5),
                "away_shots_avg": away_stats.get("shots_avg", 15),
                "away_shots_on_target_avg": away_stats.get("shots_on_target_avg", 5),
                "away_corners_avg": away_stats.get("corners_avg", 5),
                "away_fouls": away_stats.get("fouls_avg", 12),
                "away_attack_vs_def": away_stats.get("attack_vs_def", 0.0),

                # ─── MATCH STATS (from processed data) ─────
                "home_shots": home_stats.get("shots_avg", 15),
                "away_shots": away_stats.get("shots_avg", 15),
                "home_shots_on_target": home_stats.get("shots_on_target_avg", 5),
                "away_shots_on_target": away_stats.get("shots_on_target_avg", 5),
                "home_corners": home_stats.get("corners_avg", 5),
                "away_corners": away_stats.get("corners_avg", 5),

                # ─── INTERACTION FEATURES ─────────────────
                "elo_diff": home_stats.get("elo", 1500) - away_stats.get("elo", 1500),
                "form_diff": home_stats.get("form", 0.5) - away_stats.get("form", 0.5),
                "elo_form_interaction": (
                    home_stats.get("elo", 1500) * home_stats.get("form", 0.5)
                    - away_stats.get("elo", 1500) * away_stats.get("form", 0.5)
                ),
                "tempo_diff": home_stats.get("tempo", 0) - away_stats.get("tempo", 0),
                "control_diff": home_stats.get("control", 0) - away_stats.get("control", 0),
                "attack_vs_defence": (
                    home_stats.get("attack_vs_def", 0.0) 
                    - away_stats.get("attack_vs_def", 0.0)
                ),

                # ─── FORM FEATURES ────────────────────────
                "home_form_lag_1": home_stats.get("form_lag_1", 0.5),
                "home_form_lag_2": home_stats.get("form_lag_2", 0.5),
                "home_form_lag_3": home_stats.get("form_lag_3", 0.5),
                "away_form_lag_1": away_stats.get("form_lag_1", 0.5),
                "away_form_lag_2": away_stats.get("form_lag_2", 0.5),
                "away_form_lag_3": away_stats.get("form_lag_3", 0.5),

                # ─── MOMENTUM & STREAKS ───────────────────
                "home_momentum": home_stats.get("momentum", 0.0),
                "away_momentum": away_stats.get("momentum", 0.0),
                "home_win_streak": home_stats.get("win_streak", 0),
                "away_win_streak": away_stats.get("win_streak", 0),

                # ─── VOLATILITY ───────────────────────────
                "home_volatility": home_stats.get("volatility", 0.5),
                "away_volatility": away_stats.get("volatility", 0.5),

                # ─── CONVERSION & EFFICIENCY ──────────────
                "home_conversion_rate": home_stats.get("conversion_rate", 0.1),
                "away_conversion_rate": away_stats.get("conversion_rate", 0.1),

                # ─── DECAYED FORM (recent > older) ────────
                "home_form_decayed": home_stats.get("form_decayed", 0.5),
                "away_form_decayed": away_stats.get("form_decayed", 0.5),
                "home_form_ewma": home_stats.get("form_ewma", 0.5),
                "away_form_ewma": away_stats.get("form_ewma", 0.5),

                # ─── ADJUSTED STATS ───────────────────────
                "home_attack_adj": home_stats.get("attack_adj", 0.0),
                "away_attack_adj": away_stats.get("attack_adj", 0.0),

                # ─── REST DAYS ────────────────────────────
                "home_rest_days": home_stats.get("rest_days", 3),
                "away_rest_days": away_stats.get("rest_days", 3),

                # ─── PRESSURE INDEX (from standings) ──────
                "pressure_index_home": home_pos["pressure"],
                "pressure_index_away": away_pos["pressure"],
                "pressure_form_home": home_stats.get("pressure_form", 0.5),
                "pressure_form_away": away_stats.get("pressure_form", 0.5),
                "home_pressure_form": home_stats.get("pressure_form", 0.5),
                "away_pressure_form": away_stats.get("pressure_form", 0.5),
                "elo_pressure_interaction": (
                    (home_stats.get("elo", 1500) * home_pos["pressure"])
                    - (away_stats.get("elo", 1500) * away_pos["pressure"])
                ),

                # ─── ODDS ─────────────────────────────────
                "odds_home": home_stats.get("odds_home", 2.5),
                "odds_draw": home_stats.get("odds_draw", 3.0),
                "odds_away": away_stats.get("odds_away", 2.5),

                # ─── H2H ──────────────────────────────────
                "h2h_home_win_rate": h2h_home_win_rate,
            }

            return features

        except Exception as e:
            logger.error(f"❌ Feature building failed: {e}")
            return self._get_default_features()

    def _extract_team_stats(
        self,
        team: str,
        week: int | None = None,
    ) -> dict:
        """Extract comprehensive stats for a team."""
        try:
            df = self.processed_df

            # Filter team matches (last 5 or before week)
            team_home = df[df["home_team"] == team].copy()
            team_away = df[df["away_team"] == team].copy()

            if week is not None:
                team_home = team_home[team_home["week"] <= week]
                team_away = team_away[team_away["week"] <= week]

            team_home = team_home.tail(5)
            team_away = team_away.tail(5)

            matches = pd.concat([team_home, team_away])

            if matches.empty:
                return {}

            stats = {}

            # ─── ELO ──────────────────────────────────
            if "home_elo" in team_home.columns and len(team_home) > 0:
                stats["elo"] = float(team_home["home_elo"].iloc[-1])
            elif "away_elo" in team_away.columns and len(team_away) > 0:
                stats["elo"] = float(team_away["away_elo"].iloc[-1])

            # ─── FORM (Points from last 5) ────────────
            points = []
            for _, row in matches.iterrows():
                if row["home_team"] == team:
                    if row["result"] == "H":
                        points.append(3)
                    elif row["result"] == "D":
                        points.append(1)
                    else:
                        points.append(0)
                else:
                    if row["result"] == "A":
                        points.append(3)
                    elif row["result"] == "D":
                        points.append(1)
                    else:
                        points.append(0)

            if points:
                stats["form"] = np.mean(points) / 3.0
                stats["form_ewma"] = np.average(
                    points, weights=np.exp(np.arange(len(points)))
                ) / 3.0
                stats["win_streak"] = self._get_win_streak(points)
                stats["momentum"] = (points[-1] - np.mean(points[:2])) / 3.0

            # ─── GOALS ────────────────────────────────
            goals_for = []
            goals_against = []
            for _, row in matches.iterrows():
                if row["home_team"] == team:
                    goals_for.append(row["home_goals"])
                    goals_against.append(row["away_goals"])
                else:
                    goals_for.append(row["away_goals"])
                    goals_against.append(row["home_goals"])

            if goals_for:
                stats["goals_scored_avg"] = np.mean(goals_for)
                stats["goals_conceded_avg"] = np.mean(goals_against)
                stats["conversion_rate"] = np.mean(goals_for) / max(
                    np.mean([row.get("home_shots", 15) for _ in matches]), 1
                )

            # ─── SHOTS ────────────────────────────────
            shots = []
            shots_on_target = []
            for _, row in matches.iterrows():
                if row["home_team"] == team:
                    shots.append(row.get("home_shots", 15))
                    shots_on_target.append(row.get("home_shots_on_target", 5))
                else:
                    shots.append(row.get("away_shots", 15))
                    shots_on_target.append(row.get("away_shots_on_target", 5))

            if shots:
                stats["shots_avg"] = np.mean(shots)
                stats["shots_on_target_avg"] = np.mean(shots_on_target)

            # ─── CORNERS & FOULS ──────────────────────
            corners = []
            fouls = []
            for _, row in matches.iterrows():
                if row["home_team"] == team:
                    corners.append(row.get("home_corners", 5))
                    fouls.append(row.get("home_fouls", 12))
                else:
                    corners.append(row.get("away_corners", 5))
                    fouls.append(row.get("away_fouls", 12))

            if corners:
                stats["corners_avg"] = np.mean(corners)
                stats["fouls_avg"] = np.mean(fouls)

            # ─── FORM LAGS ────────────────────────────
            if len(points) >= 1:
                stats["form_lag_1"] = points[-1] / 3.0
            if len(points) >= 2:
                stats["form_lag_2"] = points[-2] / 3.0
            if len(points) >= 3:
                stats["form_lag_3"] = points[-3] / 3.0

            # ─── VOLATILITY ────────────────────────────
            if len(points) > 1:
                stats["volatility"] = np.std(points) / 3.0
            else:
                stats["volatility"] = 0.5

            # ─── DECAYED FORM ──────────────────────────
            if points:
                recent_points = points[-3:]
                stats["form_decayed"] = np.mean(recent_points) / 3.0
                stats["form_ewma"] = np.average(
                    recent_points, weights=[1, 1.5, 2]
                ) / 3.0

            # ─── ATTACK VS DEF ────────────────────────
            if goals_for and goals_against:
                stats["attack_vs_def"] = (
                    np.mean(goals_for) - np.mean(goals_against)
                ) / (np.mean(goals_for) + np.mean(goals_against) + 1e-6)
                stats["attack_adj"] = np.mean(goals_for) / 1.5
                stats["pressure_form"] = (np.mean(points) / 3.0 + np.mean(goals_for) / 2.5) / 2

            # ─── REST DAYS (default) ───────────────────
            stats["rest_days"] = 3

            # ─── ODDS (from data if available) ────────
            if "odds_home" in matches.columns:
                stats["odds_home"] = matches["odds_home"].iloc[-1] if len(matches) > 0 else 2.5
            if "odds_draw" in matches.columns:
                stats["odds_draw"] = matches["odds_draw"].iloc[-1] if len(matches) > 0 else 3.0
            if "odds_away" in matches.columns:
                stats["odds_away"] = matches["odds_away"].iloc[-1] if len(matches) > 0 else 2.5

            return stats

        except Exception as e:
            logger.warning(f"⚠️  Error extracting stats for {team}: {e}")
            return {}

    def _get_standings_position(
        self,
        home_team: str,
        away_team: str,
        week: int | None = None,
    ) -> tuple[dict, dict]:
        """Get standings position and pressure index for both teams."""
        try:
            if not self.standings_by_week or week is None:
                return (
                    {"position": 10, "points": 30, "pressure": 0.5},
                    {"position": 10, "points": 30, "pressure": 0.5},
                )

            if week not in self.standings_by_week:
                week = max(self.standings_by_week.keys())

            standings = self.standings_by_week[week]

            home_row = standings[standings["team"] == home_team]
            away_row = standings[standings["team"] == away_team]

            home_pos = (
                {
                    "position": int(home_row["position"].iloc[0]),
                    "points": int(home_row["points"].iloc[0]),
                    "pressure": 1.0 - (int(home_row["position"].iloc[0]) - 1) / 19.0,
                }
                if len(home_row) > 0
                else {"position": 10, "points": 30, "pressure": 0.5}
            )

            away_pos = (
                {
                    "position": int(away_row["position"].iloc[0]),
                    "points": int(away_row["points"].iloc[0]),
                    "pressure": 1.0 - (int(away_row["position"].iloc[0]) - 1) / 19.0,
                }
                if len(away_row) > 0
                else {"position": 10, "points": 30, "pressure": 0.5}
            )

            return home_pos, away_pos

        except Exception as e:
            logger.warning(f"⚠️  Error getting standings position: {e}")
            return (
                {"position": 10, "points": 30, "pressure": 0.5},
                {"position": 10, "points": 30, "pressure": 0.5},
            )

    def _get_h2h(self, home_team: str, away_team: str) -> float:
        """Get head-to-head win rate for home team."""
        try:
            h2h = self.processed_df[
                ((self.processed_df["home_team"] == home_team) & 
                 (self.processed_df["away_team"] == away_team)) |
                ((self.processed_df["home_team"] == away_team) & 
                 (self.processed_df["away_team"] == home_team))
            ]

            if h2h.empty:
                return 0.5

            home_wins = len(h2h[(h2h["home_team"] == home_team) & (h2h["result"] == "H")])
            total = len(h2h[h2h["home_team"] == home_team])

            return home_wins / total if total > 0 else 0.5

        except Exception as e:
            logger.warning(f"⚠️  Error calculating H2H: {e}")
            return 0.5

    def _get_win_streak(self, points: list) -> int:
        """Calculate current win streak."""
        if not points:
            return 0
        streak = 0
        for p in reversed(points):
            if p == 3:
                streak += 1
            else:
                break
        return streak

    def _get_default_features(self) -> dict:
        """Return neutral default features."""
        return {
            "home_elo": 1500,
            "home_form": 0.5,
            "home_goals_scored_avg": 1.5,
            "home_goals_conceded_avg": 1.5,
            "home_shots_avg": 15,
            "home_shots_on_target_avg": 5,
            "home_corners_avg": 5,
            "home_fouls": 12,
            "home_attack_vs_def": 0.0,
            "away_elo": 1500,
            "away_form": 0.5,
            "away_goals_scored_avg": 1.5,
            "away_goals_conceded_avg": 1.5,
            "away_shots_avg": 15,
            "away_shots_on_target_avg": 5,
            "away_corners_avg": 5,
            "away_fouls": 12,
            "away_attack_vs_def": 0.0,
            "home_shots": 15,
            "away_shots": 15,
            "home_shots_on_target": 5,
            "away_shots_on_target": 5,
            "home_corners": 5,
            "away_corners": 5,
            "elo_diff": 0,
            "form_diff": 0,
            "elo_form_interaction": 0,
            "tempo_diff": 0,
            "control_diff": 0,
            "attack_vs_defence": 0.0,
            "home_form_lag_1": 0.5,
            "home_form_lag_2": 0.5,
            "home_form_lag_3": 0.5,
            "away_form_lag_1": 0.5,
            "away_form_lag_2": 0.5,
            "away_form_lag_3": 0.5,
            "home_momentum": 0.0,
            "away_momentum": 0.0,
            "home_win_streak": 0,
            "away_win_streak": 0,
            "home_volatility": 0.5,
            "away_volatility": 0.5,
            "home_conversion_rate": 0.1,
            "away_conversion_rate": 0.1,
            "home_form_decayed": 0.5,
            "away_form_decayed": 0.5,
            "home_form_ewma": 0.5,
            "away_form_ewma": 0.5,
            "home_attack_adj": 0.0,
            "away_attack_adj": 0.0,
            "home_rest_days": 3,
            "away_rest_days": 3,
            "pressure_index_home": 0.5,
            "pressure_index_away": 0.5,
            "pressure_form_home": 0.5,
            "pressure_form_away": 0.5,
            "home_pressure_form": 0.5,
            "away_pressure_form": 0.5,
            "elo_pressure_interaction": 0.0,
            "odds_home": 2.5,
            "odds_draw": 3.0,
            "odds_away": 2.5,
            "h2h_home_win_rate": 0.5,
        }