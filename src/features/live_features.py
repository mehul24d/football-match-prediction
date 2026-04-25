from __future__ import annotations

import pandas as pd
from pathlib import Path
from rapidfuzz import process

from src.data.preprocessing import load_processed
from src.features.engineering import build_features

# Advanced features
from src.features.advanced_features import (
    add_opponent_adjusted_metrics,
    add_tactical_features,
    add_interaction_features,
    compute_weighted_form,
)

# 🔥 NEW: Live standings
from src.data.live_standings import LiveStandings

# OPTIONAL pressure engine
try:
    from src.features.match_importance import MatchImportanceCalculator
except ImportError:
    MatchImportanceCalculator = None


class LiveFeatureBuilder:
    def __init__(self, data_path="data/processed/matches_clean.csv"):
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                "Processed dataset not found. Run pipeline first."
            )

        self.df = load_processed(self.data_path)

        # ─────────────────────────────────────────────
        # Base features
        # ─────────────────────────────────────────────
        self.feature_df = build_features(self.df)

        # ─────────────────────────────────────────────
        # Advanced features 🔥
        # ─────────────────────────────────────────────
        self.feature_df = compute_weighted_form(self.feature_df)
        self.feature_df = add_opponent_adjusted_metrics(self.feature_df)
        self.feature_df = add_tactical_features(self.feature_df)
        self.feature_df = add_interaction_features(self.feature_df)

        # Teams
        self.teams = pd.concat([
            self.feature_df["home_team"],
            self.feature_df["away_team"]
        ]).unique()

        # 🔥 INIT LIVE STANDINGS CLIENT
        self.standings_api = LiveStandings()

    # ─────────────────────────────────────────────
    # Team normalization
    # ─────────────────────────────────────────────
    def _normalize_team(self, team: str) -> str:
        team = team.lower().strip()

        match = process.extractOne(team, self.teams)

        if match is None:
            raise ValueError(f"Team '{team}' not recognized")

        matched_name, score, _ = match

        if score < 85:
            suggestions = process.extract(team, self.teams, limit=5)
            suggestions = [s[0] for s in suggestions]

            raise ValueError(
                f"Team '{team}' not recognized. Did you mean: {suggestions}?"
            )

        return matched_name

    # ─────────────────────────────────────────────
    # Latest stats
    # ─────────────────────────────────────────────
    def get_latest_team_stats(self, team: str):
        team = self._normalize_team(team)

        df = self.feature_df

        team_games = df[
            (df["home_team"] == team) | (df["away_team"] == team)
        ].sort_values("date")

        if team_games.empty:
            raise ValueError(f"No data found for team: {team}")

        latest = team_games.iloc[-1]

        prefix = "home_" if latest["home_team"] == team else "away_"

        return {
            "elo": latest[f"{prefix}elo"],
            "form": latest[f"{prefix}form"],
            "form_decayed": latest.get(f"{prefix}form_decayed", 0),
            "form_ewma": latest.get(f"{prefix}form_ewma", 0),

            "goals_scored": latest[f"{prefix}goals_scored_avg"],
            "goals_conceded": latest[f"{prefix}goals_conceded_avg"],

            "shots": latest[f"{prefix}shots_avg"],
            "shots_on_target": latest[f"{prefix}shots_on_target_avg"],
            "corners": latest[f"{prefix}corners_avg"],

            "rest_days": latest[f"{prefix}rest_days"],

            # Advanced
            "attack_adj": latest.get(f"{prefix}attack_adj", 0),
            "defence_adj": latest.get(f"{prefix}defence_adj", 0),
            "conversion": latest.get(f"{prefix}conversion_rate", 0),
            "def_compact": latest.get(f"{prefix}defensive_compactness", 0),
        }

    # ─────────────────────────────────────────────
    # 🔥 AUTO STANDINGS FETCH
    # ─────────────────────────────────────────────
    def _get_live_context(self):
        try:
            standings = self.standings_api.get_table()

            # Estimate current week dynamically
            current_week = int(standings["played"].max())

            return standings, current_week

        except Exception:
            # fallback if API fails
            return None, None

    # ─────────────────────────────────────────────
    # FINAL MATCH FEATURES 🚀
    # ─────────────────────────────────────────────
    def build_match_features(
        self,
        home_team: str,
        away_team: str,
        standings: pd.DataFrame | None = None,
        current_week: int | None = None,
    ):
        home = self.get_latest_team_stats(home_team)
        away = self.get_latest_team_stats(away_team)

        elo_diff = home["elo"] - away["elo"]
        form_diff = home["form"] - away["form"]

        # ─────────────────────────────────────────────
        # 🔥 AUTO CONTEXT (if not passed)
        # ─────────────────────────────────────────────
        if standings is None or current_week is None:
            standings, current_week = self._get_live_context()

        # ─────────────────────────────────────────────
        # PRESSURE INDEX
        # ─────────────────────────────────────────────
        pressure_home, pressure_away = 0.5, 0.5

        if (
            MatchImportanceCalculator
            and standings is not None
            and current_week is not None
        ):
            calc = MatchImportanceCalculator(standings)
            pressure_home, pressure_away = calc.calculate(
                home_team, away_team, current_week
            )

        # ─────────────────────────────────────────────
        # FINAL FEATURES
        # ─────────────────────────────────────────────
        return {
            "home_elo": home["elo"],
            "away_elo": away["elo"],
            "elo_diff": elo_diff,

            "home_form": home["form"],
            "away_form": away["form"],
            "form_diff": form_diff,

            "home_form_decayed": home["form_decayed"],
            "away_form_decayed": away["form_decayed"],

            "home_form_ewma": home["form_ewma"],
            "away_form_ewma": away["form_ewma"],

            "home_attack_adj": home["attack_adj"],
            "away_attack_adj": away["attack_adj"],
            "home_defence_adj": home["defence_adj"],
            "away_defence_adj": away["defence_adj"],

            "home_conversion_rate": home["conversion"],
            "away_conversion_rate": away["conversion"],

            "home_defensive_compactness": home["def_compact"],
            "away_defensive_compactness": away["def_compact"],

            "home_goals_scored_avg": home["goals_scored"],
            "away_goals_scored_avg": away["goals_scored"],

            "home_shots_avg": home["shots"],
            "away_shots_avg": away["shots"],

            "home_rest_days": home["rest_days"],
            "away_rest_days": away["rest_days"],

            # interactions
            "elo_form_interaction": elo_diff * form_diff,
            "attack_vs_defence": home["attack_adj"] / (away["defence_adj"] + 1e-6),

            # 🔥 pressure
            "pressure_index_home": pressure_home,
            "pressure_index_away": pressure_away,

            "home_pressure_form": pressure_home * home["form"],
            "away_pressure_form": pressure_away * away["form"],

            "elo_pressure_interaction": elo_diff * (pressure_home - pressure_away),
        }