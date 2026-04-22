from __future__ import annotations

import pandas as pd
from pathlib import Path
from rapidfuzz import process

from src.data.preprocessing import load_processed
from src.features.engineering import build_features


class LiveFeatureBuilder:
    def __init__(self, data_path="data/processed/matches_clean.csv"):
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                "Processed dataset not found. Run pipeline first."
            )

        self.df = load_processed(self.data_path)

        # Build features once (heavy operation → do once only)
        self.feature_df = build_features(self.df)

        # ✅ Precompute ALL unique teams (home + away)
        self.teams = pd.concat([
            self.feature_df["home_team"],
            self.feature_df["away_team"]
        ]).unique()

    # ─────────────────────────────────────────────
    # Team normalization (SAFE fuzzy matching)
    # ─────────────────────────────────────────────
    def _normalize_team(self, team: str) -> str:
        team = team.lower().strip()

        match = process.extractOne(team, self.teams)

        if match is None:
            raise ValueError(f"Team '{team}' not recognized")

        matched_name, score, _ = match

        # ✅ stricter threshold (prevents nonsense like "lipton")
        if score < 85:
            suggestions = process.extract(
                team, self.teams, limit=5
            )
            suggestions = [s[0] for s in suggestions]

            raise ValueError(
                f"Team '{team}' not recognized. "
                f"Did you mean: {suggestions}?"
            )

        return matched_name

    # ─────────────────────────────────────────────
    # Get latest team stats
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

        if latest["home_team"] == team:
            return {
                "elo": latest["home_elo"],
                "form": latest["home_form"],
                "form_decayed": latest["home_form_decayed"],
                "goals_scored": latest["home_goals_scored_avg"],
                "goals_conceded": latest["home_goals_conceded_avg"],
                "shots": latest["home_shots_avg"],
                "shots_on_target": latest["home_shots_on_target_avg"],
                "corners": latest["home_corners_avg"],
                "rest_days": latest["home_rest_days"],
            }
        else:
            return {
                "elo": latest["away_elo"],
                "form": latest["away_form"],
                "form_decayed": latest["away_form_decayed"],
                "goals_scored": latest["away_goals_scored_avg"],
                "goals_conceded": latest["away_goals_conceded_avg"],
                "shots": latest["away_shots_avg"],
                "shots_on_target": latest["away_shots_on_target_avg"],
                "corners": latest["away_corners_avg"],
                "rest_days": latest["away_rest_days"],
            }

    # ─────────────────────────────────────────────
    # Build match features (FINAL)
    # ─────────────────────────────────────────────
    def build_match_features(self, home_team: str, away_team: str):
        home = self.get_latest_team_stats(home_team)
        away = self.get_latest_team_stats(away_team)

        elo_diff = home["elo"] - away["elo"]
        form_diff = home["form"] - away["form"]

        return {
            "home_elo": home["elo"],
            "away_elo": away["elo"],
            "elo_diff": elo_diff,

            "home_form": home["form"],
            "away_form": away["form"],
            "form_diff": form_diff,

            "home_form_decayed": home["form_decayed"],
            "away_form_decayed": away["form_decayed"],

            "home_goals_scored_avg": home["goals_scored"],
            "home_goals_conceded_avg": home["goals_conceded"],
            "away_goals_scored_avg": away["goals_scored"],
            "away_goals_conceded_avg": away["goals_conceded"],

            "home_shots_avg": home["shots"],
            "away_shots_avg": away["shots"],

            "home_shots_on_target_avg": home["shots_on_target"],
            "away_shots_on_target_avg": away["shots_on_target"],

            "home_corners_avg": home["corners"],
            "away_corners_avg": away["corners"],

            "home_rest_days": home["rest_days"],
            "away_rest_days": away["rest_days"],

            "elo_form_interaction": elo_diff * form_diff,
        }