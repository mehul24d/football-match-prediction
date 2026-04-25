"""
src/data/live_standings.py
---------------------------
Robust LIVE standings wrapper (production-safe)
"""

from __future__ import annotations

import os
import time
import requests
import pandas as pd
from loguru import logger


class LiveStandings:
    BASE_URL = "http://api.football-data.org/v4/competitions/{code}/standings"

    def __init__(self, api_key: str | None = None, retries: int = 3):
        self.api_key = api_key or os.getenv("FOOTBALL_API_KEY")
        self.retries = retries

        if not self.api_key:
            raise ValueError("❌ API key not provided. Set FOOTBALL_API_KEY env variable.")

    # ─────────────────────────────────────────
    # Fetch (with retry + timeout)
    # ─────────────────────────────────────────
    def fetch(self, league_code: str = "PL", season: int = 2023):
        url = self.BASE_URL.format(code=league_code)

        headers = {"X-Auth-Token": self.api_key}
        params = {"season": season}

        for attempt in range(self.retries):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=10  # 🔥 prevent hanging
                )

                if response.status_code == 200:
                    return response.json()

                logger.warning(f"API error ({response.status_code}): {response.text}")

            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")

            time.sleep(1)  # small retry delay

        raise RuntimeError("❌ Failed to fetch standings after retries")

    # ─────────────────────────────────────────
    # Parse
    # ─────────────────────────────────────────
    def parse(self, data: dict) -> pd.DataFrame:
        try:
            table = data["standings"][0]["table"]
        except (KeyError, IndexError):
            raise ValueError("❌ Unexpected API response format")

        rows = []
        for team in table:
            rows.append({
                "team": team["team"]["name"],
                "points": team["points"],
                "position": team["position"],
                "matches_played": team["playedGames"],
                "goal_diff": team["goalDifference"],
                "form": team.get("form", ""),
            })

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────
    def get_standings(self, league_code="PL", season=2023):
        logger.info("🌍 Fetching LIVE standings...")

        try:
            data = self.fetch(league_code, season)
            df = self.parse(data)

            logger.success("✅ Live standings fetched")
            return df

        except Exception as e:
            logger.error(f"❌ Live standings failed: {e}")

            # 🔥 Safe fallback (important for pipeline)
            return pd.DataFrame({
                "team": [],
                "points": [],
                "position": [],
                "matches_played": [],
                "goal_diff": [],
                "form": []
            })