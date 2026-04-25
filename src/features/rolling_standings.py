"""
src/features/rolling_standings.py
----------------------------------
Builds rolling (no-leakage) standings from historical match data.
Country-wise and Season-wise standings with proper matchday tracking.

MATCHDAY LOGIC:
- Matchday N = when EVERY team has played EXACTLY N matches
- Total matchdays = (num_teams - 1) * 2
- Example: 20 teams → 38 matchdays, 18 teams → 34 matchdays

We only save standings AFTER all teams have completed a full round.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from loguru import logger
from tabulate import tabulate
from pathlib import Path


# ─── Internal table helpers ──────────────────────────────────────────────────

def _init_table(teams: list[str]) -> dict:
    """Initialize standings table for teams."""
    return {
        team: {
            "position": 0,
            "played": 0,
            "won": 0,
            "drawn": 0,
            "lost": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "points": 0,
        }
        for team in teams
    }


def _update_table(
    table: dict,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
) -> None:
    """Update standings after a match."""
    # Home team
    table[home]["played"] += 1
    table[home]["goals_for"] += home_goals
    table[home]["goals_against"] += away_goals

    # Away team
    table[away]["played"] += 1
    table[away]["goals_for"] += away_goals
    table[away]["goals_against"] += home_goals

    # Result
    if home_goals > away_goals:
        table[home]["won"] += 1
        table[home]["points"] += 3
        table[away]["lost"] += 1
    elif home_goals < away_goals:
        table[away]["won"] += 1
        table[away]["points"] += 3
        table[home]["lost"] += 1
    else:
        table[home]["drawn"] += 1
        table[home]["points"] += 1
        table[away]["drawn"] += 1
        table[away]["points"] += 1

    # Update goal diff
    table[home]["goal_diff"] = table[home]["goals_for"] - table[home]["goals_against"]
    table[away]["goal_diff"] = table[away]["goals_for"] - table[away]["goals_against"]


def _table_to_df(table: dict) -> pd.DataFrame:
    """Convert standings table to DataFrame."""
    rows = []
    for team, stats in table.items():
        rows.append({"team": team, **stats})

    df = pd.DataFrame(rows)

    # Sort by points (desc), then goal diff, then goals for
    df = df.sort_values(
        by=["points", "goal_diff", "goals_for"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # Add position (1-indexed)
    df["position"] = range(1, len(df) + 1)

    return df


def _get_country_name(country_code: str) -> str:
    """Convert country code to league name."""
    mapping = {
        "E0": "🏴 Premier League (England)",
        "E1": "🏴 Championship (England)",
        "D1": "🇩🇪 Bundesliga (Germany)",
        "D2": "🇩🇪 2. Bundesliga (Germany)",
        "F1": "🇫🇷 Ligue 1 (France)",
        "F2": "🇫🇷 Ligue 2 (France)",
        "SP1": "🇪🇸 La Liga (Spain)",
        "SP2": "🇪🇸 Segunda División (Spain)",
        "I1": "🇮🇹 Serie A (Italy)",
        "I2": "🇮🇹 Serie B (Italy)",
        "P1": "🇵🇹 Primeira Liga (Portugal)",
    }
    return mapping.get(country_code, country_code)


# ─── VISUALIZATION FUNCTIONS ────────────────────────────────────────────────

def print_league_table(
    standings_df: pd.DataFrame,
    country: str | None = None,
    season: str | None = None,
    matchday: int | None = None,
    top_n: int | None = None,
) -> None:
    """
    Print league table in a formatted table.
    
    Parameters:
    -----------
    standings_df : pd.DataFrame
        Standings DataFrame
    country : str, optional
        Country code (E0, D1, etc.)
    season : str, optional
        Season (e.g., "2022/23")
    matchday : int, optional
        Matchday number
    top_n : int, optional
        Only print top N teams
    """
    df = standings_df.copy()

    if top_n:
        df = df.head(top_n)

    # Format columns for display
    display_df = df[[
        "position", "team", "played", "won", "drawn", "lost",
        "goals_for", "goals_against", "goal_diff", "points"
    ]].copy()

    # Rename for readability
    display_df.columns = [
        "Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"
    ]

    # Build title
    parts = []
    if country:
        parts.append(_get_country_name(country))
    if season:
        parts.append(f"Season {season}")
    if matchday:
        parts.append(f"Matchday {matchday}")
    
    title = " | ".join(parts) if parts else "LEAGUE TABLE"

    # Print header
    print(f"\n{'='*100}")
    print(f"{title:^100}")
    print(f"{'='*100}\n")

    # Print table
    print(tabulate(
        display_df,
        headers="keys",
        tablefmt="grid",
        showindex=False,
        intfmt=","
    ))
    print()


def print_match_summary(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    matchday: int | None = None,
) -> None:
    """Print match result summary."""
    result_emoji = "✅" if home_goals > away_goals else "➖" if home_goals == away_goals else "❌"

    if matchday:
        print(f"  {result_emoji} {home_team:20} {home_goals}-{away_goals} {away_team:20}")
    else:
        print(f"{result_emoji} {home_team:20} {home_goals}-{away_goals} {away_team:20}")


def get_table_comparison(
    standings_df_before: pd.DataFrame,
    standings_df_after: pd.DataFrame,
    matchday: int,
) -> None:
    """
    Show changes in standings after a matchday.
    """
    # Get teams that changed position
    before = standings_df_before[["team", "position", "points"]].copy()
    after = standings_df_after[["team", "position", "points"]].copy()

    merged = before.merge(after, on="team", suffixes=("_before", "_after"))

    merged["pos_change"] = merged["position_before"] - merged["position_after"]
    merged["pts_change"] = merged["points_after"] - merged["points_before"]

    # Filter teams with changes
    changes = merged[
        (merged["pos_change"] != 0) | (merged["pts_change"] != 0)
    ].copy()

    if changes.empty:
        return

    # Sort by biggest gainers
    changes = changes.sort_values("pos_change", ascending=False)

    print(f"  📊 Position Changes:")
    for _, row in changes.iterrows():
        pos_arrow = "📈" if row["pos_change"] > 0 else "📉"
        print(f"    {pos_arrow} {row['team']:20} {int(row['position_before']):2d}→{int(row['position_after']):2d}  "
              f"({int(row['points_before']):2d}→{int(row['points_after']):2d} pts)")
    print()


# ─── MAIN ROLLING STANDINGS BUILDER ──────────────────────────────────────────

def build_rolling_standings_country_season(
    df: pd.DataFrame,
    country: str,
    season: str,
    verbose: bool = True,
    print_every_matchday: bool = False,
) -> dict[int, pd.DataFrame]:
    """
    Build rolling standings for a specific country and season.
    
    Matchday Logic:
    - Matchday N = when EVERY team has played EXACTLY N matches
    - Total matchdays = (num_teams - 1) * 2
    - We only save standings when all teams have completed a round
    
    Parameters:
    -----------
    df : pd.DataFrame
        Match data for the country/season
    country : str
        Country code (E0, D1, F1, SP1)
    season : str
        Season code (e.g., "2022_23")
    verbose : bool
        Print progress
    print_every_matchday : bool
        Print table after each complete matchday
        
    Returns:
    --------
    standings_by_matchday : dict[int, pd.DataFrame]
        Standings for each matchday (all teams have played exactly N games)
    """
    
    if df.empty:
        logger.warning(f"No data for {country} {season}")
        return {}

    # Validate columns
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert to numeric
    df = df.copy()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    # Get unique teams
    teams = sorted(
        set(df["home_team"].unique()) | set(df["away_team"].unique())
    )
    
    num_teams = len(teams)
    expected_matchdays = (num_teams - 1) * 2
    
    if verbose:
        season_formatted = season.replace("_", "/")
        logger.info(
            f"{_get_country_name(country)} {season_formatted} | "
            f"{num_teams} teams | {expected_matchdays} matchdays | {len(df)} matches"
        )

    # Build standings by processing matches in order
    standings_by_matchday = {}
    table = _init_table(teams)
    table_before = None
    prev_min_matches = 0
    
    for idx, row in df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        home_goals = int(row["home_goals"])
        away_goals = int(row["away_goals"])
        
        # Update table
        _update_table(table, home, away, home_goals, away_goals)
        
        # Check if all teams have completed a new round
        # (i.e., minimum matches played increased)
        matches_played = [table[team]["played"] for team in teams]
        current_min_matches = min(matches_played)
        
        # When minimum increases, a matchday is complete
        if current_min_matches > prev_min_matches:
            matchday = prev_min_matches + 1
            
            if verbose and print_every_matchday:
                season_formatted = season.replace("_", "/")
                # Count matches in this matchday
                matchday_match_count = sum(
                    1 for team in teams 
                    if table[team]["played"] == matchday
                ) // 2  # Approximate
                
                print(f"\n{'#'*100}")
                print(f"# {_get_country_name(country)} {season_formatted} | MATCHDAY {matchday}")
                print(f"{'#'*100}\n")

            # Store standings for this matchday
            table_after_df = _table_to_df(table).copy()
            standings_by_matchday[matchday] = table_after_df

            if verbose and print_every_matchday:
                season_formatted = season.replace("_", "/")
                print_league_table(
                    table_after_df,
                    country=country,
                    season=season_formatted,
                    matchday=matchday,
                )

            prev_min_matches = current_min_matches

    if verbose:
        season_formatted = season.replace("_", "/")
        logger.success(
            f"✅ {_get_country_name(country)} {season_formatted} | "
            f"Built standings for {len(standings_by_matchday)} matchdays (expected: {expected_matchdays})"
        )

    return standings_by_matchday


def build_rolling_standings_multi_country(
    data_dir: Path | str,
    country: str | None = None,
    season: str | None = None,
    verbose: bool = True,
    print_every_matchday: bool = False,
) -> dict[str, dict[str, dict[int, pd.DataFrame]]]:
    """
    Build rolling standings for multiple country/season combinations.
    
    Structure: {country: {season: {matchday: standings_df}}}
    """
    data_dir = Path(data_dir)
    all_standings = {}

    # Find all CSV files
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return {}

    # Filter by country/season if specified
    if country or season:
        csv_files = [
            f for f in csv_files
            if (not country or f.stem.startswith(country))
            and (not season or season in f.stem)
        ]

    for csv_file in csv_files:
        # Parse filename (e.g., E0_2022_23.csv)
        stem = csv_file.stem
        parts = stem.split("_")

        if len(parts) < 3:
            logger.warning(f"Skipping {stem} - invalid format")
            continue

        # Extract country and season
        file_country = "_".join(parts[:-2])  # E0, D1, SP1, F1, etc.
        file_season = "_".join(parts[-2:])   # 2022_23

        try:
            df = pd.read_csv(csv_file)
            standings = build_rolling_standings_country_season(
                df,
                country=file_country,
                season=file_season,
                verbose=verbose,
                print_every_matchday=print_every_matchday,
            )

            if standings:
                if file_country not in all_standings:
                    all_standings[file_country] = {}
                all_standings[file_country][file_season] = standings

        except Exception as e:
            logger.error(f"❌ Error processing {stem}: {e}")

    return all_standings


def get_standings_for_match(
    all_standings: dict[str, dict[str, dict[int, pd.DataFrame]]],
    home_team: str,
    away_team: str,
    country: str,
    season: str,
    matchday: int,
) -> tuple[dict, dict]:
    """
    Get standings position for both teams at a specific matchday.
    
    Returns:
    --------
    home_pos, away_pos : tuple
        {position, points, played, won, drawn, lost, goal_diff}
    """
    try:
        if country not in all_standings or season not in all_standings[country]:
            return (
                {"position": 10, "points": 30, "played": 0},
                {"position": 10, "points": 30, "played": 0},
            )

        standings = all_standings[country][season]

        if matchday not in standings:
            # Use latest matchday before this one
            valid_matchdays = [m for m in standings.keys() if m <= matchday]
            if not valid_matchdays:
                return (
                    {"position": 10, "points": 30, "played": 0},
                    {"position": 10, "points": 30, "played": 0},
                )
            matchday = max(valid_matchdays)

        standings_df = standings[matchday]

        home_row = standings_df[standings_df["team"] == home_team]
        away_row = standings_df[standings_df["team"] == away_team]

        home_pos = (
            {
                "position": int(home_row["position"].iloc[0]),
                "points": int(home_row["points"].iloc[0]),
                "played": int(home_row["played"].iloc[0]),
                "won": int(home_row["won"].iloc[0]),
                "drawn": int(home_row["drawn"].iloc[0]),
                "lost": int(home_row["lost"].iloc[0]),
                "goal_diff": int(home_row["goal_diff"].iloc[0]),
            }
            if len(home_row) > 0
            else {"position": 10, "points": 30, "played": 0, "won": 0, "drawn": 0, "lost": 0, "goal_diff": 0}
        )

        away_pos = (
            {
                "position": int(away_row["position"].iloc[0]),
                "points": int(away_row["points"].iloc[0]),
                "played": int(away_row["played"].iloc[0]),
                "won": int(away_row["won"].iloc[0]),
                "drawn": int(away_row["drawn"].iloc[0]),
                "lost": int(away_row["lost"].iloc[0]),
                "goal_diff": int(away_row["goal_diff"].iloc[0]),
            }
            if len(away_row) > 0
            else {"position": 10, "points": 30, "played": 0, "won": 0, "drawn": 0, "lost": 0, "goal_diff": 0}
        )

        return home_pos, away_pos

    except Exception as e:
        logger.warning(f"⚠️  Error getting standings: {e}")
        return (
            {"position": 10, "points": 30, "played": 0, "won": 0, "drawn": 0, "lost": 0, "goal_diff": 0},
            {"position": 10, "points": 30, "played": 0, "won": 0, "drawn": 0, "lost": 0, "goal_diff": 0},
        )
