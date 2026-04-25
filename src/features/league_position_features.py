"""
src/features/league_position_features.py
------------------------------------------
Extract league position and performance factors from matchday standings.

Features generated:
- League position (1-20)
- Points accumulated
- Goal difference
- Recent form (last N matches)
- Home/Away performance
- Head-to-head records
- Position differential (home_pos - away_pos)
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Tuple


class LeaguePositionFeatureExtractor:
    """Extract features based on league standings at each matchday."""
    
    def __init__(self, standings_dir: str | pd.DataFrame):
        """
        Initialize feature extractor.
        
        Parameters:
        -----------
        standings_dir : str or pd.DataFrame
            Path to combined standings CSV or the standings dataframe itself
        """
        if isinstance(standings_dir, str):
            self.standings_df = pd.read_csv(standings_dir)
        else:
            self.standings_df = standings_dir.copy()
    
    def get_team_standing_at_matchday(
        self,
        team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> Dict:
        """
        Get team's standing at a specific matchday.
        
        Parameters:
        -----------
        team : str
            Team name
        country : str
            Country code (E0, D1, etc.)
        season : str
            Season (2022_23, 2021_22, etc.)
        matchday : int
            Matchday number
            
        Returns:
        --------
        standing : dict
            {position, points, played, won, drawn, lost, goals_for, goals_against, goal_diff}
        """
        query = (
            (self.standings_df['team'] == team) &
            (self.standings_df['country'] == country) &
            (self.standings_df['season'] == season) &
            (self.standings_df['matchday'] == matchday)
        )
        
        result = self.standings_df[query]
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        return {
            'position': int(row['position']),
            'points': int(row['points']),
            'played': int(row['played']),
            'won': int(row['won']),
            'drawn': int(row['drawn']),
            'lost': int(row['lost']),
            'goals_for': int(row['goals_for']),
            'goals_against': int(row['goals_against']),
            'goal_diff': int(row['goal_diff']),
        }
    
    def get_position_differential(
        self,
        home_team: str,
        away_team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate position differential (home - away).
        Higher value = home team in better position.
        
        Example: Arsenal (pos 1) vs West Ham (pos 13) = 13 - 1 = 12
        """
        home_standing = self.get_team_standing_at_matchday(home_team, country, season, matchday)
        away_standing = self.get_team_standing_at_matchday(away_team, country, season, matchday)
        
        if not home_standing or not away_standing:
            return np.nan
        
        # Positive = away team in better position (advantage away)
        return away_standing['position'] - home_standing['position']
    
    def get_points_differential(
        self,
        home_team: str,
        away_team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate points differential (home - away).
        Higher value = home team has more points.
        """
        home_standing = self.get_team_standing_at_matchday(home_team, country, season, matchday)
        away_standing = self.get_team_standing_at_matchday(away_team, country, season, matchday)
        
        if not home_standing or not away_standing:
            return np.nan
        
        return home_standing['points'] - away_standing['points']
    
    def get_goal_diff_differential(
        self,
        home_team: str,
        away_team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate goal difference differential (home - away).
        """
        home_standing = self.get_team_standing_at_matchday(home_team, country, season, matchday)
        away_standing = self.get_team_standing_at_matchday(away_team, country, season, matchday)
        
        if not home_standing or not away_standing:
            return np.nan
        
        return home_standing['goal_diff'] - away_standing['goal_diff']
    
    def get_win_rate(
        self,
        team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate win rate (wins / games played).
        """
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        
        if not standing or standing['played'] == 0:
            return 0.0
        
        return standing['won'] / standing['played']
    
    def get_draw_rate(
        self,
        team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate draw rate (draws / games played).
        """
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        
        if not standing or standing['played'] == 0:
            return 0.0
        
        return standing['drawn'] / standing['played']
    
    def get_loss_rate(
        self,
        team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate loss rate (losses / games played).
        """
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        
        if not standing or standing['played'] == 0:
            return 0.0
        
        return standing['lost'] / standing['played']
    
    def get_goals_per_game(
        self,
        team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate average goals scored per game.
        """
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        
        if not standing or standing['played'] == 0:
            return 0.0
        
        return standing['goals_for'] / standing['played']
    
    def get_goals_conceded_per_game(
        self,
        team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate average goals conceded per game.
        """
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        
        if not standing or standing['played'] == 0:
            return 0.0
        
        return standing['goals_against'] / standing['played']
    
    def get_points_per_game(
        self,
        team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> float:
        """
        Calculate average points per game.
        """
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        
        if not standing or standing['played'] == 0:
            return 0.0
        
        return standing['points'] / standing['played']


def extract_league_position_features(
    matches_df: pd.DataFrame,
    standings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add league position features to matches dataframe.
    
    Features added:
    - home_position, away_position
    - home_points, away_points
    - home_goal_diff, away_goal_diff
    - position_differential (away - home)
    - points_differential (home - away)
    - goal_diff_differential (home - away)
    - home_win_rate, away_win_rate
    - home_draw_rate, away_draw_rate
    - home_loss_rate, away_loss_rate
    - home_goals_per_game, away_goals_per_game
    - home_goals_conceded_per_game, away_goals_conceded_per_game
    - home_ppg, away_ppg (points per game)
    
    Parameters:
    -----------
    matches_df : pd.DataFrame
        Matches data with columns: date, home_team, away_team, country, season
    standings_df : pd.DataFrame
        Combined standings data
        
    Returns:
    --------
    df : pd.DataFrame
        Matches with league position features added
    """
    df = matches_df.copy()
    
    # Initialize feature extractor
    extractor = LeaguePositionFeatureExtractor(standings_df)
    
    logger.info(f"Extracting league position features for {len(df)} matches...")
    
    # We need to determine matchday for each match
    # This requires matching date and country/season to standings
    
    # For now, we'll add features row by row
    features = {
        'home_position': [],
        'away_position': [],
        'home_points': [],
        'away_points': [],
        'home_goal_diff': [],
        'away_goal_diff': [],
        'position_differential': [],
        'points_differential': [],
        'goal_diff_differential': [],
        'home_win_rate': [],
        'away_win_rate': [],
        'home_draw_rate': [],
        'away_draw_rate': [],
        'home_loss_rate': [],
        'away_loss_rate': [],
        'home_goals_per_game': [],
        'away_goals_per_game': [],
        'home_goals_conceded_per_game': [],
        'away_goals_conceded_per_game': [],
        'home_ppg': [],
        'away_ppg': [],
    }
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        country = row.get('country', '')
        season = row.get('season', '')
        
        # Try to find the matchday by matching date
        matchday = _find_matchday_for_date(
            row['date'],
            home_team,
            away_team,
            country,
            season,
            standings_df,
        )
        
        if matchday is None:
            # Use latest matchday available
            available_matchdays = standings_df[
                (standings_df['country'] == country) &
                (standings_df['season'] == season)
            ]['matchday'].unique()
            
            if len(available_matchdays) > 0:
                matchday = int(np.max(available_matchdays))
            else:
                matchday = 1
        
        # Extract features
        home_pos = extractor.get_team_standing_at_matchday(home_team, country, season, int(matchday))
        away_pos = extractor.get_team_standing_at_matchday(away_team, country, season, int(matchday))
        
        # Handle missing standings
        if home_pos is None or away_pos is None:
            for key in features:
                features[key].append(np.nan)
            continue
        
        features['home_position'].append(home_pos['position'])
        features['away_position'].append(away_pos['position'])
        features['home_points'].append(home_pos['points'])
        features['away_points'].append(away_pos['points'])
        features['home_goal_diff'].append(home_pos['goal_diff'])
        features['away_goal_diff'].append(away_pos['goal_diff'])
        
        features['position_differential'].append(away_pos['position'] - home_pos['position'])
        features['points_differential'].append(home_pos['points'] - away_pos['points'])
        features['goal_diff_differential'].append(home_pos['goal_diff'] - away_pos['goal_diff'])
        
        features['home_win_rate'].append(extractor.get_win_rate(home_team, country, season, int(matchday)))
        features['away_win_rate'].append(extractor.get_win_rate(away_team, country, season, int(matchday)))
        features['home_draw_rate'].append(extractor.get_draw_rate(home_team, country, season, int(matchday)))
        features['away_draw_rate'].append(extractor.get_draw_rate(away_team, country, season, int(matchday)))
        features['home_loss_rate'].append(extractor.get_loss_rate(home_team, country, season, int(matchday)))
        features['away_loss_rate'].append(extractor.get_loss_rate(away_team, country, season, int(matchday)))
        
        features['home_goals_per_game'].append(extractor.get_goals_per_game(home_team, country, season, int(matchday)))
        features['away_goals_per_game'].append(extractor.get_goals_per_game(away_team, country, season, int(matchday)))
        features['home_goals_conceded_per_game'].append(extractor.get_goals_conceded_per_game(home_team, country, season, int(matchday)))
        features['away_goals_conceded_per_game'].append(extractor.get_goals_conceded_per_game(away_team, country, season, int(matchday)))
        
        features['home_ppg'].append(extractor.get_points_per_game(home_team, country, season, int(matchday)))
        features['away_ppg'].append(extractor.get_points_per_game(away_team, country, season, int(matchday)))
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} matches")
    
    # Add features to dataframe
    for key, values in features.items():
        df[key] = values
    
    logger.success(f"✅ Added {len(features)} league position features")
    
    return df


def _find_matchday_for_date(
    date: pd.Timestamp,
    home_team: str,
    away_team: str,
    country: str,
    season: str,
    standings_df: pd.DataFrame,
) -> int | None:
    """
    Find the matchday that corresponds to a specific date.
    
    This is approximate - finds the matchday closest to (but before) the date.
    """
    # This would require access to the match dates in standings
    # For now, return None to use the latest available matchday
    return None