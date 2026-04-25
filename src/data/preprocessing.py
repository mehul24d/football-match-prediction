"""
src/data/preprocessing.py
--------------------------
Data preprocessing utilities.
"""

import pandas as pd
from loguru import logger


def preprocess_raw_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw football CSV data.
    
    Maps raw columns to standard format:
    - HomeTeam → home_team
    - AwayTeam → away_team
    - FTHG → home_goals
    - FTAG → away_goals
    - FTR → result (H/A/D)
    - Date → date
    - HS, AS → home_shots, away_shots
    - HST, AST → home_shots_on_target, away_shots_on_target
    - HC, AC → home_corners, away_corners
    - HF, AF → home_fouls, away_fouls
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw CSV data
        
    Returns:
    --------
    df : pd.DataFrame
        Preprocessed data with standard columns
    """
    df = df.copy()
    
    # Core columns mapping
    column_mapping = {
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'FTR': 'result',
        'Date': 'date',
        'HS': 'home_shots',
        'AS': 'away_shots',
        'HST': 'home_shots_on_target',
        'AST': 'away_shots_on_target',
        'HC': 'home_corners',
        'AC': 'away_corners',
        'HF': 'home_fouls',
        'AF': 'away_fouls',
        'HY': 'home_yellow_cards',
        'AY': 'away_yellow_cards',
        'HR': 'home_red_cards',
        'AR': 'away_red_cards',
    }
    
    # Rename available columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Convert date format (handles both DD/MM/YYYY and other formats)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    
    # Convert numeric columns
    numeric_cols = [
        'home_goals', 'away_goals', 'home_shots', 'away_shots',
        'home_shots_on_target', 'away_shots_on_target',
        'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
        'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only required columns
    required_cols = [
        'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result',
        'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
        'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
        'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards'
    ]
    
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols].dropna(subset=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def load_processed(filepath: str | pd.DataFrame) -> pd.DataFrame:
    """
    Load preprocessed data.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    df : pd.DataFrame
        Loaded data
    """
    if isinstance(filepath, pd.DataFrame):
        return filepath
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df