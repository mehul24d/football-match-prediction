"""
tests/test_rolling_standings.py
--------------------------------
Test rolling standings with verbose output - country/season wise.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.preprocessing import preprocess_raw_csv
from src.features.rolling_standings import (
    build_rolling_standings_country_season,
    print_league_table,
)
from src.utils.helpers import load_config
from loguru import logger


def test_epl_2022_23():
    """Test EPL 2022/23 every matchday."""
    
    config = load_config("configs/config.yaml")
    data_dir = Path(config["data"]["raw_dir"])
    
    # Try to find EPL 2022/23 file
    epl_file = data_dir / "E0_2022_23.csv"
    
    if not epl_file.exists():
        logger.error(f"File not found: {epl_file}")
        print(f"Available files in {data_dir}:")
        for f in sorted(data_dir.glob("*.csv")):
            print(f"  - {f.name}")
        return
    
    logger.info(f"Loading {epl_file.name}...")
    df = pd.read_csv(epl_file)
    
    # Preprocess
    df = preprocess_raw_csv(df)
    
    logger.info(f"Preprocessed data:")
    logger.info(f"  Total matches: {len(df)}")
    logger.info(f"  Unique teams: {df['home_team'].nunique()}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Columns: {df.columns.tolist()}\n")
    
    # Build standings with verbose output
    standings_by_matchday = build_rolling_standings_country_season(
        df,
        country="E0",
        season="2022_23",
        verbose=True,
        print_every_matchday=True,  # ← PRINT EVERY MATCHDAY
    )
    
    print(f"\n{'='*100}")
    print(f"{'✅ SUMMARY - FINAL STANDINGS':^100}")
    print(f"{'='*100}\n")
    
    # Print final standings
    final_matchday = max(standings_by_matchday.keys())
    final_standings = standings_by_matchday[final_matchday]
    
    print_league_table(
        final_standings,
        country="E0",
        season="2022/23",
        matchday=final_matchday,
    )


def test_sample_matchdays():
    """Print specific matchdays."""
    
    config = load_config("configs/config.yaml")
    data_dir = Path(config["data"]["raw_dir"])
    
    epl_file = data_dir / "E0_2022_23.csv"
    
    if not epl_file.exists():
        logger.error(f"File not found: {epl_file}")
        return
    
    logger.info(f"Loading {epl_file.name}...")
    df = pd.read_csv(epl_file)
    df = preprocess_raw_csv(df)
    
    # Build without printing every matchday
    standings_by_matchday = build_rolling_standings_country_season(
        df,
        country="E0",
        season="2022_23",
        verbose=True,
        print_every_matchday=False,
    )
    
    # Print specific matchdays
    print(f"\n{'='*100}")
    print(f"{'SPECIFIC MATCHDAYS':^100}")
    print(f"{'='*100}\n")
    
    for matchday in [1, 5, 10, 15, 20, 25, 30, 35, 38]:
        if matchday in standings_by_matchday:
            print_league_table(
                standings_by_matchday[matchday],
                country="E0",
                season="2022/23",
                matchday=matchday,
                top_n=6,  # Only top 6
            )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        test_sample_matchdays()
    else:
        # Default: test EPL 2022/23 with every matchday
        test_epl_2022_23()