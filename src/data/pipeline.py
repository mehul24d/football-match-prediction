"""
src/data/pipeline.py
---------------------
Complete data preprocessing pipeline:
1. Load raw CSV files
2. Preprocess match data
3. Build rolling standings (matchday tables)
4. Save processed outputs

Usage:
    from src.data.pipeline import run_pipeline
    all_standings = run_pipeline(config_path="configs/config.yaml")
    
    Or from command line:
    python -m src.data.pipeline
"""

from pathlib import Path
from loguru import logger
import pandas as pd

from src.data.preprocessing import preprocess_raw_csv, load_processed
from src.features.rolling_standings import build_rolling_standings_country_season
from src.utils.helpers import load_config, ensure_dir


def preprocess_and_build_standings(
    raw_data_dir: Path,
    processed_data_dir: Path,
    standings_data_dir: Path,
    overwrite: bool = False,
) -> dict[str, dict[str, dict[int, pd.DataFrame]]]:
    """
    Complete pipeline:
    1. Load raw CSV files
    2. Preprocess match data → matches_{country}_{season}.csv
    3. Build rolling standings → standings_{country}_{season}.csv
    4. Save individual matchday tables
    
    Parameters:
    -----------
    raw_data_dir : Path
        Directory with raw CSV files (E0_2022_23.csv, etc.)
    processed_data_dir : Path
        Directory to save processed matches
    standings_data_dir : Path
        Directory to save matchday standings tables
    overwrite : bool
        If True, reprocess even if files exist
        
    Returns:
    --------
    all_standings : dict[str, dict[str, dict[int, pd.DataFrame]]]
        {country: {season: {matchday: standings_df}}}
    """
    
    raw_data_dir = Path(raw_data_dir)
    processed_data_dir = Path(processed_data_dir)
    standings_data_dir = Path(standings_data_dir)
    
    # Create output directories
    ensure_dir(processed_data_dir)
    ensure_dir(standings_data_dir)
    
    # Find all raw CSV files
    csv_files = sorted(raw_data_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error(f"❌ No CSV files found in {raw_data_dir}")
        return {}
    
    logger.info(f"Found {len(csv_files)} raw CSV files\n")
    all_standings = {}
    
    for csv_file in csv_files:
        stem = csv_file.stem
        parts = stem.split("_")
        
        if len(parts) < 3:
            logger.warning(f"⏭️  Skipping {stem} - invalid format (expected: COUNTRY_YEAR_YEAR)")
            continue
        
        # Extract country and season
        file_country = "_".join(parts[:-2])  # E0, D1, SP1, F1
        file_season = "_".join(parts[-2:])   # 2022_23
        
        logger.info(f"{'='*100}")
        logger.info(f"Processing: {stem}")
        logger.info(f"{'='*100}")
        
        try:
            # ─── STEP 1: Load raw data ───────────────────────────────────────
            logger.info(f"1️⃣  Loading raw data from {csv_file.name}...")
            df_raw = pd.read_csv(csv_file)
            logger.info(f"   ✅ Loaded {len(df_raw)} matches")
            
            # ─── STEP 2: Preprocess matches ───────────────────────────────────
            logger.info(f"2️⃣  Preprocessing match data...")
            df_processed = preprocess_raw_csv(df_raw)
            logger.info(f"   ✅ Preprocessed to {len(df_processed)} matches")
            
            # Save processed matches
            processed_file = processed_data_dir / f"matches_{file_country}_{file_season}.csv"
            df_processed.to_csv(processed_file, index=False)
            logger.info(f"   💾 Saved to {processed_file.name}\n")
            
            # ─── STEP 3: Build rolling standings ──────────────────────────────
            logger.info(f"3️⃣  Building rolling standings...")
            standings = build_rolling_standings_country_season(
                df_processed,
                country=file_country,
                season=file_season,
                verbose=True,
                print_every_matchday=False,
            )
            
            if not standings:
                logger.warning(f"⚠️  No standings generated for {stem}\n")
                continue
            
            # ─── STEP 4: Save standings for each matchday ─────────────────────
            logger.info(f"4️⃣  Saving matchday standings...")
            
            # Create country/season subdirectory
            standings_subdir = standings_data_dir / file_country / file_season
            ensure_dir(standings_subdir)
            
            # Save each matchday as separate CSV
            for matchday, standings_df in standings.items():
                standings_file = standings_subdir / f"matchday_{matchday:02d}.csv"
                standings_df.to_csv(standings_file, index=False)
            
            logger.info(f"   ✅ Saved {len(standings)} matchday tables")
            logger.info(f"   📁 Location: {standings_subdir}\n")
            
            # ─── STEP 5: Save combined standings CSV ──────────────────────────
            logger.info(f"5️⃣  Creating combined standings file...")
            
            # Flatten standings: each row = one team at one matchday
            combined_rows = []
            for matchday, standings_df in standings.items():
                temp_df = standings_df.copy()
                temp_df['matchday'] = matchday
                temp_df['country'] = file_country
                temp_df['season'] = file_season
                combined_rows.append(temp_df)
            
            if combined_rows:
                combined_standings_df = pd.concat(combined_rows, ignore_index=True)
                
                # Reorder columns
                cols = [
                    'country', 'season', 'matchday', 'position', 'team',
                    'played', 'won', 'drawn', 'lost',
                    'goals_for', 'goals_against', 'goal_diff', 'points'
                ]
                combined_standings_df = combined_standings_df[cols]
                
                combined_file = standings_data_dir / f"standings_{file_country}_{file_season}.csv"
                combined_standings_df.to_csv(combined_file, index=False)
                logger.info(f"   ✅ Saved combined standings")
                logger.info(f"   📁 File: {combined_file.name}\n")
            
            # Store in memory
            if file_country not in all_standings:
                all_standings[file_country] = {}
            all_standings[file_country][file_season] = standings
            
        except Exception as e:
            logger.error(f"❌ Error processing {stem}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_standings


def run_pipeline(config_path: str = "configs/config.yaml") -> dict:
    """
    Main entry point for the entire data preprocessing pipeline.
    
    Parameters:
    -----------
    config_path : str
        Path to config YAML file
        
    Returns:
    --------
    result : dict
        {
            'all_standings': dict,
            'processed_dir': Path,
            'standings_dir': Path,
            'summary': dict
        }
    """
    
    # Load config
    logger.info(f"Loading config from {config_path}...")
    config = load_config(config_path)
    
    raw_data_dir = Path(config["data"]["raw_dir"])
    processed_data_dir = Path(config["data"]["processed_dir"])
    standings_data_dir = processed_data_dir / "standings"
    
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Processed data directory: {processed_data_dir}")
    logger.info(f"Standings directory: {standings_data_dir}\n")
    
    # Run preprocessing and standings building
    all_standings = preprocess_and_build_standings(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        standings_data_dir=standings_data_dir,
        overwrite=False,
    )
    
    # ─── Print Summary ────────────────────────────────────────────────────────
    logger.info(f"\n{'='*100}")
    logger.info(f"{'PIPELINE EXECUTION COMPLETE':^100}")
    logger.info(f"{'='*100}\n")
    
    summary = {}
    for country in sorted(all_standings.keys()):
        for season in sorted(all_standings[country].keys()):
            matchdays = len(all_standings[country][season])
            key = f"{country}_{season}"
            summary[key] = matchdays
            logger.info(f"✅ {country} {season}: {matchdays} matchdays")
    
    logger.info(f"\n{'📊 OUTPUT DIRECTORY STRUCTURE':^100}")
    logger.info(f"{'='*100}\n")
    logger.info(f"data/processed/")
    logger.info(f"├── matches_E0_2022_23.csv")
    logger.info(f"├── matches_E0_2021_22.csv")
    logger.info(f"├── matches_D1_2022_23.csv")
    logger.info(f"├── ... (all preprocessed match data)")
    logger.info(f"└── standings/")
    logger.info(f"    ├── standings_E0_2022_23.csv (combined all matchdays)")
    logger.info(f"    ├── standings_E0_2021_22.csv")
    logger.info(f"    ├── standings_D1_2022_23.csv")
    logger.info(f"    ├── E0/")
    logger.info(f"    │   ├── 2022_23/")
    logger.info(f"    │   │   ├── matchday_01.csv")
    logger.info(f"    │   │   ├── matchday_02.csv")
    logger.info(f"    │   │   └── ... (38 files for EPL)")
    logger.info(f"    │   └── 2021_22/")
    logger.info(f"    │       └── ... (matchday tables)")
    logger.info(f"    └── D1/")
    logger.info(f"        └── 2022_23/")
    logger.info(f"            └── ... (matchday tables)\n")
    
    return {
        'all_standings': all_standings,
        'processed_dir': processed_data_dir,
        'standings_dir': standings_data_dir,
        'summary': summary,
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_pipeline()
