"""
src/data/run_pipeline.py
------------------------
Main entry point for data preprocessing pipeline.

Usage:
    python src/data/run_pipeline.py
"""

from pathlib import Path
from src.utils.helpers import load_config
from src.data.pipeline import preprocess_and_build_standings


def main():
    """Run the complete preprocessing pipeline."""
    
    # Load config
    config = load_config("configs/config.yaml")
    
    raw_data_dir = Path(config["data"]["raw_dir"])
    processed_data_dir = Path(config["data"]["processed_dir"])
    
    # Create standings directory
    standings_data_dir = Path(config["data"]["processed_dir"]) / "standings"
    
    # Run pipeline
    all_standings = preprocess_and_build_standings(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        standings_data_dir=standings_data_dir,
        overwrite=False,
    )
    
    print("\n" + "="*100)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*100)
    print(f"\n📁 Output Directory Structure:")
    print(f"   {processed_data_dir}/")
    print(f"   ├── matches_E0_2022_23.csv")
    print(f"   ├── matches_D1_2021_22.csv")
    print(f"   ├── ... (processed match data)")
    print(f"   └── standings/")
    print(f"       ├── standings_E0_2022_23.csv (combined)")
    print(f"       ├── E0/2022_23/")
    print(f"       │   ├── matchday_01.csv")
    print(f"       │   ├── matchday_02.csv")
    print(f"       │   └── ... (individual matchday tables)")
    print(f"       └── ... (other countries/seasons)\n")


if __name__ == "__main__":
    main()