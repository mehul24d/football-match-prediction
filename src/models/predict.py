"""
src/models/predict.py
---------------------
Load the best saved model and produce match-outcome probabilities.

Usage (CLI)
-----------
    python -m src.models.predict \
        --home "Arsenal" \
        --away "Chelsea" \
        --home-elo 1620 \
        --away-elo 1580

Usage (library)
---------------
    from src.models.predict import Predictor
    predictor = Predictor()
    result = predictor.predict(home_features)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import load_config


# ─── Predictor class ─────────────────────────────────────────────────────────

class Predictor:
    """
    Wraps the serialised model + scaler to produce outcome probabilities.

    Attributes
    ----------
    model          : fitted sklearn-compatible classifier
    scaler         : fitted StandardScaler
    feature_cols   : ordered list of feature column names the model expects
    class_labels   : list of string outcome labels ['Home Win', 'Draw', 'Away Win']
    """

    CLASS_LABELS = ["Home Win", "Draw", "Away Win"]

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        cfg = load_config(config_path)
        api_cfg = cfg["api"]

        model_path = Path(api_cfg["model_path"])
        scaler_path = Path(api_cfg["scaler_path"])
        feat_col_path = Path(cfg["models"]["output_dir"]) / "feature_columns.txt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Train the model first: python -m src.models.train"
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if feat_col_path.exists():
            with open(feat_col_path) as fh:
                self.feature_cols = [line.strip() for line in fh if line.strip()]
        else:
            # Fallback to default columns defined in train.py
            from src.models.train import FEATURE_COLS
            self.feature_cols = FEATURE_COLS

        logger.info(f"Predictor loaded: {len(self.feature_cols)} features")

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Predict match outcome probabilities.

        Parameters
        ----------
        features : dict mapping feature names to values

        Returns
        -------
        dict with keys:
          - 'home_win_prob'  : float
          - 'draw_prob'      : float
          - 'away_win_prob'  : float
          - 'predicted_outcome' : str  (most likely outcome)
          - 'probabilities'  : dict[str, float]  (all three)
        """
        x = np.array([[features.get(col, np.nan) for col in self.feature_cols]])

        # Warn if any feature is missing
        missing = [c for c in self.feature_cols if c not in features]
        if missing:
            logger.warning(f"Missing features (using NaN): {missing}")

        x_scaled = self.scaler.transform(x)
        probs = self.model.predict_proba(x_scaled)[0]

        prob_dict = {
            label: float(p) for label, p in zip(self.CLASS_LABELS, probs)
        }
        predicted = max(prob_dict, key=prob_dict.__getitem__)

        return {
            "home_win_prob": prob_dict["Home Win"],
            "draw_prob": prob_dict["Draw"],
            "away_win_prob": prob_dict["Away Win"],
            "predicted_outcome": predicted,
            "probabilities": prob_dict,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for multiple matches at once.

        Parameters
        ----------
        df : DataFrame with at least the required feature columns.

        Returns
        -------
        DataFrame with added columns: home_win_prob, draw_prob, away_win_prob,
        predicted_outcome.
        """
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].values
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        pred_idx = np.argmax(probs, axis=1)

        df = df.copy()
        df["home_win_prob"] = probs[:, 0]
        df["draw_prob"] = probs[:, 1]
        df["away_win_prob"] = probs[:, 2]
        df["predicted_outcome"] = [self.CLASS_LABELS[i] for i in pred_idx]
        return df


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict football match outcome."
    )
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--home-elo", type=float, default=1500)
    parser.add_argument("--away-elo", type=float, default=1500)
    parser.add_argument("--home-form", type=float, default=1.5)
    parser.add_argument("--away-form", type=float, default=1.5)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    predictor = Predictor(config_path=args.config)
    features = {
        "home_elo": args.home_elo,
        "away_elo": args.away_elo,
        "elo_diff": args.home_elo - args.away_elo,
        "home_form": args.home_form,
        "away_form": args.away_form,
        "form_diff": args.home_form - args.away_form,
    }
    result = predictor.predict(features)

    print(f"\nMatch: {args.home} vs {args.away}")
    print(f"  Home Win : {result['home_win_prob']:.1%}")
    print(f"  Draw     : {result['draw_prob']:.1%}")
    print(f"  Away Win : {result['away_win_prob']:.1%}")
    print(f"  Predicted: {result['predicted_outcome']}")


if __name__ == "__main__":
    main()
