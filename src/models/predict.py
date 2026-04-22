"""
src/models/predict.py
---------------------
Load trained model and generate match predictions (robust production version).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import load_config


class Predictor:
    """Production-ready prediction wrapper with safe feature handling"""

    CLASS_LABELS = ["Home Win", "Draw", "Away Win"]

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        cfg = load_config(config_path)

        model_path = Path(cfg["api"]["model_path"])
        scaler_path = Path(cfg["api"]["scaler_path"])
        feat_path = Path(cfg["models"]["output_dir"]) / "feature_columns.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if feat_path.exists():
            with open(feat_path) as f:
                self.feature_cols = [line.strip() for line in f if line.strip()]
        else:
            from src.models.train import FEATURE_COLS
            self.feature_cols = FEATURE_COLS

        logger.success(f"Loaded model with {len(self.feature_cols)} features")

    # ─────────────────────────────────────────────
    # Internal helper
    # ─────────────────────────────────────────────

    def _prepare_dataframe(self, features: dict[str, float]) -> pd.DataFrame:
        """Convert input dict → complete feature DataFrame"""

        df = pd.DataFrame([features])

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing features detected → filling defaults: {missing_cols}")

        # Fill missing with safe defaults
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Ensure correct order
        df = df[self.feature_cols]

        return df

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Predict single match outcome"""

        df = self._prepare_dataframe(features)

        try:
            X_scaled = self.scaler.transform(df.values)
        except Exception as exc:
            logger.error(f"Scaling failed: {exc}")
            raise ValueError(f"Scaling failed: {exc}")

        try:
            probs = self.model.predict_proba(X_scaled)[0]
        except Exception as exc:
            logger.error(f"Prediction failed: {exc}")
            raise ValueError(f"Prediction failed: {exc}")

        prob_dict = dict(zip(self.CLASS_LABELS, probs))
        predicted = self.CLASS_LABELS[int(np.argmax(probs))]

        return {
            "home_win_prob": float(prob_dict["Home Win"]),
            "draw_prob": float(prob_dict["Draw"]),
            "away_win_prob": float(prob_dict["Away Win"]),
            "predicted_outcome": predicted,
            "probabilities": {k: float(v) for k, v in prob_dict.items()},
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict multiple matches safely"""

        df = df.copy()

        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing batch features → filling defaults: {missing_cols}")

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        df_model = df[self.feature_cols]

        X_scaled = self.scaler.transform(df_model.values)
        probs = self.model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)

        df["home_win_prob"] = probs[:, 0]
        df["draw_prob"] = probs[:, 1]
        df["away_win_prob"] = probs[:, 2]
        df["predicted_outcome"] = [
            self.CLASS_LABELS[i] for i in preds
        ]

        return df


# ─────────────────────────────────────────────
# CLI (for testing)
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict football match outcome")

    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)

    parser.add_argument("--home-elo", type=float, default=1500)
    parser.add_argument("--away-elo", type=float, default=1500)

    parser.add_argument("--home-form", type=float, default=1.5)
    parser.add_argument("--away-form", type=float, default=1.5)

    parser.add_argument("--config", default="configs/config.yaml")

    args = parser.parse_args()

    predictor = Predictor(args.config)

    # Minimal features (API-like input)
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
    print(f"Home Win : {result['home_win_prob']:.2%}")
    print(f"Draw     : {result['draw_prob']:.2%}")
    print(f"Away Win : {result['away_win_prob']:.2%}")
    print(f"Prediction: {result['predicted_outcome']}")


if __name__ == "__main__":
    main()