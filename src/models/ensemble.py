"""
src/models/ensemble.py
-----------------------
Stacking / blending ensemble implementations.

Level-0 base models:
  - Random Forest
  - XGBoost
  - LightGBM
  - (Optionally) Neural Network / Bayesian Ridge

Level-1 meta-learner:
  - Logistic Regression trained on out-of-fold (OOF) predictions

The ensemble produces calibrated probability estimates for
{Home Win, Draw, Away Win}.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ─── Default base-model factory ───────────────────────────────────────────────

def _default_base_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        ),
        "logistic_regression_base": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
    if _HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            random_state=42,
            verbosity=-1,
        )
    return models


# ─── Stacking Ensemble ────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Two-level stacking ensemble.

    Parameters
    ----------
    base_models  : dict mapping name → unfitted sklearn estimator.
                   If None, uses the default set (RF, XGB, LGB, LR).
    meta_learner : Unfitted meta-learner estimator (default: LogisticRegression).
    n_splits     : Number of CV folds for OOF generation.
    calibrate    : If True, apply Platt calibration to the meta-learner output.
    random_state : Random seed for reproducibility.
    """

    def __init__(
        self,
        base_models: Optional[dict[str, Any]] = None,
        meta_learner: Optional[Any] = None,
        n_splits: int = 5,
        calibrate: bool = True,
        random_state: int = 42,
    ):
        self.base_models = base_models or _default_base_models()
        self.meta_learner = meta_learner or LogisticRegression(
            max_iter=1000, C=1.0, random_state=random_state
        )
        self.n_splits = n_splits
        self.calibrate = calibrate
        self.random_state = random_state

        self._fitted_base: dict[str, Any] = {}
        self._fitted_meta: Any = None
        self._n_classes: int = 3

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """
        Train the ensemble using k-fold OOF stacking.

        Parameters
        ----------
        X : Feature matrix (n_samples, n_features) – already scaled.
        y : Integer labels (0, 1, 2).
        """
        logger.info(
            f"Training stacking ensemble with {len(self.base_models)} base models "
            f"and {self.n_splits}-fold OOF."
        )

        self._n_classes = len(np.unique(y))
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        # OOF predictions matrix: (n_samples, n_base_models * n_classes)
        oof_preds = np.zeros((len(X), len(self.base_models) * self._n_classes))

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.debug(f"Fold {fold_idx + 1}/{self.n_splits}")
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            for m_idx, (name, model) in enumerate(self.base_models.items()):
                import copy
                m = copy.deepcopy(model)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_val)
                start = m_idx * self._n_classes
                oof_preds[val_idx, start: start + self._n_classes] = proba

        # Train meta-learner on full OOF
        if self.calibrate:
            meta = CalibratedClassifierCV(self.meta_learner, cv=3, method="sigmoid")
        else:
            meta = self.meta_learner

        meta.fit(oof_preds, y)
        self._fitted_meta = meta

        # Refit base models on full training data
        for name, model in self.base_models.items():
            import copy
            m = copy.deepcopy(model)
            m.fit(X, y)
            self._fitted_base[name] = m
            logger.info(f"  Base model '{name}' trained.")

        train_meta_input = self._base_predict_proba(X)
        train_log_loss = log_loss(y, meta.predict_proba(train_meta_input))
        logger.success(f"Ensemble training complete. Train log-loss: {train_log_loss:.4f}")

        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability estimates (n_samples, n_classes)."""
        if not self._fitted_base:
            raise RuntimeError("Ensemble has not been fitted yet. Call .fit() first.")
        meta_input = self._base_predict_proba(X)
        return self._fitted_meta.predict_proba(meta_input)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (argmax of probabilities)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def _base_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Concatenate base-model probability outputs for meta-learner input."""
        parts = []
        for name, model in self._fitted_base.items():
            parts.append(model.predict_proba(X))
        return np.hstack(parts)

    # ── Convenience ───────────────────────────────────────────────────────────

    def base_model_names(self) -> list[str]:
        return list(self.base_models.keys())
