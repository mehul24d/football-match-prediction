"""
src/models/ensemble.py
-----------------------
Clean & efficient stacking ensemble (NO leakage, ROBUST).
"""

import numpy as np
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV


class StackingEnsemble:
    def __init__(self, base_models, n_splits=5, random_state=42, calibrate=True):
        self.base_models = base_models
        self.meta_model = LogisticRegression(max_iter=1000)
        self.n_splits = n_splits
        self.random_state = random_state
        self.calibrate = calibrate

        self.fitted_base_models = []
        self.fitted_meta_model = None
        self.n_classes_ = None

    # ─────────────────────────────────────────────
    # Training (OOF stacking)
    # ─────────────────────────────────────────────
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))

        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        meta_features = np.zeros(
            (X.shape[0], len(self.base_models) * self.n_classes_)
        )

        # Generate OOF predictions
        for i, model in enumerate(self.base_models):
            oof_preds = np.zeros((X.shape[0], self.n_classes_))

            for train_idx, val_idx in kf.split(X, y):
                m = copy.deepcopy(model)
                m.fit(X[train_idx], y[train_idx])

                probs = m.predict_proba(X[val_idx])

                # Handle class mismatch (VERY IMPORTANT)
                aligned_probs = np.zeros((len(val_idx), self.n_classes_))
                aligned_probs[:, :probs.shape[1]] = probs

                oof_preds[val_idx] = aligned_probs

            meta_features[:, i*self.n_classes_:(i+1)*self.n_classes_] = oof_preds

            # Train final model on full data
            final_model = copy.deepcopy(model)
            final_model.fit(X, y)
            self.fitted_base_models.append(final_model)

        # ─────────────────────────────────────────
        # Meta model (with optional calibration)
        # ─────────────────────────────────────────
        if self.calibrate:
            meta = CalibratedClassifierCV(self.meta_model, cv=3, method="sigmoid")
        else:
            meta = self.meta_model

        meta.fit(meta_features, y)
        self.fitted_meta_model = meta

    # ─────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────
    def predict_proba(self, X):
        if not self.fitted_base_models:
            raise RuntimeError("Model not fitted. Call fit() first.")

        meta_features = []

        for model in self.fitted_base_models:
            probs = model.predict_proba(X)

            aligned_probs = np.zeros((X.shape[0], self.n_classes_))
            aligned_probs[:, :probs.shape[1]] = probs

            meta_features.append(aligned_probs)

        meta_X = np.hstack(meta_features)
        return self.fitted_meta_model.predict_proba(meta_X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)