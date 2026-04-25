"""
src/models/bayesian.py
-----------------------
Bayesian Ridge regression wrapper for football match outcome prediction.

Provides uncertainty quantification: in addition to class probabilities the
model reports posterior standard deviations, giving calibrated confidence
intervals on predictions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from sklearn.linear_model import BayesianRidge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize


class BayesianMatchPredictor:
    """
    Bayesian Ridge regression for 3-class football outcome prediction.

    Uses a One-vs-Rest strategy with BayesianRidge on binarised labels.
    Returns both class probabilities and uncertainty estimates.

    Parameters
    ----------
    alpha_1, alpha_2 : Shape / rate hyperparameters for weight variance prior.
    lambda_1, lambda_2: Shape / rate hyperparameters for noise variance prior.
    max_iter         : Maximum EM iterations for hyperparameter optimisation.
    """

    CLASSES = [0, 1, 2]  # Home Win, Draw, Away Win

    def __init__(
        self,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        max_iter: int = 300,
    ):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iter = max_iter

        self._models: list[BayesianRidge] = []
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianMatchPredictor":
        logger.info("Training Bayesian Ridge predictor (OvR)…")
        X_sc = self._scaler.fit_transform(X)
        Y = label_binarize(y, classes=self.CLASSES)  # (n, 3)

        self._models = []
        for c in range(len(self.CLASSES)):
            m = BayesianRidge(
                alpha_1=self.alpha_1,
                alpha_2=self.alpha_2,
                lambda_1=self.lambda_1,
                lambda_2=self.lambda_2,
                max_iter=self.max_iter,
                compute_score=True,
            )
            m.fit(X_sc, Y[:, c].astype(float))
            self._models.append(m)

        self._fitted = True
        logger.success("Bayesian Ridge predictor trained.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax-normalised class probabilities."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X_sc = self._scaler.transform(X)
        raw = np.column_stack([m.predict(X_sc) for m in self._models])
        # Softmax to convert regression outputs to probabilities
        exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
        return exp_raw / exp_raw.sum(axis=1, keepdims=True)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (probabilities, uncertainty) where uncertainty is the mean
        posterior standard deviation across classes.

        Returns
        -------
        probs : (n_samples, 3)
        std   : (n_samples,)  – average posterior std across OvR models
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X_sc = self._scaler.transform(X)
        raw_preds, raw_stds = [], []

        for m in self._models:
            pred, std = m.predict(X_sc, return_std=True)
            raw_preds.append(pred)
            raw_stds.append(std)

        raw = np.column_stack(raw_preds)
        stds = np.column_stack(raw_stds).mean(axis=1)

        exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)

        return probs, stds

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
