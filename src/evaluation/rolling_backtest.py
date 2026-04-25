"""
src/evaluation/rolling_backtest.py
------------------------------------
Rolling-window (walk-forward) backtesting framework.

Simulates real-world deployment: the model is trained on all data up to a
cutoff date, evaluated on the next window, then the window slides forward.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss

from src.evaluation.calibration import brier_score_multiclass


# ─── Rolling Backtest ─────────────────────────────────────────────────────────

class RollingBacktest:
    """
    Walk-forward backtest with a fixed or expanding training window.

    Parameters
    ----------
    model_factory    : Callable that returns a fresh unfitted estimator each fold.
    initial_train_size : Number of samples (or fraction) for the first training set.
    test_size        : Number of samples per evaluation window.
    step_size        : How many samples to advance per fold.
                       Defaults to ``test_size`` (non-overlapping windows).
    expanding        : If True, grow the training set each fold (expanding window).
                       If False, use a fixed-size sliding window.
    scaler_factory   : Optional callable returning a fresh scaler each fold.
    """

    def __init__(
        self,
        model_factory: Callable,
        initial_train_size: int | float = 0.7,
        test_size: int = 50,
        step_size: Optional[int] = None,
        expanding: bool = True,
        scaler_factory: Optional[Callable] = None,
    ):
        self.model_factory = model_factory
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.expanding = expanding
        self.scaler_factory = scaler_factory

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """
        Execute the rolling backtest.

        Parameters
        ----------
        X : Feature matrix (n_samples, n_features) – sorted chronologically.
        y : Labels (n_samples,).

        Returns
        -------
        Dictionary with per-fold and aggregate metrics.
        """
        n = len(X)

        # Resolve initial training size
        if isinstance(self.initial_train_size, float):
            init_train = int(n * self.initial_train_size)
        else:
            init_train = int(self.initial_train_size)

        if init_train < 10:
            raise ValueError("initial_train_size results in < 10 training samples.")

        fold_results = []
        all_y_true, all_y_pred, all_y_prob = [], [], []

        fold = 0
        train_start = 0
        train_end = init_train

        while train_end + self.test_size <= n:
            test_start = train_end
            test_end = min(train_end + self.test_size, n)

            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            # Optional scaling
            if self.scaler_factory is not None:
                scaler = self.scaler_factory()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model = self.model_factory()
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)
            y_pred = np.argmax(y_prob, axis=1)

            fold_acc = float(accuracy_score(y_test, y_pred))
            fold_ll = float(log_loss(y_test, y_prob))
            fold_bs = float(brier_score_multiclass(y_test, y_prob))

            fold_results.append({
                "fold": fold,
                "train_size": train_end - train_start,
                "test_start": test_start,
                "test_end": test_end,
                "accuracy": fold_acc,
                "log_loss": fold_ll,
                "brier_score": fold_bs,
            })

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())

            logger.debug(
                f"Fold {fold}: train={train_end - train_start}  "
                f"test={test_end - test_start}  "
                f"acc={fold_acc:.3f}  ll={fold_ll:.4f}  bs={fold_bs:.4f}"
            )

            fold += 1
            if self.expanding:
                train_end += self.step_size
            else:
                train_start += self.step_size
                train_end += self.step_size

        if not fold_results:
            raise ValueError(
                "No folds completed. Increase the dataset size or reduce test_size."
            )

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.array(all_y_prob)

        aggregate = {
            "n_folds": len(fold_results),
            "avg_accuracy": float(np.mean([f["accuracy"] for f in fold_results])),
            "avg_log_loss": float(np.mean([f["log_loss"] for f in fold_results])),
            "avg_brier_score": float(np.mean([f["brier_score"] for f in fold_results])),
            "overall_accuracy": float(accuracy_score(all_y_true, all_y_pred)),
            "overall_log_loss": float(log_loss(all_y_true, all_y_prob)),
            "overall_brier_score": float(brier_score_multiclass(all_y_true, all_y_prob)),
            "fold_details": fold_results,
            "all_y_true": all_y_true,
            "all_y_pred": all_y_pred,
            "all_y_prob": all_y_prob,
        }

        logger.success(
            f"Rolling backtest complete – {len(fold_results)} folds  "
            f"avg_acc={aggregate['avg_accuracy']:.3f}  "
            f"avg_brier={aggregate['avg_brier_score']:.4f}"
        )
        return aggregate
