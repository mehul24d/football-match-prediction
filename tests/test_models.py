"""
tests/test_models.py
--------------------
Unit tests for model training, evaluation, and prediction utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    brier_score_multiclass,
    calibration_data,
    compare_models,
    compute_metrics,
)
from src.models.train import (
    FEATURE_COLS,
    brier_score_multiclass as train_brier,
    prepare_data,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_feature_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic feature DataFrame mimicking the output of build_features."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "home_elo": rng.uniform(1400, 1700, n),
        "away_elo": rng.uniform(1400, 1700, n),
        "home_form": rng.uniform(0, 3, n),
        "away_form": rng.uniform(0, 3, n),
        "home_form_decayed": rng.uniform(0, 3, n),
        "away_form_decayed": rng.uniform(0, 3, n),
        "home_goals_scored_avg": rng.uniform(0.5, 3.0, n),
        "home_goals_conceded_avg": rng.uniform(0.5, 2.5, n),
        "away_goals_scored_avg": rng.uniform(0.5, 3.0, n),
        "away_goals_conceded_avg": rng.uniform(0.5, 2.5, n),
        "home_shots_avg": rng.uniform(5, 20, n),
        "away_shots_avg": rng.uniform(5, 20, n),
        "home_shots_on_target_avg": rng.uniform(1, 8, n),
        "away_shots_on_target_avg": rng.uniform(1, 8, n),
        "home_corners_avg": rng.uniform(2, 8, n),
        "away_corners_avg": rng.uniform(2, 8, n),
        "result_label": rng.integers(0, 3, n),
    })
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["form_diff"] = df["home_form"] - df["away_form"]
    return df


# ─── brier_score_multiclass ──────────────────────────────────────────────────

class TestBrierScore:
    def test_perfect_prediction_is_zero(self):
        y_true = np.array([0, 1, 2])
        y_prob = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        score = brier_score_multiclass(y_true, y_prob)
        assert score == pytest.approx(0.0)

    def test_uniform_prediction(self):
        """Uniform 1/3 probabilities should give a fixed Brier score."""
        y_true = np.array([0, 1, 2] * 10)
        y_prob = np.full((30, 3), 1.0 / 3)
        score = brier_score_multiclass(y_true, y_prob)
        # For each sample: sum_k (1/3 - o_k)^2 = (1/3-1)^2 + 2*(1/3)^2 = 4/9+2/9 = 6/9
        # Mean across 30 samples = 6/9 ≈ 0.667
        assert score == pytest.approx(2.0 / 3, abs=1e-9)

    def test_worse_than_perfect(self):
        y_true = np.array([0, 0, 0])
        y_prob_good = np.array([[0.9, 0.05, 0.05]] * 3)
        y_prob_bad = np.array([[0.1, 0.45, 0.45]] * 3)
        assert brier_score_multiclass(y_true, y_prob_good) < brier_score_multiclass(
            y_true, y_prob_bad
        )

    def test_same_result_in_train_and_evaluate_modules(self):
        y_true = np.array([0, 1, 2, 0])
        y_prob = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],
        ])
        assert train_brier(y_true, y_prob) == pytest.approx(
            brier_score_multiclass(y_true, y_prob)
        )


# ─── compute_metrics ─────────────────────────────────────────────────────────

class TestComputeMetrics:
    def _sample(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 2, 1, 0])
        y_prob = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.4, 0.3, 0.3],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4],
        ])
        return y_true, y_pred, y_prob

    def test_returns_expected_keys(self):
        y_true, y_pred, y_prob = self._sample()
        metrics = compute_metrics(y_true, y_pred, y_prob)
        for key in ["accuracy", "log_loss", "brier_score", "classification_report"]:
            assert key in metrics

    def test_accuracy_range(self):
        y_true, y_pred, y_prob = self._sample()
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_log_loss_positive(self):
        y_true, y_pred, y_prob = self._sample()
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["log_loss"] > 0

    def test_confusion_matrix_shape(self):
        y_true, y_pred, y_prob = self._sample()
        metrics = compute_metrics(y_true, y_pred, y_prob)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (3, 3)


# ─── calibration_data ────────────────────────────────────────────────────────

class TestCalibrationData:
    def test_returns_arrays(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 3, 100)
        y_prob = rng.dirichlet(np.ones(3), 100)
        prob_pred, prob_true = calibration_data(y_true, y_prob, class_idx=0)
        assert len(prob_pred) == len(prob_true)
        assert all(0 <= p <= 1 for p in prob_pred)
        assert all(0 <= p <= 1 for p in prob_true)


# ─── compare_models ───────────────────────────────────────────────────────────

class TestCompareModels:
    def test_returns_dataframe(self):
        results = {
            "lr": {"accuracy": 0.50, "log_loss": 1.05, "brier_score": 0.65, "cv_accuracy_mean": 0.48},
            "rf": {"accuracy": 0.55, "log_loss": 0.98, "brier_score": 0.60, "cv_accuracy_mean": 0.53},
        }
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["model", "accuracy", "log_loss", "brier_score", "cv_accuracy"]

    def test_sorted_by_log_loss(self):
        results = {
            "lr": {"accuracy": 0.50, "log_loss": 1.05, "brier_score": 0.65, "cv_accuracy_mean": 0.48},
            "rf": {"accuracy": 0.55, "log_loss": 0.98, "brier_score": 0.60, "cv_accuracy_mean": 0.53},
        }
        df = compare_models(results)
        assert df.iloc[0]["model"] == "rf"


# ─── prepare_data ────────────────────────────────────────────────────────────

class TestPrepareData:
    def test_returns_correct_shapes(self):
        df = _make_feature_df(200)
        X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_data(
            df, test_size=0.2, random_seed=0
        )
        n_total = len(df.dropna(subset=feat_cols + ["result_label"]))
        assert len(X_train) + len(X_test) == n_total
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_scaler_applied(self):
        df = _make_feature_df(200)
        X_train, X_test, y_train, y_test, scaler, _ = prepare_data(df, random_seed=0)
        # Scaled training data should have near-zero mean
        assert abs(X_train.mean()) < 0.5

    def test_raises_if_no_features(self):
        df = pd.DataFrame({"result_label": [0, 1, 2]})
        with pytest.raises(ValueError):
            prepare_data(df, feature_cols=["nonexistent_col"])

    def test_labels_are_integers(self):
        df = _make_feature_df(100)
        _, _, y_train, y_test, _, _ = prepare_data(df, random_seed=0)
        assert y_train.dtype in [np.int32, np.int64]
        assert set(np.unique(np.concatenate([y_train, y_test]))).issubset({0, 1, 2})


# ─── End-to-end: train and predict (fast, no file I/O) ───────────────────────

class TestEndToEndTrainPredict:
    """Smoke test: train a logistic regression on synthetic data and predict."""

    def test_logreg_trains_and_predicts(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        df = _make_feature_df(300, seed=42)
        X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_data(
            df, test_size=0.2, random_seed=42
        )

        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        assert len(y_pred) == len(y_test)
        assert y_prob.shape == (len(y_test), 3)
        assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-6)
