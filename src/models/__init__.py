"""Model training, evaluation and prediction sub-package."""

from src.models.train import FEATURE_COLS, brier_score_multiclass, prepare_data
from src.models.ensemble import StackingEnsemble
from src.models.bayesian import BayesianMatchPredictor
from src.models.neural_network import FootballMLP
from src.models.sequential import SimpleRNNClassifier

__all__ = [
    "FEATURE_COLS",
    "brier_score_multiclass",
    "prepare_data",
    "StackingEnsemble",
    "BayesianMatchPredictor",
    "FootballMLP",
    "SimpleRNNClassifier",
]
