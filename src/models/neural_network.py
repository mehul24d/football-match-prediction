"""
src/models/neural_network.py
-----------------------------
Feedforward neural network for football match outcome prediction.

Uses scikit-learn's MLPClassifier as the primary implementation to avoid
hard dependencies on TensorFlow/PyTorch.  A thin TensorFlow wrapper is
provided when TensorFlow is available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ─── scikit-learn based MLP ───────────────────────────────────────────────────

class FootballMLP:
    """
    Fully-connected feedforward neural network for match prediction.

    Wraps scikit-learn's MLPClassifier with sensible defaults for the
    3-class (Home Win / Draw / Away Win) classification task.

    Parameters
    ----------
    hidden_layers  : Tuple of hidden layer sizes (default: 3 layers × 128 units).
    dropout        : Not natively supported in sklearn MLP; use alpha for L2 reg.
    alpha          : L2 regularisation strength.
    learning_rate  : Initial learning rate.
    max_iter       : Maximum training epochs.
    random_state   : Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 128, 64),
        alpha: float = 1e-4,
        learning_rate: float = 1e-3,
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

        self._model: Optional[MLPClassifier] = None
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FootballMLP":
        logger.info(
            f"Training MLP: layers={self.hidden_layers}, "
            f"lr={self.learning_rate}, max_iter={self.max_iter}"
        )
        X_scaled = self._scaler.fit_transform(X)
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        self._model.fit(X_scaled, y)
        logger.success(
            f"MLP training complete. Best val loss: {self._model.best_loss_:.4f}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
        return self._model.predict_proba(self._scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ─── TensorFlow-based MLP (optional) ─────────────────────────────────────────

def build_tf_model(
    input_dim: int,
    n_classes: int = 3,
    hidden_units: tuple[int, ...] = (128, 128, 64),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
):
    """
    Build a Keras/TensorFlow feedforward model.

    Returns None if TensorFlow is not installed.

    Parameters
    ----------
    input_dim    : Number of input features.
    n_classes    : Number of output classes (3 for football outcomes).
    hidden_units : Sizes of hidden layers.
    dropout_rate : Dropout fraction after each hidden layer.
    learning_rate: Adam learning rate.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        logger.warning("TensorFlow not installed. build_tf_model returns None.")
        return None

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for units in hidden_units:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
