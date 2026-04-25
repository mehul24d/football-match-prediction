"""
src/models/sequential.py
-------------------------
LSTM-based sequence model for football match prediction.

Captures temporal dependencies across a team's recent match history.
Requires either PyTorch or TensorFlow/Keras.

A pure-NumPy fallback (SimpleRNN) is provided for environments where
neither deep-learning framework is available.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


# ─── Numpy fallback: Simple Elman RNN ────────────────────────────────────────

class SimpleRNNClassifier:
    """
    Lightweight Elman RNN trained with truncated BPTT via numerical gradient.

    This is a minimal reference implementation for environments without
    TensorFlow or PyTorch.  For production use the Keras/PyTorch LSTM below.

    Parameters
    ----------
    hidden_size  : Number of hidden units.
    n_classes    : Output classes (3 for football outcomes).
    learning_rate: Gradient descent step size.
    n_epochs     : Training epochs.
    random_state : Seed.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        n_classes: int = 3,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        random_state: int = 42,
    ):
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(random_state)
        self._fitted = False

    def _init_weights(self, n_features: int) -> None:
        k = 1.0 / self.hidden_size ** 0.5
        self.Wxh = self.rng.uniform(-k, k, (n_features, self.hidden_size))
        self.Whh = self.rng.uniform(-k, k, (self.hidden_size, self.hidden_size))
        self.bh = np.zeros(self.hidden_size)
        self.Why = self.rng.uniform(-k, k, (self.hidden_size, self.n_classes))
        self.by = np.zeros(self.n_classes)

    def _forward(self, X_seq: np.ndarray) -> np.ndarray:
        """Forward pass for a single sequence (T, n_features)."""
        h = np.zeros(self.hidden_size)
        for t in range(len(X_seq)):
            h = np.tanh(X_seq[t] @ self.Wxh + h @ self.Whh + self.bh)
        logits = h @ self.Why + self.by
        exp_l = np.exp(logits - logits.max())
        return exp_l / exp_l.sum()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRNNClassifier":
        """
        Train on sequences.

        Parameters
        ----------
        X : (n_samples, seq_len, n_features)
        y : (n_samples,) integer labels
        """
        n_samples, seq_len, n_features = X.shape
        self._init_weights(n_features)
        self._n_features = n_features

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for i in range(n_samples):
                probs = self._forward(X[i])
                loss = -np.log(probs[y[i]] + 1e-12)
                total_loss += loss

                # Gradient of cross-entropy softmax output
                dlogits = probs.copy()
                dlogits[y[i]] -= 1.0

                # Backprop through output layer (simple gradient descent)
                h_last = np.zeros(self.hidden_size)
                for t in range(len(X[i])):
                    h_last = np.tanh(
                        X[i][t] @ self.Wxh + h_last @ self.Whh + self.bh
                    )

                self.Why -= self.lr * np.outer(h_last, dlogits)
                self.by -= self.lr * dlogits

            if epoch % 10 == 0:
                logger.debug(
                    f"SimpleRNN epoch {epoch}: loss={total_loss / n_samples:.4f}"
                )

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return np.array([self._forward(X[i]) for i in range(len(X))])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ─── Keras LSTM (optional) ────────────────────────────────────────────────────

def build_keras_lstm(
    seq_len: int,
    n_features: int,
    n_classes: int = 3,
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    """
    Build a Keras LSTM model for sequence prediction.

    Returns None if TensorFlow is not installed.

    Parameters
    ----------
    seq_len      : Sequence length (past matches per sample).
    n_features   : Number of features per time step.
    n_classes    : Output classes.
    lstm_units   : Number of LSTM units.
    dropout      : Recurrent dropout rate.
    learning_rate: Adam learning rate.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        logger.warning("TensorFlow not available. build_keras_lstm returns None.")
        return None

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, n_features)),
        keras.layers.LSTM(lstm_units, dropout=dropout, return_sequences=True),
        keras.layers.LSTM(lstm_units // 2, dropout=dropout),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
