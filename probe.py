"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` to match the competition interface, while using
    a regularized linear sklearn probe internally.  With only hundreds of
    labels and high-dimensional hidden-state features, this is less brittle
    than a small MLP trained directly on every feature.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._reducer: TruncatedSVD | None = None
        self._classifier: LogisticRegression | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sklearn-probe logits for a tensor input.

        The evaluation code uses ``predict`` and ``predict_proba`` directly,
        but keeping ``forward`` makes the ``nn.Module`` wrapper coherent.
        """
        probs = self.predict_proba(x.detach().cpu().numpy())[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
        return torch.as_tensor(logits, dtype=x.dtype, device=x.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features, compresses high-dimensional hidden-state pools with
        SVD, appends the scalar aggregation features, and fits one regularized
        logistic regression probe.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        torch.manual_seed(42)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)

        X_scaled = self._scaler.fit_transform(X)
        X_model = self._fit_transform_features(X_scaled)

        self._classifier = LogisticRegression(
            C=0.05,
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
            solver="liblinear",
        )
        self._classifier.fit(X_model, y)

        train_probs = self._positive_probability(X_model)
        self._threshold = self._best_threshold(y, train_probs)
        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")

        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=int)
        X_scaled = self._scaler.transform(X_val)
        X_model = self._transform_features(X_scaled)
        probs = self._positive_probability(X_model)
        self._threshold = self._best_threshold(y_val, probs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X)
        X_model = self._transform_features(X_scaled)
        prob_pos = self._positive_probability(X_model)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def _fit_transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        n_samples, n_features = X_scaled.shape
        if n_features <= 256 or n_samples <= 40:
            self._reducer = None
            return X_scaled

        n_components = min(64, n_features, n_samples - 1)
        self._reducer = TruncatedSVD(
            n_components=n_components,
            random_state=42,
            n_iter=5,
        )
        X_reduced = self._reducer.fit_transform(X_scaled)
        return np.hstack([X_reduced, self._scalar_features(X_scaled)])

    def _transform_features(self, X_scaled: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            return X_scaled

        X_reduced = self._reducer.transform(X_scaled)
        return np.hstack([X_reduced, self._scalar_features(X_scaled)])

    def _positive_probability(self, X_model: np.ndarray) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")

        probabilities = self._classifier.predict_proba(X_model)
        positive_col = int(np.where(self._classifier.classes_ == 1)[0][0])
        return probabilities[:, positive_col]

    @staticmethod
    def _scalar_features(X_scaled: np.ndarray) -> np.ndarray:
        # aggregation.aggregate appends six scalar features after the hidden
        # state pools.  Keeping them outside the SVD preserves cheap signals
        # such as length and activation scale.
        if X_scaled.shape[1] < 6:
            return X_scaled[:, :0]
        return X_scaled[:, -6:]

    @staticmethod
    def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
        candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 181)]))

        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (probs >= threshold).astype(int)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if acc > best_accuracy or (acc == best_accuracy and f1 > best_f1):
                best_accuracy = acc
                best_f1 = f1
                best_threshold = float(threshold)

        return best_threshold
