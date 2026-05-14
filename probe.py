"""
probe.py — Hallucination probe: multi-seed CatBoost ensemble.

Trains 5 CatBoost models with different random seeds and averages
their predictions to reduce variance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 123, 456, 789, 2024]


class HallucinationProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))
        self._scaler = StandardScaler()
        self._models: list[CatBoostClassifier] = []
        self._threshold: float = 0.5

    @staticmethod
    def _make_model(seed: int = 42) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=500, depth=4, learning_rate=0.03,
            l2_leaf_reg=10, auto_class_weights="Balanced",
            rsm=0.3, random_seed=seed, verbose=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Use predict/predict_proba instead of forward()")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)
        self._models = []
        for seed in SEEDS:
            m = self._make_model(seed=seed)
            m.fit(X_scaled, y)
            self._models.append(m)
        self._tune_threshold_internal(X_scaled, y)
        return self

    def _tune_threshold_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2:
            return
        all_probs = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for tr_idx, va_idx in skf.split(X, y):
            m = self._make_model(seed=42)
            m.fit(X[tr_idx], y[tr_idx])
            all_probs[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        best_threshold, best_f1 = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 61):
            score = f1_score(y, (all_probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)
        self._threshold = best_threshold

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray,
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)
        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        probs = np.mean(
            [m.predict_proba(X_scaled) for m in self._models], axis=0,
        )
        return probs
