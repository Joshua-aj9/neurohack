"""Decoder interface: scores(window) -> raw scores; predict_proba(window) -> calibrated probabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Decoder(ABC):
    temperature: float = 1.0

    @abstractmethod
    def scores(self, window: np.ndarray) -> np.ndarray:
        """window [n_ch, n_samples] -> scores [n_targets] (higher = more likely)."""

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        return softmax(self.scores(window) / self.temperature)

    def predict(self, window: np.ndarray) -> int:
        return int(np.argmax(self.scores(window)))
