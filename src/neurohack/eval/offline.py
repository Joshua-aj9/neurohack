"""Offline evaluation helpers on epoched data."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from neurohack.decoders.base import Decoder, softmax


def scores_by_length(
    decoder: Decoder, epochs: np.ndarray, lengths: list[float], srate: float, offset_s: float = 0.0
) -> np.ndarray:
    """epochs [n_trials, n_ch, n] (t=0 at stimulus onset) -> scores [n_trials, n_lens, n_targets].

    Window for length L = [offset_s, offset_s + L).
    """
    off = int(round(offset_s * srate))
    out = []
    for ep in epochs:
        row = []
        for L in lengths:
            n = int(round(L * srate))
            if off + n > ep.shape[1]:
                raise ValueError(f"epoch too short for window {L}s + offset {offset_s}s")
            row.append(decoder.scores(ep[:, off : off + n]))
        out.append(row)
    return np.asarray(out)


def probs_from_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(scores / temperature, axis=-1)


def accuracy_by_length(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """scores [n_trials, n_lens, n_targets] -> accuracy [n_lens]."""
    return (scores.argmax(axis=2) == labels[:, None]).mean(axis=0)


def binomial_p(n_correct: int, n_trials: int, chance: float) -> float:
    """One-sided p-value that accuracy exceeds chance."""
    return float(binomtest(n_correct, n_trials, chance, alternative="greater").pvalue)


def confusion(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=int)
    for p, y in zip(pred, labels, strict=True):
        if p >= 0:
            m[y, p] += 1
    return m
