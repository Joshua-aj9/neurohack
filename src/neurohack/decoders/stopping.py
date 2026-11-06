"""Confidence-gated dynamic stopping with quality-based abstention.

Online: `StoppingPolicy.step()` is called as the window grows (min, min+step, ..., max).
Offline: `simulate()` / `fit_theta()` operate on precomputed probabilities per window length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurohack.config import frange
from neurohack.eval.itr import wolpaw_bits

COMMIT, ABSTAIN_QUALITY, TIMEOUT = "commit", "abstain_quality", "timeout"


@dataclass(frozen=True)
class Decision:
    kind: str  # commit | abstain_quality | timeout
    target: int | None  # None if no selection made
    p_max: float
    window_s: float
    quality: float


@dataclass
class StoppingPolicy:
    theta: float
    min_window_s: float = 1.0
    step_s: float = 0.25
    max_window_s: float = 4.0
    quality_min: float = 0.5
    on_timeout: str = "abstain"

    @classmethod
    def from_config(cls, stop_cfg: dict) -> StoppingPolicy:
        return cls(
            stop_cfg["theta"],
            stop_cfg["min_window_s"],
            stop_cfg["step_s"],
            stop_cfg["max_window_s"],
            stop_cfg["quality_min"],
            stop_cfg["on_timeout"],
        )

    @property
    def lengths(self) -> list[float]:
        return frange(self.min_window_s, self.max_window_s, self.step_s)

    def step(self, window_s: float, probs: np.ndarray, quality: float) -> Decision | None:
        """Return a Decision to end the trial, or None to keep growing the window."""
        j, p = int(np.argmax(probs)), float(np.max(probs))
        if quality < self.quality_min:
            return Decision(ABSTAIN_QUALITY, None, p, window_s, quality)
        if p >= self.theta:
            return Decision(COMMIT, j, p, window_s, quality)
        if window_s >= self.max_window_s - 1e-9:
            target = j if self.on_timeout == "argmax" else None
            return Decision(TIMEOUT, target, p, window_s, quality)
        return None


def simulate(
    probs_by_len: np.ndarray,
    labels: np.ndarray,
    theta: float,
    lengths: list[float],
    on_timeout: str = "abstain",
) -> tuple[np.ndarray, np.ndarray]:
    """probs_by_len [n_trials, n_lens, n_targets] -> (pred [-1 = no selection], decision time s)."""
    n = probs_by_len.shape[0]
    pred = np.full(n, -1)
    tdec = np.full(n, lengths[-1])
    pmax = probs_by_len.max(axis=2)
    for i in range(n):
        hit = np.nonzero(pmax[i] >= theta)[0]
        if hit.size:
            pred[i] = int(np.argmax(probs_by_len[i, hit[0]]))
            tdec[i] = lengths[hit[0]]
        elif on_timeout == "argmax":
            pred[i] = int(np.argmax(probs_by_len[i, -1]))
    return pred, tdec


def summarize(
    pred: np.ndarray, tdec: np.ndarray, labels: np.ndarray, n_targets: int, gap_s: float
) -> dict:
    committed = pred >= 0
    n_c = int(committed.sum())
    acc = float((pred[committed] == labels[committed]).mean()) if n_c else float("nan")
    mean_t = float(tdec.mean())
    # ITR over all trials: every trial costs time; only commits deliver bits
    total_min = float((tdec + gap_s).sum()) / 60.0
    bits = wolpaw_bits(n_targets, acc) if n_c else 0.0  # bits per committed selection
    itr = bits * n_c / total_min if total_min > 0 else 0.0
    return {
        "accuracy_committed": acc,
        "commit_rate": n_c / len(pred),
        "mean_decision_s": mean_t,
        "itr_bits_min": itr,
        "n_committed": n_c,
    }


def fit_theta(
    probs_by_len: np.ndarray,
    labels: np.ndarray,
    lengths: list[float],
    target_accuracy: float,
    grid: tuple[float, float, float],
    n_targets: int,
    gap_s: float = 1.0,
    on_timeout: str = "abstain",
    min_commit_rate: float = 0.2,
) -> tuple[float, list[dict]]:
    """Smallest theta whose committed accuracy >= target (fastest such policy).

    If no theta reaches the target, returns the theta with the highest committed accuracy among those
    meeting `min_commit_rate`. Returns (theta, table of per-theta summaries).
    """
    table = []
    for th in frange(*grid):
        pred, tdec = simulate(probs_by_len, labels, th, lengths, on_timeout)
        row = {"theta": th, **summarize(pred, tdec, labels, n_targets, gap_s)}
        table.append(row)
    ok = [
        r
        for r in table
        if r["commit_rate"] >= min_commit_rate and r["accuracy_committed"] >= target_accuracy
    ]
    if ok:
        return ok[0]["theta"], table
    valid = [r for r in table if r["commit_rate"] >= min_commit_rate] or table
    best = max(valid, key=lambda r: (np.nan_to_num(r["accuracy_committed"]), -r["theta"]))
    return best["theta"], table
