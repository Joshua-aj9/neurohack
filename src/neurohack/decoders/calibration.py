"""Temperature scaling (Guo et al. 2017) and expected calibration error."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from neurohack.decoders.base import softmax


def nll(scores: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    p = softmax(scores / temperature)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0))))


def fit_temperature(
    scores: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (1e-4, 1e2)
) -> float:
    """T minimising NLL of softmax(scores/T). Optimised over log T."""
    labels = np.asarray(labels, dtype=int)
    res = minimize_scalar(
        lambda lt: nll(scores, labels, float(np.exp(lt))),
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
    )
    return float(np.exp(res.x))


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Per-bin confidence, accuracy and counts for a reliability diagram (confidence = max prob)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore"):
        acc = np.bincount(b, weights=correct, minlength=n_bins) / cnt
        mconf = np.bincount(b, weights=conf, minlength=n_bins) / cnt
    return {"edges": edges, "confidence": mconf, "accuracy": acc, "count": cnt}


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    r = reliability(probs, labels, n_bins)
    m = r["count"] > 0
    return float(
        np.sum(r["count"][m] / r["count"].sum() * np.abs(r["accuracy"][m] - r["confidence"][m]))
    )


def fit_policy(
    scores_by_len: np.ndarray,
    labels: np.ndarray,
    lengths: list[float],
    stop_cfg: dict,
    n_bins: int = 10,
    gap_s: float = 1.0,
) -> dict:
    """Fit temperature (pooled over window lengths) then theta for dynamic stopping.

    scores_by_len [n_trials, n_lens, n_targets]. Returns T, theta, ECE before/after and the theta table.
    """
    from neurohack.decoders.stopping import fit_theta

    n_tr, n_len, n_tg = scores_by_len.shape
    pooled = scores_by_len.reshape(-1, n_tg)
    pooled_y = np.repeat(labels, n_len)
    t = fit_temperature(pooled, pooled_y)
    probs = softmax(scores_by_len / t)
    theta, table = fit_theta(
        probs,
        labels,
        lengths,
        stop_cfg["target_accuracy"],
        tuple(stop_cfg["theta_grid"]),
        n_tg,
        gap_s,
        stop_cfg["on_timeout"],
    )
    return {
        "temperature": t,
        "theta": theta,
        "ece_before": ece(softmax(pooled), pooled_y, n_bins),
        "ece_after": ece(softmax(pooled / t), pooled_y, n_bins),
        "theta_table": table,
    }
