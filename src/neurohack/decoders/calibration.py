"""Temperature scaling (Guo et al. 2017), expected calibration error, and the no-SSVEP score floor.

Floor: softmax confidence is relative (which target is least bad), so with NO flicker on screen the
decoder is just as confident (LIVE 6 Nov, G3 rest periods: p_max >= theta in 48% of 1 s windows, 58% of
picks = 8.57 Hz). The floor is an absolute gate on the winning raw score, fitted on no-flicker windows.
"""

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


def fit_null_floor(
    rest_max_by_len: dict[float, np.ndarray], quantile: float, min_windows: int
) -> dict:
    """Fit floor(L) = coef * L**exponent through the `quantile` of the winning score on no-flicker
    windows of each length L (log-log least squares). Lengths with < min_windows windows are skipped.

    Noise CCA scores shrink with window length (~1/L for squared correlations; LIVE G3 rest 1-1.5 s:
    exponent -0.92), so the fit extrapolates to the longer windows that rest periods cannot supply.
    One usable length -> exponent fixed at -1.
    """
    pts = {
        float(L): float(np.quantile(v, quantile))
        for L, v in sorted(rest_max_by_len.items())
        if len(v) >= min_windows
    }
    if not pts:
        raise ValueError("no window length has enough no-flicker windows for a floor")
    ls, qs = np.log(list(pts)), np.log(list(pts.values()))
    if len(pts) >= 2:
        b, a = np.polyfit(ls, qs, 1)
    else:
        b, a = -1.0, float(qs[0] - (-1.0) * ls[0])
    return {
        "coef": float(np.exp(a)),
        "exponent": float(b),
        "quantile": quantile,
        "per_length": pts,
        "n_windows": {float(L): len(v) for L, v in rest_max_by_len.items()},
    }


def null_floor_at(floor: dict | None, window_s: float) -> float:
    """Floor for a window length (0 if no floor -> gate disabled)."""
    if not floor:
        return 0.0
    return float(floor["coef"] * window_s ** floor["exponent"])


def floor_mask(scores_by_len: np.ndarray, lengths: list[float], floor: dict | None) -> np.ndarray:
    """[n_trials, n_lens] True where the winning score clears the floor at that length."""
    thr = np.array([null_floor_at(floor, L) for L in lengths])
    return scores_by_len.max(axis=-1) >= thr[None, :]


def fit_policy(
    scores_by_len: np.ndarray,
    labels: np.ndarray,
    lengths: list[float],
    stop_cfg: dict,
    n_bins: int = 10,
    gap_s: float = 1.0,
    floor: dict | None = None,
) -> dict:
    """Fit temperature (pooled over window lengths) then theta for dynamic stopping.

    scores_by_len [n_trials, n_lens, n_targets]. With `floor`, theta is fitted for the gated policy
    (commit only where the winning score clears the floor). Returns T, theta, ECE before/after and the
    theta table.
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
        allowed=floor_mask(scores_by_len, lengths, floor) if floor else None,
    )
    return {
        "temperature": t,
        "theta": theta,
        "ece_before": ece(softmax(pooled), pooled_y, n_bins),
        "ece_after": ece(softmax(pooled / t), pooled_y, n_bins),
        "theta_table": table,
    }
