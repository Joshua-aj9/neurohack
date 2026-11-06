"""Timestamp de-jittering and gap handling using the device sample counter."""

from __future__ import annotations

import numpy as np


def dejitter(
    timestamps: np.ndarray, counter: np.ndarray, srate: float, fit_rate: bool = True
) -> np.ndarray:
    """Replace jittery per-sample LSL timestamps with a linear fit t = a + b * counter.

    With fit_rate=False the slope is fixed to 1/srate (only the offset is estimated, robustly via median).
    The fit uses the device counter, so dropped samples do not shift later timestamps.
    """
    c = np.asarray(counter, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    if fit_rate and c.size > 10:
        b, a = np.polyfit(c, t, 1)
        resid = t - (a + b * c)
        keep = np.abs(resid - np.median(resid)) < 5 * (
            np.median(np.abs(resid - np.median(resid))) + 1e-9
        )
        b, a = np.polyfit(c[keep], t[keep], 1)
        return a + b * c
    a = np.median(t - c / srate)
    return a + c / srate


def effective_rate(timestamps: np.ndarray, counter: np.ndarray | None = None) -> float:
    """Samples per second: counter span / time span (if counter given) else n / time span."""
    t = np.asarray(timestamps, dtype=float)
    span = t[-1] - t[0]
    if span <= 0:
        return float("nan")
    if counter is not None:
        return float((counter[-1] - counter[0]) / span)
    return float((t.size - 1) / span)


def gap_report(counter: np.ndarray) -> dict:
    c = np.asarray(counter, dtype=np.int64)
    d = np.diff(c)
    gaps = d[d > 1] - 1
    expected = int(c[-1] - c[0] + 1)
    return {
        "n_received": int(c.size),
        "n_expected": expected,
        "n_lost": int(gaps.sum()),
        "loss_frac": float(gaps.sum() / expected) if expected else 0.0,
        "n_gap_events": int(gaps.size),
        "max_gap": int(gaps.max()) if gaps.size else 0,
        "n_backwards": int((d < 1).sum()),
    }


def fill_gaps(
    data: np.ndarray, counter: np.ndarray, timestamps: np.ndarray, srate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate missing samples on the counter grid.

    Returns (data_filled, counter_filled, timestamps_filled, was_filled mask).
    """
    c = np.asarray(counter, dtype=np.int64)
    full = np.arange(c[0], c[-1] + 1)
    filled = np.empty((data.shape[0], full.size))
    for i in range(data.shape[0]):
        filled[i] = np.interp(full, c, data[i])
    ts = dejitter(timestamps, c, srate)
    b, a = np.polyfit(c, ts, 1)
    mask = ~np.isin(full, c)
    return filled, full, a + b * full, mask
