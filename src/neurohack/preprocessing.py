"""Causal, stateful filtering for streams + pure helpers (CAR, windowing)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def design_stream_sos(
    srate: float,
    notch_hz: float | None,
    notch_q: float,
    bandpass_hz: tuple[float, float] | None,
    order: int,
    bandstop_hz: tuple[float, float] | None = None,
    bandstop_order: int = 2,
) -> np.ndarray:
    """SOS for: Butterworth bandstop (optional; broad mains rejection), notch (optional),
    Butterworth bandpass (optional)."""
    parts = []
    if bandstop_hz:
        parts.append(
            signal.butter(
                bandstop_order, list(bandstop_hz), btype="bandstop", fs=srate, output="sos"
            )
        )
    if notch_hz:
        b, a = signal.iirnotch(notch_hz, notch_q, fs=srate)
        parts.append(signal.tf2sos(b, a))
    if bandpass_hz:
        lo, hi = bandpass_hz
        hi = min(hi, 0.45 * srate)
        parts.append(signal.butter(order, [lo, hi], btype="bandpass", fs=srate, output="sos"))
    if not parts:
        return np.array([[1.0, 0, 0, 1.0, 0, 0]])
    return np.vstack(parts)


class StreamFilter:
    """Applies SOS filters chunk-by-chunk, preserving state between calls (causal)."""

    def __init__(self, sos: np.ndarray, n_channels: int) -> None:
        self.sos = sos
        self.n_channels = n_channels
        self._zi: np.ndarray | None = None

    @classmethod
    def from_config(cls, pre_cfg: dict, srate: float, n_channels: int) -> StreamFilter:
        sos = design_stream_sos(
            srate,
            pre_cfg.get("notch_hz"),
            pre_cfg.get("notch_q", 30.0),
            pre_cfg.get("bandpass_hz"),
            pre_cfg.get("bandpass_order", 4),
            pre_cfg.get("bandstop_hz"),
            pre_cfg.get("bandstop_order", 2),
        )
        return cls(sos, n_channels)

    def reset(self) -> None:
        self._zi = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] == 0:
            return x
        if self._zi is None:
            # initialise to steady state for the first sample to avoid a big step transient
            zi0 = signal.sosfilt_zi(self.sos)  # (n_sections, 2)
            self._zi = zi0[:, None, :] * x[None, :, 0:1]
        y, self._zi = signal.sosfilt(self.sos, x, axis=1, zi=self._zi)
        return y


def car(x: np.ndarray) -> np.ndarray:
    """Common average reference across channels (rows)."""
    return x - x.mean(axis=0, keepdims=True)


def detrend(x: np.ndarray) -> np.ndarray:
    return signal.detrend(x, axis=-1, type="linear")


def epochs_from_onsets(data: np.ndarray, onsets: np.ndarray, start: int, length: int) -> np.ndarray:
    """Cut epochs [n_epochs, n_ch, length] starting at onset+start samples. Out-of-range onsets dropped."""
    out = [
        data[:, o + start : o + start + length]
        for o in onsets
        if o + start >= 0 and o + start + length <= data.shape[1]
    ]
    return np.stack(out) if out else np.zeros((0, data.shape[0], length))
