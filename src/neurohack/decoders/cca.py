"""Standard CCA SSVEP decoder (Lin et al. 2006) + shared helpers (references, max canonical corr)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from neurohack.decoders.base import Decoder


@lru_cache(maxsize=256)
def _reference_cached(
    freq: float, n: int, srate: float, n_harmonics: int, max_hz: float
) -> np.ndarray:
    t = np.arange(n) / srate
    rows = []
    for k in range(1, n_harmonics + 1):
        if k * freq >= max_hz:
            break
        rows += [np.sin(2 * np.pi * k * freq * t), np.cos(2 * np.pi * k * freq * t)]
    if not rows:
        raise ValueError(f"no harmonic of {freq} Hz below {max_hz} Hz")
    ref = np.asarray(rows)
    ref.setflags(write=False)
    return ref


def reference(
    freq: float, n: int, srate: float, n_harmonics: int, max_hz: float | None = None
) -> np.ndarray:
    """sin/cos references at k*f, k=1..Nh, keeping k*f < max_hz (default Nyquist). Shape [2*Nh', n]."""
    nyq = srate / 2
    return _reference_cached(
        float(freq), int(n), float(srate), int(n_harmonics), float(min(max_hz or nyq, nyq))
    )


def _orth_basis(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Orthonormal basis of the column space of centred x [n, d], dropping (near-)zero columns."""
    x = x - x.mean(axis=0, keepdims=True)
    q, r = np.linalg.qr(x)
    keep = np.abs(np.diag(r)) > eps * max(1.0, np.abs(r).max())
    return q[:, keep]


def max_canon_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Largest canonical correlation between x [n, p] and y [n, q] (rows = samples)."""
    qx, qy = _orth_basis(x), _orth_basis(y)
    if qx.shape[1] == 0 or qy.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0))


class CCADecoder(Decoder):
    def __init__(
        self, freqs: list[float], srate: float, n_harmonics: int = 5, temperature: float = 1.0
    ) -> None:
        self.freqs = list(freqs)
        self.srate = srate
        self.n_harmonics = n_harmonics
        self.temperature = temperature

    def scores(self, window: np.ndarray) -> np.ndarray:
        n = window.shape[1]
        x = window.T
        return np.array(
            [
                max_canon_corr(x, reference(f, n, self.srate, self.n_harmonics).T) ** 2
                for f in self.freqs
            ]
        )
