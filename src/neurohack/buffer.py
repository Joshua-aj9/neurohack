"""Thread-safe fixed-capacity ring buffer for multichannel samples + timestamps."""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, n_channels: int, capacity: int) -> None:
        self.n_channels = n_channels
        self.capacity = capacity
        self._data = np.zeros((n_channels, capacity))
        self._ts = np.full(capacity, np.nan)
        self._write = 0  # total samples ever written
        self._lock = threading.Lock()

    @property
    def n_written(self) -> int:
        return self._write

    @property
    def n_available(self) -> int:
        return min(self._write, self.capacity)

    def append(self, data: np.ndarray, ts: np.ndarray) -> None:
        n = data.shape[1]
        if n == 0:
            return
        if n > self.capacity:
            data, ts = data[:, -self.capacity :], ts[-self.capacity :]
            skipped = n - self.capacity
            n = self.capacity
        else:
            skipped = 0
        with self._lock:
            start = (self._write + skipped) % self.capacity
            idx = (start + np.arange(n)) % self.capacity
            self._data[:, idx] = data
            self._ts[idx] = ts
            self._write += skipped + n

    def latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Most recent n samples (fewer if not available), oldest first."""
        with self._lock:
            n = min(n, self.n_available)
            idx = (self._write - n + np.arange(n)) % self.capacity
            return self._data[:, idx].copy(), self._ts[idx].copy()

    def window(self, t_start: float, n: int) -> tuple[np.ndarray, np.ndarray] | None:
        """n samples starting at the first sample with ts >= t_start; None if not (yet) available."""
        with self._lock:
            m = self.n_available
            idx = (self._write - m + np.arange(m)) % self.capacity
            ts = self._ts[idx]
            first = int(np.searchsorted(ts, t_start, side="left"))
            if first + n > m:
                return None
            if first == 0 and m == self.capacity and ts[0] > t_start:
                return None  # requested data already overwritten
            sel = idx[first : first + n]
            return self._data[:, sel].copy(), self._ts[sel].copy()
