"""Synthetic Unicorn-like EEG source with controllable SSVEP, alpha and artifacts (SYNTHETIC only).

Same 17-channel layout as the real device so all downstream code runs unchanged.
`realtime=True` paces by wall clock (pylsl.local_clock); `realtime=False` returns fixed chunks on a
virtual clock (for fast tests and long soak runs).
"""

from __future__ import annotations

import numpy as np
import pylsl
from scipy import signal

from neurohack.sources.base import Source, StreamInfo, empty_read

# Paul Kellet's pink-noise IIR approximation (1/f), stable, unit-ish gain.
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
_PINK_STD = float(
    np.std(
        signal.lfilter(_PINK_B, _PINK_A, np.random.default_rng(0).standard_normal(200_000))[1000:]
    )
)


class SyntheticSource(Source):
    def __init__(
        self,
        info: StreamInfo,
        cfg: dict,
        realtime: bool = True,
        clock=pylsl.local_clock,
    ) -> None:
        self.info = info
        self.cfg = cfg
        self.realtime = realtime
        self._clock = clock
        self._rng = np.random.default_rng(cfg.get("seed", 0))
        self._eeg = info.picks("eeg")
        self._k = 0  # samples generated so far (sample index)
        self._t0 = 0.0
        self._running = False
        n_eeg = len(self._eeg)
        self._pink_zi = np.zeros((n_eeg, len(_PINK_A) - 1))
        self._emg_sos = signal.butter(
            4, [20, min(100, info.srate / 2 - 1)], "bandpass", fs=info.srate, output="sos"
        )
        self._emg_zi = np.zeros((self._emg_sos.shape[0], n_eeg, 2))
        # background alpha = narrowband noise (not phase-locked), shared across channels
        a_hz = cfg["alpha_hz"]
        self._alpha_sos = signal.butter(
            2, [a_hz - 1.5, a_hz + 1.5], "bandpass", fs=info.srate, output="sos"
        )
        self._alpha_zi = np.zeros((self._alpha_sos.shape[0], 2))
        probe = signal.sosfilt(self._alpha_sos, np.random.default_rng(1).standard_normal(100_000))
        self._alpha_norm = float(np.std(probe[2000:]))
        ss = cfg["ssvep"]
        self._ssvep_gain = np.array(
            [ss["spatial_gain"].get(info.ch_names[i], 0.1) for i in self._eeg]
        )
        post = {"Oz", "PO7", "PO8", "Pz", "O1", "O2", "POz"}
        self._alpha_gain_ch = np.array(
            [1.0 if info.ch_names[i] in post else 0.3 for i in self._eeg]
        )
        # controllable state
        self.stim_freq: float | None = None
        self.stim_phase: float = 0.0
        self.stim_onset: float = 0.0
        self.alpha_gain: float = 1.0
        self.ssvep_gain: float = 1.0
        self.artifacts: set[str] = set()
        self.drop_prob: float = 0.0
        self.extra_noise_uv: float = 0.0

    # ---- control -------------------------------------------------------------------------
    def set_stimulus(
        self, freq: float | None, phase: float = 0.0, onset: float | None = None
    ) -> None:
        """Start (or stop with None) a simulated SSVEP from `onset` (LSL/virtual time)."""
        self.stim_freq = freq
        self.stim_phase = phase
        self.stim_onset = self.now() if onset is None else onset

    def now(self) -> float:
        return self._clock() if self.realtime else self._t0 + self._k / self.info.srate

    # ---- Source API ----------------------------------------------------------------------
    def start(self) -> None:
        self._t0 = self._clock() if self.realtime else 0.0
        self._k = 0
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._running:
            return empty_read(self.info.n_channels)
        if self.realtime:
            due = int((self._clock() - self._t0) * self.info.srate) - self._k
        else:
            due = max(1, int(round(self.cfg["chunk_s"] * self.info.srate)))
        if due <= 0:
            return empty_read(self.info.n_channels)
        return self.generate(due)

    # ---- generation ----------------------------------------------------------------------
    def generate(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        cfg, fs = self.cfg, self.info.srate
        k = self._k + np.arange(n)
        t = self._t0 + k / fs
        n_eeg = len(self._eeg)
        rng = self._rng

        white = rng.standard_normal((n_eeg, n))
        pink, self._pink_zi = signal.lfilter(_PINK_B, _PINK_A, white, axis=1, zi=self._pink_zi)
        eeg = cfg["noise_uv"] * pink / _PINK_STD + cfg["white_uv"] * rng.standard_normal((n_eeg, n))
        eeg += self.extra_noise_uv * rng.standard_normal((n_eeg, n))
        alpha, self._alpha_zi = signal.sosfilt(
            self._alpha_sos, rng.standard_normal(n), zi=self._alpha_zi
        )
        eeg += (
            cfg["alpha_uv"]
            * self.alpha_gain
            * self._alpha_gain_ch[:, None]
            * (alpha / self._alpha_norm)[None, :]
        )
        line_amp = cfg["line_uv"] * (30.0 if "line" in self.artifacts else 1.0)
        eeg += line_amp * np.sin(2 * np.pi * 50.0 * t)[None, :]
        eeg += cfg["dc_uv"]

        if self.stim_freq is not None:
            ss = cfg["ssvep"]
            tt = t - self.stim_onset - ss["latency_s"]
            on = tt >= 0
            wave = np.zeros(n)
            for h in range(1, ss["n_harmonics"] + 1):
                wave += ss["harmonic_decay"] ** (h - 1) * np.sin(
                    2 * np.pi * h * self.stim_freq * tt + h * self.stim_phase
                )
            eeg += (
                ss["amplitude_uv"]
                * self.ssvep_gain
                * self._ssvep_gain[:, None]
                * (wave * on)[None, :]
            )

        if "emg" in self.artifacts:
            burst, self._emg_zi = signal.sosfilt(
                self._emg_sos, 40.0 * rng.standard_normal((n_eeg, n)), axis=1, zi=self._emg_zi
            )
            eeg += burst
        if "flat" in self.artifacts:
            eeg[:] = cfg["dc_uv"]
        if "rail" in self.artifacts:
            eeg[:] = 1e5

        out = np.zeros((self.info.n_channels, n))
        out[self._eeg] = eeg
        motion = "motion" in self.artifacts
        for i, typ in enumerate(self.info.ch_types):
            name = self.info.ch_names[i]
            if typ == "accel":
                base = 1.0 if name.endswith("Z") else 0.0
                out[i] = base + (0.2 if motion else 0.002) * rng.standard_normal(n)
            elif typ == "gyro":
                out[i] = (60.0 if motion else 0.3) * rng.standard_normal(n)
            elif typ == "battery":
                out[i] = 100.0
            elif typ == "validation":
                out[i] = 1.0

        # Device counter = sample index + 1. A dropped sample is missing from the output, so the
        # counter of the next delivered sample jumps (exactly what a lost Bluetooth frame looks like).
        if "counter" in self.info.ch_types:
            out[self.info.ch_types.index("counter")] = k + 1
        keep = rng.random(n) >= self.drop_prob if self.drop_prob > 0 else np.ones(n, dtype=bool)
        self._k += n
        return out[:, keep], t[keep]
