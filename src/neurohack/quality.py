"""Per-channel signal quality in [0, 1] with human-readable reasons. Pure functions.

Two views of the same window (thresholds in quality.yaml):
  - RAW signal: clipping (true amplifier limits) and mains amplitude, used as an electrode-CONTACT check.
    LIVE 6 Nov: wetted electrodes picked up ~11-17 uV of mains, dry ones ~500 uV (Joshua's room).
  - DECODER-FILTERED signal (mains band-stop + bandpass, what the decoder sees): flatline, railing
    (movement / drift), EMG. Scoring these on the raw signal let mains alone zero every window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass
class QualityReport:
    ch_names: list[str]
    scores: np.ndarray  # [n_ch] in [0,1]
    penalties: dict[str, np.ndarray]  # check -> [n_ch] penalty in [0,1]
    metrics: dict[str, np.ndarray]  # check -> [n_ch] raw metric
    reasons: list[str] = field(default_factory=list)

    def score_of(self, names: list[str]) -> float:
        """Min score over the given channels (the decoder is only as good as its worst channel)."""
        idx = [self.ch_names.index(n) for n in names if n in self.ch_names]
        return float(self.scores[idx].min()) if idx else 0.0

    def reasons_for(self, names: list[str]) -> list[str]:
        """Reasons that concern `names` (e.g. the decoder channels), channel lists trimmed to them.
        Unused electrodes (LIVE 6 Nov: unwetted Fz..Pz) otherwise dominate the messages."""
        out = []
        for r in self.reasons:
            who, what = r.split(": ", 1)
            if who == "all":
                out.append(r)
                continue
            mine = [c for c in who.split(", ") if c in names]
            if mine:
                out.append(f"{', '.join(mine)}: {what}")
        return out

    def as_dict(self) -> dict:
        return {
            "scores": dict(zip(self.ch_names, np.round(self.scores, 3).tolist(), strict=True)),
            "reasons": self.reasons,
        }


_LABELS = {
    "flatline": "flat signal (electrode off?)",
    "railing": "huge amplitude (saturation / movement)",
    "clipping": "clipped samples (amplifier railing)",
    "line_noise": "high 50 Hz mains (poor contact?)",
    "emg": "high EMG (jaw clench / frown?)",
    "motion": "head motion",
    "counter": "dropped samples (Bluetooth)",
}


def ramp(x: np.ndarray | float, warn: float, bad: float) -> np.ndarray:
    """0 at `warn`, 1 at `bad`, linear between, clipped. Works for either direction (bad < warn)."""
    x = np.asarray(x, dtype=float)
    return np.clip((x - warn) / (bad - warn), 0.0, 1.0)


def band_power(f: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    m = (f >= band[0]) & (f <= band[1])
    return pxx[..., m].sum(axis=-1)


def psd(x: np.ndarray, srate: float) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[-1]
    return signal.welch(x, fs=srate, nperseg=min(n, int(srate)), axis=-1, detrend="linear")


def band_rms(x: np.ndarray, srate: float, band: tuple[float, float]) -> np.ndarray:
    """RMS amplitude (uV) of each row inside `band`, from the Welch PSD used by every check."""
    f, pxx = psd(x, srate)
    return np.sqrt(band_power(f, pxx, band) * (f[1] - f[0]))


def counter_loss(counter: np.ndarray) -> float:
    """Fraction of samples lost, from gaps in a +1-incrementing sample counter."""
    c = np.asarray(counter, dtype=float)
    if c.size < 2:
        return 0.0
    d = np.diff(c)
    lost = np.clip(d - 1, 0, None).sum()
    expected = c[-1] - c[0] + 1
    return float(lost / expected) if expected > 0 else 0.0


def assess(
    eeg: np.ndarray,
    srate: float,
    ch_names: list[str],
    cfg: dict,
    accel: np.ndarray | None = None,
    gyro: np.ndarray | None = None,
    counter: np.ndarray | None = None,
    filtered: np.ndarray | None = None,
) -> QualityReport:
    """eeg [n_ch, n] RAW microvolts over the quality window; filtered [n_ch, n] the same window after
    the decoder's preprocessing (None: use raw for every check)."""
    n_ch = eeg.shape[0]
    x = signal.detrend(eeg if filtered is None else filtered, axis=1, type="linear")
    f, pxx = psd(x, srate)
    metrics: dict[str, np.ndarray] = {}
    pen: dict[str, np.ndarray] = {}

    metrics["flatline"] = x.std(axis=1)
    pen["flatline"] = ramp(metrics["flatline"], cfg["flatline"]["warn"], cfg["flatline"]["bad"])
    metrics["railing"] = np.ptp(x, axis=1)
    pen["railing"] = ramp(metrics["railing"], cfg["railing"]["warn"], cfg["railing"]["bad"])
    lo, hi = eeg.min(axis=1, keepdims=True), eeg.max(axis=1, keepdims=True)
    stuck = ((eeg == lo) | (eeg == hi)).mean(axis=1)
    stuck[(hi - lo).ravel() == 0] = 0.0  # perfectly flat handled by flatline
    metrics["clipping"] = stuck
    pen["clipping"] = ramp(stuck, cfg["clipping"]["warn"], cfg["clipping"]["bad"])

    ln = cfg["line_noise"]
    eps = 1e-12
    metrics["line_noise"] = band_rms(
        signal.detrend(eeg, axis=1, type="linear"), srate, ln["band_hz"]
    )
    pen["line_noise"] = ramp(metrics["line_noise"], ln["warn_uv"], ln["bad_uv"])
    em = cfg["emg"]
    metrics["emg"] = band_power(f, pxx, em["band_hz"]) / (
        band_power(f, pxx, em["ref_band_hz"]) + eps
    )
    pen["emg"] = ramp(metrics["emg"], em["warn"], em["bad"])

    motion_pen = 0.0
    if accel is not None and accel.size:
        a_std = float(np.linalg.norm(accel, axis=0).std())
        motion_pen = max(motion_pen, float(ramp(a_std, **cfg["motion"]["accel_std_g"])))
    if gyro is not None and gyro.size:
        g_mean = float(np.linalg.norm(gyro, axis=0).mean())
        motion_pen = max(motion_pen, float(ramp(g_mean, **cfg["motion"]["gyro_mean_dps"])))
    metrics["motion"] = np.full(n_ch, motion_pen)
    pen["motion"] = np.full(n_ch, motion_pen)

    loss = counter_loss(counter) if counter is not None else 0.0
    metrics["counter"] = np.full(n_ch, loss)
    pen["counter"] = np.full(n_ch, float(ramp(loss, cfg["counter"]["warn"], cfg["counter"]["bad"])))

    # a flat channel has no meaningful spectrum: do not also blame it for line noise / EMG
    flat = pen["flatline"] >= 1.0
    pen["line_noise"][flat] = 0.0
    pen["emg"][flat] = 0.0

    scores = np.prod([1.0 - p for p in pen.values()], axis=0)
    reasons = []
    thr = cfg["reason_min_penalty"]
    for check in ("motion", "counter"):
        if pen[check][0] >= thr:
            reasons.append(f"all: {_LABELS[check]}")
    for check, p in pen.items():
        if check in ("motion", "counter"):
            continue
        bad = [name for i, name in enumerate(ch_names) if p[i] >= thr]
        if bad:
            who = "all" if len(bad) == n_ch else ", ".join(bad)
            reasons.append(f"{who}: {_LABELS[check]}")
    return QualityReport(list(ch_names), scores, pen, metrics, reasons)
