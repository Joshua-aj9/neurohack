"""Online SSVEP pipeline: Source -> ring buffers -> causal filter -> quality + decoder -> stopping.

Identical code path for synthetic, replay and LSL sources (G1 requirement).
Usage per trial: `start_trial(onset)`, then call `update()` until it returns a Decision.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from neurohack.buffer import RingBuffer
from neurohack.decoders.base import softmax
from neurohack.decoders.fbcca import make_decoder
from neurohack.decoders.stopping import Decision, StoppingPolicy
from neurohack.preprocessing import StreamFilter, car
from neurohack.quality import QualityReport, assess
from neurohack.sources.base import Source


@dataclass
class TrialState:
    onset: float
    next_idx: int = 0
    history: list[dict] = field(default_factory=list)


class OnlinePipeline:
    def __init__(
        self,
        source: Source,
        decoder_cfg: dict,
        quality_cfg: dict,
        freqs: list[float],
        buffer_s: float = 60.0,
        record: bool = False,
    ) -> None:
        self.source = source
        self.record = record  # keep every raw chunk (for offline diagnosis of live sessions)
        self._rec: list[tuple[np.ndarray, np.ndarray]] = []
        self.dcfg = decoder_cfg
        self.qcfg = quality_cfg
        self.freqs = list(freqs)
        info = source.info
        self.srate = info.srate
        self.eeg_idx = info.picks("eeg")
        self.eeg_names = [info.ch_names[i] for i in self.eeg_idx]
        self.dec_idx = [self.eeg_names.index(c) for c in decoder_cfg["channels"]]
        self.acc_idx, self.gyr_idx = info.picks("accel"), info.picks("gyro")
        self.cnt_idx = info.picks("counter")
        cap = int(buffer_s * self.srate)
        self.raw = RingBuffer(info.n_channels, cap)
        self.filt = RingBuffer(len(self.eeg_idx), cap)
        self.filter = StreamFilter.from_config(
            decoder_cfg["preprocessing"], self.srate, len(self.eeg_idx)
        )
        self.decoder = make_decoder(decoder_cfg, self.freqs, self.srate)
        self.policy = StoppingPolicy.from_config(decoder_cfg["stopping"])
        self.lengths = self.policy.lengths
        self.latency = float(decoder_cfg["visual_latency_s"])
        self.trial: TrialState | None = None
        self.last_history: list[dict] = []
        self.compute_ms: deque[float] = deque(maxlen=10_000)

    # ---- data ingestion ------------------------------------------------------------------
    def pump(self) -> int:
        data, ts = self.source.read()
        n = ts.size
        if n:
            self.raw.append(data, ts)
            if self.record:
                self._rec.append((data.copy(), ts.copy()))
            eeg = self.filter(data[self.eeg_idx])
            if self.dcfg["preprocessing"].get("car"):
                eeg = car(eeg)
            self.filt.append(eeg, ts)
        return n

    def save_recording(self, path) -> bool:
        """Write every pumped raw sample to `path` (.npz, same layout as recorder.Recorder)."""
        if not self._rec:
            return False
        info = self.source.info
        np.savez_compressed(
            path,
            data=np.concatenate([d for d, _ in self._rec], axis=1),
            timestamps=np.concatenate([t for _, t in self._rec]),
            srate=info.srate,
            ch_names=np.array(info.ch_names),
            ch_types=np.array(info.ch_types),
        )
        return True

    def latest_time(self) -> float:
        _, ts = self.raw.latest(1)
        return float(ts[-1]) if ts.size else float("-inf")

    # ---- quality -------------------------------------------------------------------------
    def quality(self, t_end: float | None = None) -> QualityReport:
        n = int(self.qcfg["window_s"] * self.srate)
        data, ts = self.raw.latest(n)
        filt, fts = self.filt.latest(n)  # appended together with raw -> same timestamps
        if t_end is not None and ts.size:
            data = data[:, ts <= t_end][:, -n:]
            filt = filt[:, fts <= t_end][:, -n:]
        eeg = data[self.eeg_idx]
        if eeg.shape[1] < 8:
            return assess(
                np.zeros((len(self.eeg_idx), int(self.srate))),
                self.srate,
                self.eeg_names,
                self.qcfg,
            )
        acc = data[self.acc_idx] if self.acc_idx else None
        gyr = data[self.gyr_idx] if self.gyr_idx else None
        cnt = data[self.cnt_idx[0]] if self.cnt_idx else None
        filt = filt if filt.shape == eeg.shape else None
        return assess(eeg, self.srate, self.eeg_names, self.qcfg, acc, gyr, cnt, filtered=filt)

    # ---- decoding ------------------------------------------------------------------------
    def window(self, onset: float, length_s: float) -> np.ndarray | None:
        n = int(round(length_s * self.srate))
        w = self.filt.window(onset + self.latency, n)
        return None if w is None else w[0][self.dec_idx]

    def set_calibration(self, temperature: float, theta: float, floor: dict | None = None) -> None:
        """floor: no-SSVEP floor from calibration (None = no absolute gate, softmax confidence only)."""
        self.decoder.temperature = temperature
        self.policy.theta = theta
        self.policy.floor = floor

    def start_trial(self, onset: float) -> None:
        self.trial = TrialState(onset=onset)

    def update(self) -> Decision | None:
        """Evaluate every window length whose data has arrived; return a Decision when the trial ends."""
        tr = self.trial
        if tr is None:
            return None
        while tr.next_idx < len(self.lengths):
            L = self.lengths[tr.next_idx]
            w = self.window(tr.onset, L)
            if w is None:
                return None
            t0 = time.perf_counter()
            scores = self.decoder.scores(w)
            probs = softmax(scores / self.decoder.temperature)
            self.compute_ms.append((time.perf_counter() - t0) * 1e3)
            q = self.quality(t_end=tr.onset + self.latency + L).score_of(self.dcfg["channels"])
            tr.history.append(
                {
                    "window_s": L,
                    "scores": scores.tolist(),
                    "probs": probs.tolist(),
                    "quality": q,
                    "floor": self.policy.floor_at(L),
                }
            )
            tr.next_idx += 1
            dec = self.policy.step(L, probs, q, float(np.max(scores)))
            if dec is not None:
                self.last_history = tr.history
                self.trial = None
                return dec
        return None
