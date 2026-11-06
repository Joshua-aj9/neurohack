"""Closed-loop template: EEG feature -> controller (threshold/hysteresis/refractory) -> actuator.

`ProtocolRunner` alternates baseline (sham: triggers are logged, actuator not fired) and stimulation
blocks, logging every controller tick to the session events.jsonl. Software latency is measured from
the LSL timestamp of the newest sample in the window that crossed threshold to the actuator call.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pylsl

from neurohack.actuators.base import Actuator
from neurohack.buffer import RingBuffer
from neurohack.preprocessing import StreamFilter
from neurohack.quality import assess, band_power, psd
from neurohack.sources.base import Source


def bandpower_feature(
    eeg: np.ndarray, srate: float, band: tuple[float, float], ref_band: tuple[float, float] | None
) -> float:
    """Mean over channels of band power (relative to ref_band if given)."""
    f, pxx = psd(eeg, srate)
    p = band_power(f, pxx, band)
    if ref_band is not None:
        p = p / (band_power(f, pxx, ref_band) + 1e-12)
    return float(np.mean(p))


@dataclass
class Controller:
    threshold_on: float
    threshold_off: float
    refractory_s: float
    armed: bool = True
    last_trigger: float = float("-inf")

    def __post_init__(self) -> None:
        if self.threshold_off > self.threshold_on:
            raise ValueError("threshold_off must be <= threshold_on (hysteresis)")

    def update(self, t: float, value: float) -> bool:
        """Returns True when a trigger should fire at time t."""
        if not self.armed and value < self.threshold_off:
            self.armed = True
        if (
            self.armed
            and value > self.threshold_on
            and (t - self.last_trigger) >= self.refractory_s
        ):
            self.armed = False
            self.last_trigger = t
            return True
        return False


class ProtocolRunner:
    def __init__(
        self,
        source: Source,
        cfg: dict,
        pre_cfg: dict,
        quality_cfg: dict,
        actuator: Actuator,
        log: Callable[..., None],
        realtime: bool = True,
    ) -> None:
        self.source, self.cfg, self.qcfg, self.actuator, self.log = (
            source,
            cfg,
            quality_cfg,
            actuator,
            log,
        )
        self.realtime = realtime
        info = source.info
        self.srate = info.srate
        self.eeg_idx = info.picks("eeg")
        names = [info.ch_names[i] for i in self.eeg_idx]
        self.eeg_names = names
        fc = cfg["feature"]
        self.feat_idx = [names.index(c) for c in fc["channels"]]
        self.filter = StreamFilter.from_config(pre_cfg, self.srate, len(self.eeg_idx))
        self.buf = RingBuffer(len(self.eeg_idx), int(30 * self.srate))
        self.raw = RingBuffer(info.n_channels, int(30 * self.srate))
        c = cfg["controller"]
        self.controller = Controller(c["threshold_on"], c["threshold_off"], c["refractory_s"])
        self.ema: float | None = None
        self.triggers: list[dict] = []
        self.ticks: list[dict] = []
        self.adapted: dict | None = None

    def _pump(self) -> None:
        data, ts = self.source.read()
        if ts.size:
            self.raw.append(data, ts)
            self.buf.append(self.filter(data[self.eeg_idx]), ts)

    def tick(self, block: str) -> dict | None:
        fc = self.cfg["feature"]
        n = int(fc["window_s"] * self.srate)
        if self.buf.n_available < n:
            return None
        x, ts = self.buf.latest(n)
        t_data = float(ts[-1])
        t0 = time.perf_counter()
        ref = fc["ref_band_hz"] if fc["type"] == "bandpower_rel" else None
        v = bandpower_feature(x[self.feat_idx], self.srate, tuple(fc["band_hz"]), ref)
        a = fc["smoothing_alpha"]
        self.ema = v if self.ema is None else a * v + (1 - a) * self.ema
        raw, _ = self.raw.latest(int(self.qcfg["window_s"] * self.srate))
        q = assess(raw[self.eeg_idx], self.srate, self.eeg_names, self.qcfg)
        quality = q.score_of(fc["channels"])
        gated = quality < self.cfg["protocol"]["quality_min"]
        fire = False if gated else self.controller.update(t_data, self.ema)
        rec = {
            "t_data": t_data,
            "block": block,
            "feature": self.ema,
            "raw_feature": v,
            "quality": quality,
            "armed": self.controller.armed,
            "trigger": fire,
            "gated": gated,
        }
        if fire:
            if block == "stimulation":
                t_act = self.actuator.trigger({"feature": self.ema})
            else:
                t_act = pylsl.local_clock()  # sham: logged only
            rec.update(
                t_actuate=t_act,
                latency_ms=(t_act - t_data) * 1e3,
                compute_ms=(time.perf_counter() - t0) * 1e3,
                sham=block != "stimulation",
            )
            self.triggers.append(rec)
        self.ticks.append(rec)
        self.log("cl_tick", **rec)
        return rec

    def run(
        self, blocks: list[str], block_s: float, on_tick: Callable[[float], None] | None = None
    ) -> dict:
        self.source.start()
        upd = self.cfg["feature"]["update_s"]
        t_start = pylsl.local_clock()
        try:
            for bi, block in enumerate(blocks):
                self.log("block_start", block=block, index=bi)
                t_block = bi * block_s
                while True:
                    t_rel = (
                        (pylsl.local_clock() - t_start)
                        if self.realtime
                        else self._virtual_elapsed()
                    )
                    if t_rel >= t_block + block_s:
                        break
                    if on_tick:
                        on_tick(t_rel)
                    self._pump()
                    self.tick(block)
                    if self.realtime:
                        time.sleep(upd)
                self.log("block_end", block=block, index=bi)
                if bi == 0 and block == "baseline" and self.cfg["controller"]["baseline_adapt"]:
                    self.adapt_thresholds()
        finally:
            self.source.stop()
        lat = [t["latency_ms"] for t in self.triggers if not t["sham"]]
        return {
            "n_triggers": len(self.triggers),
            "n_stim_triggers": len(lat),
            "n_sham_triggers": sum(t["sham"] for t in self.triggers),
            "latency_ms_median": float(np.median(lat)) if lat else None,
            "latency_ms_max": float(np.max(lat)) if lat else None,
            "trigger_times": [t["t_data"] for t in self.triggers],
            "sham_flags": [t["sham"] for t in self.triggers],
            "adapted_thresholds": self.adapted,
        }

    def adapt_thresholds(self) -> None:
        """Person-specific thresholds from the first baseline block: mean + k*std of the smoothed feature."""
        vals = np.array([t["feature"] for t in self.ticks if t["block"] == "baseline"])
        if vals.size < 10:
            return
        c = self.cfg["controller"]
        mu, sd = float(vals.mean()), float(vals.std())
        self.controller.threshold_on = mu + c["k_on"] * sd
        self.controller.threshold_off = mu + c["k_off"] * sd
        self.controller.armed = True
        self.adapted = {
            "mean": mu,
            "std": sd,
            "threshold_on": self.controller.threshold_on,
            "threshold_off": self.controller.threshold_off,
        }
        self.log("thresholds_adapted", **self.adapted)

    def _virtual_elapsed(self) -> float:
        _, ts = self.raw.latest(1)
        return float(ts[-1]) if ts.size else 0.0
