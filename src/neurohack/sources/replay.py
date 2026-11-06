"""Replay recorded arrays (npz / xdf / MOABB / any file) through the Source API at real-time pace.

Timestamps are re-based onto the live clock so replayed data is indistinguishable from a stream.
Markers that come with the recording are re-based the same way and served by `read_markers()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pylsl

from neurohack.sources.base import Source, StreamInfo, empty_read


class ReplaySource(Source):
    def __init__(
        self,
        info: StreamInfo,
        data: np.ndarray,
        timestamps: np.ndarray | None = None,
        markers: list[tuple[float, str]] | None = None,
        realtime: bool = True,
        chunk_s: float = 0.04,
        clock=pylsl.local_clock,
    ) -> None:
        if data.shape[0] != info.n_channels:
            raise ValueError(f"data has {data.shape[0]} channels, info has {info.n_channels}")
        self.info = info
        self.data = np.asarray(data, dtype=float)
        n = self.data.shape[1]
        ts = (
            np.arange(n) / info.srate if timestamps is None else np.asarray(timestamps, dtype=float)
        )
        self._rel_ts = ts - ts[0]
        self._markers = sorted(((t - ts[0], m) for t, m in (markers or [])), key=lambda x: x[0])
        self.realtime = realtime
        self.chunk = max(1, int(round(chunk_s * info.srate)))
        self._clock = clock
        self._pos = 0
        self._mpos = 0
        self._t0 = 0.0
        self._running = False

    @property
    def finished(self) -> bool:
        return self._pos >= self.data.shape[1]

    def start(self) -> None:
        self._t0 = self._clock() if self.realtime else 0.0
        self._pos = 0
        self._mpos = 0
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _due_index(self) -> int:
        if self.realtime:
            elapsed = self._clock() - self._t0
            return int(np.searchsorted(self._rel_ts, elapsed, side="right"))
        return min(self._pos + self.chunk, self.data.shape[1])

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._running or self.finished:
            return empty_read(self.info.n_channels)
        end = self._due_index()
        if end <= self._pos:
            return empty_read(self.info.n_channels)
        sl = slice(self._pos, end)
        self._pos = end
        return self.data[:, sl], self._t0 + self._rel_ts[sl]

    def read_markers(self) -> list[tuple[float, str]]:
        """Markers whose time has been reached (by data position), on the replay clock."""
        if not self._running or self._pos == 0:
            return []
        t_now = self._rel_ts[min(self._pos, len(self._rel_ts)) - 1]
        out = []
        while self._mpos < len(self._markers) and self._markers[self._mpos][0] <= t_now:
            t, m = self._markers[self._mpos]
            out.append((self._t0 + t, m))
            self._mpos += 1
        return out


# ---- loaders ------------------------------------------------------------------------------


def from_npz(path: str | Path, realtime: bool = True) -> ReplaySource:
    """Load a recorder session (eeg.npz [+ markers.jsonl in the same folder])."""
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    info = StreamInfo(
        srate=float(z["srate"]),
        ch_names=[str(c) for c in z["ch_names"]],
        ch_types=[str(c) for c in z["ch_types"]],
        name=path.stem,
    )
    markers = []
    mfile = path.parent / "markers.jsonl"
    if mfile.exists():
        for line in mfile.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                markers.append((float(rec["t"]), rec["value"]))
    return ReplaySource(info, z["data"], z["timestamps"], markers, realtime=realtime)


def from_xdf(
    path: str | Path,
    stream_type: str = "Data",
    realtime: bool = True,
    device_cfg: dict | None = None,
) -> ReplaySource:
    """Load the first stream of `stream_type` (+ all Markers streams) from an XDF file."""
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(path))
    eeg = next(s for s in streams if s["info"]["type"][0] == stream_type)
    data = np.asarray(eeg["time_series"], dtype=float).T
    ts = np.asarray(eeg["time_stamps"], dtype=float)
    srate = float(eeg["info"]["nominal_srate"][0])
    if device_cfg is not None and data.shape[0] == len(device_cfg["channel_index"]):
        from neurohack.sources.base import unicorn_info_from_config

        info = unicorn_info_from_config(device_cfg, name=Path(path).stem)
    else:
        info = StreamInfo(srate, [f"ch{i}" for i in range(data.shape[0])], ["eeg"] * data.shape[0])
    markers = []
    for s in streams:
        if s["info"]["type"][0] == "Markers":
            markers += [
                (float(t), str(v[0]))
                for t, v in zip(s["time_stamps"], s["time_series"], strict=True)
            ]
    return ReplaySource(info, data, ts, markers, realtime=realtime)


def from_raw(raw, realtime: bool = True, event_id: dict | None = None) -> ReplaySource:
    """Wrap an MNE Raw (e.g. from MOABB or neurohack.sources.file) as a ReplaySource.

    Stim-channel events become markers "event|<label>".
    """
    import mne

    picks = mne.pick_types(raw.info, eeg=True, stim=False, misc=False)
    data = raw.get_data(picks=picks) * 1e6  # V -> uV
    names = [raw.ch_names[i] for i in picks]
    info = StreamInfo(float(raw.info["sfreq"]), names, ["eeg"] * len(names), name="raw")
    markers: list[tuple[float, str]] = []
    if mne.pick_types(raw.info, stim=True).size:
        events = mne.find_events(raw, verbose=False)
        inv = {v: k for k, v in (event_id or {}).items()}
        for samp, _, code in events:
            markers.append(
                ((samp - raw.first_samp) / raw.info["sfreq"], f"event|{inv.get(code, code)}")
            )
    return ReplaySource(info, data, None, markers, realtime=realtime)
