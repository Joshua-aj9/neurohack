"""Background recorder: pulls a Source (+ optional marker inlet) into a session folder.

Output: <dir>/eeg.npz (data [n_ch, n], timestamps [n], srate, ch_names, ch_types) and
<dir>/markers.jsonl ({"t": lsl_time, "value": str}). All times on the LSL local clock.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from neurohack.sources.base import Source


class Recorder:
    def __init__(
        self, source: Source, out_dir: Path, marker_inlet=None, poll_s: float = 0.01, on_chunk=None
    ) -> None:
        self.source = source
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.marker_inlet = marker_inlet
        self.poll_s = poll_s
        self.on_chunk = on_chunk  # optional callback(data, ts) for live consumers
        self._chunks: list[np.ndarray] = []
        self._ts: list[np.ndarray] = []
        self._markers: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            data, ts = self.source.read()
            if ts.size:
                with self._lock:
                    self._chunks.append(data)
                    self._ts.append(ts)
                if self.on_chunk:
                    self.on_chunk(data, ts)
            if self.marker_inlet is not None:
                m = self.marker_inlet.read()
                if m:
                    with self._lock:
                        self._markers.extend(m)
            time.sleep(self.poll_s)

    def add_marker(self, t: float, value: str) -> None:
        with self._lock:
            self._markers.append((t, value))

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if not self._ts:
                return np.zeros((self.source.info.n_channels, 0)), np.zeros(0)
            return np.concatenate(self._chunks, axis=1), np.concatenate(self._ts)

    def stop(self) -> Path:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.save()

    def save(self) -> Path:
        data, ts = self.snapshot()
        info = self.source.info
        path = self.out_dir / "eeg.npz"
        np.savez_compressed(
            path,
            data=data,
            timestamps=ts,
            srate=info.srate,
            ch_names=np.array(info.ch_names),
            ch_types=np.array(info.ch_types),
        )
        with (self.out_dir / "markers.jsonl").open("w", encoding="utf-8") as f:
            for t, v in sorted(self._markers):
                f.write(json.dumps({"t": t, "value": v}) + "\n")
        return path
