"""LSL inlet source (UnicornLSL or any LSL EEG stream)."""

from __future__ import annotations

import numpy as np
import pylsl

from neurohack.sources.base import Source, StreamInfo, empty_read, unicorn_info_from_config


def resolve(stream_type: str, name: str | None, timeout: float) -> pylsl.StreamInfo:
    if name:
        found = pylsl.resolve_byprop("name", name, minimum=1, timeout=timeout)
        found = [s for s in found if s.type() == stream_type] or found
    else:
        found = pylsl.resolve_byprop("type", stream_type, minimum=1, timeout=timeout)
    if not found:
        raise RuntimeError(
            f"No LSL stream of type={stream_type!r} name={name!r} within {timeout}s. "
            "Is Unicorn Suite + UnicornLSL running and 'Start' pressed?"
        )
    return found[0]


def list_streams(timeout: float = 2.0) -> list[pylsl.StreamInfo]:
    return list(pylsl.resolve_streams(wait_time=timeout))


def stream_channel_labels(info: pylsl.StreamInfo) -> list[str]:
    """Channel labels from the stream's XML description (empty strings if absent)."""
    labels = []
    ch = info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels


class LSLSource(Source):
    """Pulls chunks from one LSL inlet. Timestamps are clock-synced to pylsl.local_clock()."""

    def __init__(
        self,
        device_cfg: dict,
        stream_type: str | None = None,
        name: str | None = None,
        max_buffer_s: int = 30,
    ) -> None:
        lsl_cfg = device_cfg["lsl"]
        self.device_cfg = device_cfg
        self.stream_type = stream_type or (
            lsl_cfg["stream_type"] if lsl_cfg["mode"] == "combined" else "EEG"
        )
        self.name = name if name is not None else lsl_cfg["stream_name"]
        self.timeout = float(lsl_cfg["resolve_timeout_s"])
        self.max_buffer_s = max_buffer_s
        self.inlet: pylsl.StreamInlet | None = None
        self.lsl_info: pylsl.StreamInfo | None = None
        self.info = StreamInfo(float(device_cfg["srate"]), [], [])

    def start(self) -> None:
        self.lsl_info = resolve(self.stream_type, self.name, self.timeout)
        self.inlet = pylsl.StreamInlet(
            self.lsl_info,
            max_buflen=self.max_buffer_s,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_monotonize,
        )
        full = self.inlet.info()
        n = full.channel_count()
        srate = full.nominal_srate()
        if n == len(self.device_cfg["channel_index"]):
            self.info = unicorn_info_from_config(self.device_cfg, name=full.name())
        else:
            labels = stream_channel_labels(full)
            eeg_names = self.device_cfg["eeg_channels"]
            names = [
                lab or (eeg_names[i] if i < len(eeg_names) and n == len(eeg_names) else f"ch{i}")
                for i, lab in enumerate(labels)
            ]
            self.info = StreamInfo(srate, names, ["eeg"] * n, name=full.name())
        self.info.srate = srate
        self.info.meta = {
            "lsl_name": full.name(),
            "lsl_type": full.type(),
            "source_id": full.source_id(),
            "channel_format": full.channel_format(),
            "hostname": full.hostname(),
        }
        self.inlet.open_stream(timeout=self.timeout)

    def stop(self) -> None:
        if self.inlet is not None:
            self.inlet.close_stream()
            self.inlet = None

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if self.inlet is None:
            return empty_read(self.info.n_channels)
        samples, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=4096)
        if not ts:
            return empty_read(self.info.n_channels)
        return np.asarray(samples, dtype=float).T, np.asarray(ts, dtype=float)
