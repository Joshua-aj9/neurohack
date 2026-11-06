"""Source ABC: every data source (synthetic, replay, LSL, file, serial) exposes the same API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StreamInfo:
    """Static description of a source."""

    srate: float
    ch_names: list[str]
    ch_types: list[
        str
    ]  # "eeg" | "accel" | "gyro" | "battery" | "counter" | "validation" | "stim" | "misc"
    name: str = "source"
    meta: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.ch_names)

    def index(self, name: str) -> int:
        return self.ch_names.index(name)

    def indices(self, names: list[str]) -> list[int]:
        return [self.ch_names.index(n) for n in names]

    def picks(self, ch_type: str) -> list[int]:
        return [i for i, t in enumerate(self.ch_types) if t == ch_type]


class Source(ABC):
    """start() -> read() repeatedly -> stop(). read() never blocks for long."""

    info: StreamInfo

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all samples available since the last call.

        Returns (data[n_ch, n], timestamps[n]); n may be 0.
        """

    def __enter__(self) -> Source:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def empty_read(n_ch: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((n_ch, 0)), np.zeros(0)


def unicorn_info_from_config(device_cfg: dict, name: str = "unicorn") -> StreamInfo:
    """Build the 17-channel Unicorn StreamInfo from device.yaml `channel_index`."""
    idx: dict[str, int] = device_cfg["channel_index"]
    names = [n for n, _ in sorted(idx.items(), key=lambda kv: kv[1])]
    eeg = set(device_cfg["eeg_channels"])

    def kind(n: str) -> str:
        if n in eeg:
            return "eeg"
        if n.startswith("Acc"):
            return "accel"
        if n.startswith("Gyr"):
            return "gyro"
        return n.lower()

    return StreamInfo(
        srate=float(device_cfg["srate"]),
        ch_names=names,
        ch_types=[kind(n) for n in names],
        name=name,
    )
