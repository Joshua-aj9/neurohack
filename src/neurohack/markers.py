"""LSL marker outlet/inlet helpers. Markers are JSON strings: {"event": ..., ...fields}."""

from __future__ import annotations

import json

import pylsl


def encode(event: str, **fields) -> str:
    return json.dumps({"event": event, **fields}, separators=(",", ":"))


def decode(value: str) -> dict:
    try:
        d = json.loads(value)
        return d if isinstance(d, dict) else {"event": str(d)}
    except json.JSONDecodeError:
        return {"event": value}


class MarkerOutlet:
    def __init__(
        self, name: str = "neurohack_markers", stype: str = "Markers", source_id: str = "neurohack"
    ) -> None:
        info = pylsl.StreamInfo(name, stype, 1, pylsl.IRREGULAR_RATE, pylsl.cf_string, source_id)
        self.outlet = pylsl.StreamOutlet(info)

    def push(self, event: str, timestamp: float | None = None, **fields) -> float:
        """Push a marker; returns the LSL timestamp used."""
        t = pylsl.local_clock() if timestamp is None else timestamp
        self.outlet.push_sample([encode(event, **fields)], t)
        return t


class MarkerInlet:
    def __init__(
        self, name: str | None = "neurohack_markers", stype: str = "Markers", timeout: float = 5.0
    ) -> None:
        prop, val = ("name", name) if name else ("type", stype)
        found = pylsl.resolve_byprop(prop, val, minimum=1, timeout=timeout)
        if not found:
            raise RuntimeError(f"no marker stream {prop}={val!r}")
        self.inlet = pylsl.StreamInlet(found[0], processing_flags=pylsl.proc_clocksync)

    def read(self) -> list[tuple[float, str]]:
        samples, ts = self.inlet.pull_chunk(timeout=0.0)
        return [(t, s[0]) for s, t in zip(samples, ts, strict=True)]
