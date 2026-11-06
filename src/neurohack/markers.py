"""LSL marker outlet/inlet helpers. Markers are JSON strings: {"event": ..., ...fields}."""

from __future__ import annotations

import json
import os

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
    """String marker outlet. The source_id is unique per process by default, so two apps (or two
    copies of one app) publishing the same stream name can still be told apart by an inlet."""

    def __init__(
        self, name: str = "neurohack_markers", stype: str = "Markers", source_id: str | None = None
    ) -> None:
        self.source_id = source_id or f"{name}_pid{os.getpid()}"
        info = pylsl.StreamInfo(
            name, stype, 1, pylsl.IRREGULAR_RATE, pylsl.cf_string, self.source_id
        )
        self.outlet = pylsl.StreamOutlet(info)

    def push(self, event: str, timestamp: float | None = None, **fields) -> float:
        """Push a marker; returns the LSL timestamp used."""
        t = pylsl.local_clock() if timestamp is None else timestamp
        self.outlet.push_sample([encode(event, **fields)], t)
        return t


class MarkerInlet:
    def __init__(
        self,
        name: str | None = "neurohack_markers",
        stype: str = "Markers",
        timeout: float = 5.0,
        source_id: str | None = None,
    ) -> None:
        """Resolve by source_id (exact, preferred) else by name else by type."""
        if source_id:
            prop, val = "source_id", source_id
        else:
            prop, val = ("name", name) if name else ("type", stype)
        found = pylsl.resolve_byprop(prop, val, minimum=1, timeout=timeout)
        if not found:
            raise RuntimeError(f"no marker stream {prop}={val!r}")
        self.inlet = pylsl.StreamInlet(found[0], processing_flags=pylsl.proc_clocksync)

    def close(self) -> None:
        """Close before the outlet goes away, else liblsl logs 'transmission broke off' at exit."""
        self.inlet.close_stream()

    def read(self) -> list[tuple[float, str]]:
        samples, ts = self.inlet.pull_chunk(timeout=0.0)
        return [(t, s[0]) for s, t in zip(samples, ts, strict=True)]
