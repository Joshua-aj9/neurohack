"""No-op actuator that only records triggers (sham blocks, tests, dry runs)."""

from __future__ import annotations

import pylsl

from neurohack.actuators.base import Actuator


class LogOnlyActuator(Actuator):
    def __init__(self) -> None:
        self.events: list[tuple[float, dict]] = []

    def trigger(self, event: dict) -> float:
        t = pylsl.local_clock()
        self.events.append((t, event))
        return t
