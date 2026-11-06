"""Visual actuator: sets a flash deadline that a pygame app draws (keeps rendering in the app loop)."""

from __future__ import annotations

import pylsl

from neurohack.actuators.base import Actuator


class VisualActuator(Actuator):
    def __init__(self, flash_s: float = 0.2) -> None:
        self.flash_s = flash_s
        self.flash_until = 0.0
        self.n_triggers = 0

    def trigger(self, event: dict) -> float:
        t = pylsl.local_clock()
        self.flash_until = t + self.flash_s
        self.n_triggers += 1
        return t

    def active(self) -> bool:
        return pylsl.local_clock() < self.flash_until
