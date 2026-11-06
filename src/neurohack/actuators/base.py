"""Actuator interface: trigger(event) performs the action and returns the LSL time it was issued."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Actuator(ABC):
    @abstractmethod
    def trigger(self, event: dict) -> float:
        """Fire. Returns pylsl.local_clock() right after the action was handed to the OS/driver."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


def make_actuator(cfg: dict) -> Actuator:
    kind = cfg["type"]
    if kind == "audio":
        from neurohack.actuators.audio import AudioActuator

        return AudioActuator(cfg["tone_hz"], cfg["tone_ms"], cfg["volume"], cfg["mixer_buffer"])
    if kind == "visual":
        from neurohack.actuators.visual import VisualActuator

        return VisualActuator()
    from neurohack.actuators.log_only import LogOnlyActuator

    return LogOnlyActuator()
