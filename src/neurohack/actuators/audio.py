"""Low-latency audio actuator (pygame.mixer). Pre-generates the tone so trigger() is just play()."""

from __future__ import annotations

import numpy as np
import pylsl

from neurohack.actuators.base import Actuator

_SR = 44100


class AudioActuator(Actuator):
    def __init__(
        self, tone_hz: float, tone_ms: float, volume: float, mixer_buffer: int = 256
    ) -> None:
        import pygame

        pygame.mixer.pre_init(frequency=_SR, size=-16, channels=2, buffer=mixer_buffer)
        pygame.mixer.init()
        self.buffer_latency_ms = 1000.0 * mixer_buffer / _SR
        t = np.arange(int(_SR * tone_ms / 1000)) / _SR
        ramp = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.005)  # 5 ms fade in/out, no clicks
        wave = (volume * 32767 * ramp * np.sin(2 * np.pi * tone_hz * t)).astype(np.int16)
        self.sound = pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack([wave, wave])))
        self.n_triggers = 0

    def trigger(self, event: dict) -> float:
        self.sound.play()
        self.n_triggers += 1
        return pylsl.local_clock()

    def close(self) -> None:
        import pygame

        pygame.mixer.quit()
