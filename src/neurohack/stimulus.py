"""SSVEP visual stimulation helpers (pygame-ce): display with vsync + refresh check, luminance, frame stats.

Pure functions (luminance, frame_stats, cross_layout) are testable without a display.
"""

from __future__ import annotations

import math

import numpy as np


def luminance(t: float, freq: float, phase: float, mode: str = "sinusoid") -> float:
    """Sampled-sinusoid L(t) = 0.5*(1 + sin(2 pi f t + phi)) in [0,1]; 'square' = on/off at f."""
    s = math.sin(2 * math.pi * freq * t + phase)
    if mode == "sinusoid":
        return 0.5 * (1.0 + s)
    if mode == "square":
        return 1.0 if s >= 0 else 0.0
    raise ValueError(mode)


def frame_stats(flip_times: np.ndarray, nominal_hz: float, drop_factor: float = 1.5) -> dict:
    """Frame-interval statistics from flip timestamps (seconds).

    A dropped frame is an interval > drop_factor * nominal; the number of missed frames per such interval
    is round(interval * nominal) - 1.
    """
    d = np.diff(np.asarray(flip_times, dtype=float))
    nominal = 1.0 / nominal_hz
    late = d > drop_factor * nominal
    missed = int(np.clip(np.round(d[late] * nominal_hz) - 1, 1, None).sum()) if late.any() else 0
    expected = int(round((flip_times[-1] - flip_times[0]) * nominal_hz))
    return {
        "n_flips": int(len(flip_times)),
        "duration_s": float(flip_times[-1] - flip_times[0]),
        "measured_hz": float(1.0 / np.median(d)),
        "interval_mean_ms": float(d.mean() * 1e3),
        "interval_std_ms": float(d.std() * 1e3),
        "interval_max_ms": float(d.max() * 1e3),
        "n_late_intervals": int(late.sum()),
        "n_dropped_frames": missed,
        "dropped_frac": float(missed / max(1, expected)),
    }


def cross_layout(
    screen_w: int, screen_h: int, size: int, margin_frac: float
) -> list[tuple[int, int, int, int]]:
    """Four target rects (x, y, w, h): top, right, bottom, left — centre left free for text."""
    m = int(margin_frac * min(screen_w, screen_h))
    cx, cy = screen_w // 2, screen_h // 2
    return [
        (cx - size // 2, m, size, size),  # top
        (screen_w - m - size, cy - size // 2, size, size),  # right
        (cx - size // 2, screen_h - m - size, size, size),  # bottom
        (m, cy - size // 2, size, size),  # left
    ]


class Display:
    """Fullscreen vsync pygame display with measured refresh rate."""

    def __init__(self, stim_cfg: dict, windowed: bool = False) -> None:
        import pygame

        self.pg = pygame
        pygame.init()
        idx = stim_cfg.get("display_index", 0)
        sizes = pygame.display.get_desktop_sizes()
        self.size = sizes[idx] if not windowed else (1280, 720)
        flags = pygame.SCALED | (
            pygame.FULLSCREEN if (stim_cfg["fullscreen"] and not windowed) else 0
        )
        self.screen = pygame.display.set_mode(
            self.size, flags, vsync=1 if stim_cfg["vsync"] else 0, display=idx
        )
        pygame.display.set_caption("neurohack SSVEP")
        pygame.mouse.set_visible(windowed)
        self.cfg = stim_cfg
        self.api_refresh_hz = None
        try:
            self.api_refresh_hz = float(pygame.display.get_current_refresh_rate())
        except Exception:
            pass
        self.measured_hz = self.measure_refresh(stim_cfg["refresh_probe_frames"])
        self.period = 1.0 / self.measured_hz

    def measure_refresh(self, n: int) -> float:
        import pylsl

        ts = []
        for _ in range(n):
            self.screen.fill((0, 0, 0))
            self.pg.display.flip()
            ts.append(pylsl.local_clock())
            self.pg.event.pump()
        d = np.diff(ts[10:])
        return float(1.0 / np.median(d))

    def check_refresh(self) -> tuple[bool, str]:
        want, tol = self.cfg["refresh_hz"], self.cfg["refresh_tolerance_hz"]
        ok = abs(self.measured_hz - want) <= tol
        msg = (
            f"measured refresh {self.measured_hz:.2f} Hz (API: {self.api_refresh_hz}), config {want} Hz "
            f"± {tol} -> {'OK' if ok else 'MISMATCH'}"
        )
        return ok, msg

    def flip(self) -> float:
        """Flip (blocks on vsync) and return the LSL time right after the flip."""
        import pylsl

        self.pg.display.flip()
        return pylsl.local_clock()

    def close(self) -> None:
        self.pg.quit()


def grey(level: float) -> tuple[int, int, int]:
    v = int(round(255 * min(1.0, max(0.0, level))))
    return (v, v, v)
