"""Stimulus helpers + headless smoke test of the G4 app (SDL dummy driver; not a timing measurement)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from neurohack.stimulus import cross_layout, frame_stats, luminance


def test_luminance_range_and_mean():
    t = np.arange(6000) / 60.0
    v = np.array([luminance(x, 8.57, 0.0) for x in t])
    assert v.min() >= 0 and v.max() <= 1
    assert abs(v.mean() - 0.5) < 0.01
    assert luminance(0.0, 10, np.pi / 2) == pytest.approx(1.0)
    assert luminance(0.01, 10, 0, "square") == 1.0


def test_frame_stats_detects_drops():
    fl = np.arange(0, 600) / 60.0
    fl = np.delete(fl, [100, 200, 201])  # 1 single + 1 double drop = 3 missed frames
    st = frame_stats(fl, 60.0)
    assert st["n_dropped_frames"] == 3 and st["n_late_intervals"] == 2
    assert st["measured_hz"] == pytest.approx(60.0)


def test_cross_layout_inside_screen():
    for x, y, w, h in cross_layout(1920, 1080, 260, 0.08):
        assert 0 <= x and x + w <= 1920 and 0 <= y and y + h <= 1080


def test_ssvep_stim_headless(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps"))
    import ssvep_stim

    from neurohack import stimulus

    monkeypatch.setattr(stimulus.Display, "check_refresh", lambda self: (True, "forced for test"))
    monkeypatch.setattr(stimulus.Display, "measure_refresh", lambda self, n: 60.0)
    monkeypatch.setattr(ssvep_stim, "Session", _TmpSession(tmp_path))
    monkeypatch.setattr(sys, "argv", ["ssvep_stim", "--seconds", "0.5", "--windowed"])
    ssvep_stim.main()
    assert (tmp_path / "flip_times.npy").exists()


class _TmpSession:
    def __init__(self, root):
        self.root = root

    def __call__(self, name, label="LIVE"):
        from neurohack.session import Session

        s = Session.__new__(Session)
        s.dir = self.root
        s.label = label
        s.electrodes = "n/a"
        s._events = (self.root / "events.jsonl").open("a", encoding="utf-8")
        return s
