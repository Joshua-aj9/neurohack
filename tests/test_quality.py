"""Quality monitor: each failure mode is detected and named; clean signal scores high."""

from __future__ import annotations

import numpy as np

from neurohack.quality import assess, counter_loss
from neurohack.sources.synthetic import SyntheticSource

NAMES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


def _window(synth: SyntheticSource, artifacts=(), seconds=1.0):
    synth.artifacts = set(artifacts)
    synth.start()
    synth.generate(250)  # warm up filters
    data, _ = synth.generate(int(250 * seconds))
    info = synth.info
    return (
        data[info.picks("eeg")],
        data[info.picks("accel")],
        data[info.picks("gyro")],
        data[info.picks("counter")[0]],
    )


def test_clean_signal_high_quality(synth, cfgs):
    eeg, acc, gyr, cnt = _window(synth)
    q = assess(eeg, 250.0, NAMES, cfgs["quality"], acc, gyr, cnt)
    assert q.scores.min() > 0.8, q.as_dict()
    assert q.reasons == []


def test_flatline_detected(synth, cfgs):
    eeg, acc, gyr, cnt = _window(synth)
    eeg[6] = 5.0
    q = assess(eeg, 250.0, NAMES, cfgs["quality"], acc, gyr, cnt)
    assert q.scores[6] < 0.1
    assert any("Oz: flat" in r for r in q.reasons)
    assert q.scores[0] > 0.8


def test_emg_detected(synth, cfgs):
    eeg, acc, gyr, cnt = _window(synth, {"emg"})
    q = assess(eeg, 250.0, NAMES, cfgs["quality"], acc, gyr, cnt)
    assert q.score_of(["PO7", "Oz", "PO8"]) < 0.5
    assert any("EMG" in r for r in q.reasons)
    assert len(q.reasons) <= 2  # grouped, not one line per channel


def test_line_noise_detected(synth, cfgs):
    eeg, acc, gyr, cnt = _window(synth, {"line"})
    q = assess(eeg, 250.0, NAMES, cfgs["quality"], acc, gyr, cnt)
    assert any("50 Hz" in r for r in q.reasons)


def test_motion_detected(synth, cfgs):
    eeg, acc, gyr, cnt = _window(synth, {"motion"})
    q = assess(eeg, 250.0, NAMES, cfgs["quality"], acc, gyr, cnt)
    assert any("motion" in r for r in q.reasons)
    assert q.scores.max() < 0.5


def test_railing_detected(synth, cfgs):
    eeg, acc, gyr, cnt = _window(synth)
    eeg[5, 100:110] += 3000.0
    q = assess(eeg, 250.0, NAMES, cfgs["quality"], acc, gyr, cnt)
    assert any("PO7: huge amplitude" in r for r in q.reasons)


def test_counter_loss():
    assert counter_loss(np.arange(1, 1001)) == 0.0
    c = np.delete(np.arange(1, 1001), np.arange(0, 1000, 50)[1:])  # drop 19 samples
    assert abs(counter_loss(c) - 19 / 1000) < 1e-9
