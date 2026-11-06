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


def _filtered(cfgs, eeg):
    from neurohack.preprocessing import StreamFilter

    sf = StreamFilter.from_config(cfgs["decoder"]["preprocessing"], 250.0, eeg.shape[0])
    return sf(eeg)


def test_mains_is_a_contact_check_not_railing(synth, cfgs):
    """LIVE 6 Nov (dry): ~500 uV RMS mains zeroed every window via railing + ratio. Now mains is
    scored once (contact, absolute uV on raw) and amplitude checks use the decoder-filtered signal."""
    synth.artifacts = {"line"}
    synth.start()
    data, _ = synth.generate(1250)
    eeg = data[synth.info.picks("eeg")]
    filt = _filtered(cfgs, eeg)
    q = assess(eeg[:, -250:], 250.0, NAMES, cfgs["quality"], filtered=filt[:, -250:])
    assert any("50 Hz" in r for r in q.reasons)
    assert not any("huge amplitude" in r for r in q.reasons)
    assert q.metrics["line_noise"].min() > cfgs["quality"]["line_noise"]["bad_uv"]
    q_raw = assess(eeg[:, -250:], 250.0, NAMES, cfgs["quality"])  # legacy: no filtered signal
    assert any("huge amplitude" in r for r in q_raw.reasons)


def test_saline_level_mains_passes(synth, cfgs):
    """15 uV RMS of mains (LIVE saline 6 Nov: 8.7-20.7 uV on PO7/Oz/PO8) must not cost quality."""
    synth.start()
    data, _ = synth.generate(1250)
    eeg = data[synth.info.picks("eeg")] + 15 * np.sqrt(2) * np.sin(
        2 * np.pi * 49.75 * np.arange(1250) / 250.0
    )
    filt = _filtered(cfgs, eeg)
    q = assess(eeg[:, -250:], 250.0, NAMES, cfgs["quality"], filtered=filt[:, -250:])
    assert np.all(q.penalties["line_noise"] == 0.0)
    assert q.score_of(["PO7", "Oz", "PO8"]) > 0.8, q.as_dict()
