"""G3 analysis: channel-set eligibility uses the live quality gate (SYNTHETIC)."""

from __future__ import annotations

import importlib.util

import numpy as np

from neurohack.config import REPO_ROOT
from neurohack.preprocessing import StreamFilter

_spec = importlib.util.spec_from_file_location(
    "analyze_session", REPO_ROOT / "scripts" / "analyze_session.py"
)
analyze_session = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_session)


def test_unwetted_channel_makes_set_ineligible(synth, cfgs):
    """LIVE G3 6 Nov: Pz carried ~3 mV of mains; a set containing it passed the gate in 0% of windows."""
    synth.start()
    data, _ = synth.generate(int(20 * 250))
    eeg_idx = synth.info.picks("eeg")
    names = [synth.info.ch_names[i] for i in eeg_idx]
    raw = data[eeg_idx].copy()
    raw[names.index("Pz")] += 3000 * np.sin(2 * np.pi * 50.0 * np.arange(raw.shape[1]) / 250.0)
    filt = StreamFilter.from_config(cfgs["decoder"]["preprocessing"], 250.0, len(names))(raw)
    qmin = cfgs["decoder"]["stopping"]["quality_min"]
    skip = cfgs["decoder"]["analysis"]["quality_skip_s"]
    good = analyze_session.quality_pass_frac(
        raw, filt, 250.0, names, ["PO7", "Oz", "PO8"], cfgs["quality"], qmin, skip
    )
    bad = analyze_session.quality_pass_frac(
        raw, filt, 250.0, names, ["Pz", "PO7", "Oz", "PO8"], cfgs["quality"], qmin, skip
    )
    frac_min = cfgs["decoder"]["analysis"]["min_quality_pass_frac"]
    assert good >= frac_min and bad == 0.0


def test_rest_windows_cover_static_gaps_only():
    """No-flicker windows run from trial_end + offset to the next trial_start + latency."""
    fs, n = 250.0, 250 * 20
    ts = np.arange(n) / fs
    sess = {
        "data": np.zeros((2, n)),
        "ts": ts,
        "srate": fs,
        "ch_types": ["eeg", "eeg"],
        "markers": [
            {"t": 1.0, "value": '{"event": "trial_start"}'},
            {"t": 5.0, "value": '{"event": "trial_end"}'},
            {"t": 7.0, "value": '{"event": "trial_start"}'},
            {"t": 11.0, "value": '{"event": "trial_end"}'},
        ],
    }
    filt = np.vstack([ts, ts])  # value = time, so each window reveals where it starts
    out = analyze_session.rest_windows(sess, filt, [1.0, 1.5], offset_s=0.3, latency_s=0.14)
    w1 = out[1.0]
    starts = w1[:, 0, 0]
    # gap 1: 5.3 .. 7.14 -> starts 5.3, 5.55, 5.8, 6.05 (6.3 + 1 > 7.14); gap 2: 11.3 .. end of data
    assert np.allclose(starts[:4], [5.3, 5.55, 5.8, 6.05], atol=1 / fs)
    assert all(s >= 11.3 - 1 / fs for s in starts[4:])
    assert w1.shape[1:] == (2, 250) and out[1.5].shape[2] == 375
