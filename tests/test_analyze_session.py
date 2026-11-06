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
