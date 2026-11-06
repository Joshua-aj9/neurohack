"""Contact report (eyes open / closed) on SYNTHETIC data."""

from __future__ import annotations

import numpy as np

from neurohack.eval.contact import contact_report, mean_abs_corr


def _record(synth, cfgs, closed_gain: float, extra=None):
    cc = cfgs["quality"]["contact_check"]
    st = cfgs["closedloop"]["synthetic_test"]
    synth.start()
    synth.ssvep_gain = 0.0
    fs = synth.info.srate
    chunks, labels = [], []
    for lab, dur in [("settle", cc["settle_s"])] + [(b, cc["block_s"]) for b in cc["blocks"]]:
        synth.alpha_gain = closed_gain if lab == "closed" else st["rest_alpha_gain"]
        d, _ = synth.generate(int(dur * fs))
        chunks.append(d)
        labels.append(np.full(d.shape[1], lab))
    eeg = synth.info.picks("eeg")
    x = np.concatenate(chunks, axis=1)[eeg]
    if extra is not None:
        x = x + extra(x.shape[1], fs)[None, :]  # identical on every channel = reference noise
    names = [synth.info.ch_names[i] for i in eeg]
    dcfg = cfgs["decoder"]
    return contact_report(
        x,
        np.concatenate(labels),
        fs,
        names,
        dcfg["channels"],
        cc,
        dcfg["preprocessing"],
        cfgs["quality"],
    )


def test_alpha_reactivity_detected(synth, cfgs):
    rep = _record(synth, cfgs, cfgs["closedloop"]["synthetic_test"]["on_alpha_gain"])
    assert rep["eeg_visible"]
    assert set(rep["reactive_decision_channels"]) == {"PO7", "Oz", "PO8"}
    assert rep["n_segments"]["open"] > 0 and rep["n_segments"]["closed"] > 0


def test_no_reactivity_when_alpha_unchanged(synth, cfgs):
    rest = cfgs["closedloop"]["synthetic_test"]["rest_alpha_gain"]
    rep = _record(synth, cfgs, rest)
    assert not rep["eeg_visible"]
    assert any("occipital contact" in h for h in rep["hints"])


def test_shared_reference_noise_flagged(synth, cfgs):
    def ref_noise(n, fs):  # large mains + slow drift through the reference: same on all channels
        t = np.arange(n) / fs
        return (
            800 * np.sin(2 * np.pi * 50 * t)
            + 150 * np.sin(2 * np.pi * 0.7 * t)
            + 60 * np.sin(2 * np.pi * 3 * t)
        )

    rep = _record(synth, cfgs, 3.0, ref_noise)
    assert (
        rep["mean_abs_corr_filtered"] >= cfgs["quality"]["contact_check"]["common_mode_corr_warn"]
    )
    assert any("reference/ground" in h for h in rep["hints"])
    oz = rep["channels"]["Oz"]
    assert oz["raw_ptp_uv"] > oz["filtered_ptp_uv"]  # mains removed by the decoder filter
    assert oz["car_rms_uv"] < oz["filtered_rms_uv"]  # CAR removes the shared part


def test_mean_abs_corr_bounds():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(1000)
    assert mean_abs_corr(np.vstack([s, s, -s])) > 0.999
    assert mean_abs_corr(rng.standard_normal((4, 5000))) < 0.05
