"""CCA / FBCCA / calibration / stopping."""

from __future__ import annotations

import numpy as np
import pytest

from neurohack.decoders.calibration import ece, fit_temperature, nll
from neurohack.decoders.cca import CCADecoder, max_canon_corr, reference
from neurohack.decoders.fbcca import FBCCADecoder, design_filterbank, subband_weights
from neurohack.decoders.stopping import (
    ABSTAIN_QUALITY,
    COMMIT,
    TIMEOUT,
    StoppingPolicy,
    fit_theta,
    simulate,
)

FS = 250.0
FREQS = [8.57, 10.0, 12.0, 15.0]


def ssvep_window(f, n, snr_amp=1.0, noise=1.0, seed=0, n_ch=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    s = snr_amp * (np.sin(2 * np.pi * f * t + 0.7) + 0.5 * np.sin(2 * np.pi * 2 * f * t + 1.1))
    return np.vstack([s * g for g in np.linspace(0.6, 1.0, n_ch)]) + noise * rng.standard_normal(
        (n_ch, n)
    )


def test_reference_trims_harmonics():
    r = reference(21.0, 250, FS, 5, max_hz=88.0)
    assert r.shape == (8, 250)  # 21,42,63,84 < 88; 105 dropped
    r2 = reference(15.0, 250, FS, 5)
    assert r2.shape == (10, 250)


def test_max_canon_corr_bounds():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((500, 2))
    x = y @ np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]])
    assert max_canon_corr(x, y) == pytest.approx(1.0, abs=1e-9)
    assert max_canon_corr(rng.standard_normal((5000, 3)), rng.standard_normal((5000, 4))) < 0.1
    assert max_canon_corr(np.zeros((100, 3)), y[:100]) == 0.0  # flat channels don't crash


def test_fbcca_weights():
    w = subband_weights(5, 1.25, 0.25)
    assert w[0] == pytest.approx(1.25) and w[1] == pytest.approx(2**-1.25 + 0.25)


@pytest.mark.parametrize("dec_cls", ["cca", "fbcca"])
def test_decoders_identify_target(dec_cls, cfgs):
    fb = cfgs["decoder"]["fbcca"]
    dec = CCADecoder(FREQS, FS, 5) if dec_cls == "cca" else FBCCADecoder(FREQS, FS, fb, 5)
    correct = 0
    for i, f in enumerate(FREQS):
        for seed in range(5):
            correct += dec.predict(ssvep_window(f, 500, 0.5, 1.0, seed)) == i
    assert correct >= 19


def test_fbcca_rejects_low_freqs(cfgs):
    with pytest.raises(ValueError):
        FBCCADecoder([6.5, 7.5, 13.5, 16.5], FS, cfgs["decoder"]["fbcca"])
    fb = dict(cfgs["decoder"]["fbcca"], subband_low_start_hz=6.0)
    FBCCADecoder([6.5, 7.5, 13.5, 16.5], FS, fb)


def test_filterbank_stable(cfgs):
    for sos in design_filterbank(FS, cfgs["decoder"]["fbcca"]):
        poles = np.concatenate([np.roots(s[3:]) for s in sos])
        assert np.all(np.abs(poles) < 1)


def test_temperature_scaling_reduces_nll():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 400)
    scores = rng.normal(0, 0.05, (400, 4))
    scores[np.arange(400), labels] += 0.08  # informative but small-scale scores (like FBCCA)
    t = fit_temperature(scores, labels)
    assert nll(scores, labels, t) < nll(scores, labels, 1.0)
    from neurohack.decoders.base import softmax

    assert ece(softmax(scores / t), labels) < ece(softmax(scores), labels)


def test_ece_perfect_calibration_is_small():
    rng = np.random.default_rng(1)
    n = 20000
    conf = rng.uniform(0.25, 1.0, n)
    correct = rng.random(n) < conf
    labels = np.zeros(n, dtype=int)
    probs = np.zeros((n, 4))
    probs[:, 0] = conf
    probs[:, 1:] = ((1 - conf) / 3)[:, None]
    labels[~correct] = 1
    assert ece(probs, labels) < 0.02


def test_stopping_policy_paths():
    pol = StoppingPolicy(
        theta=0.8, min_window_s=1.0, step_s=0.25, max_window_s=2.0, quality_min=0.5
    )
    assert pol.lengths == [1.0, 1.25, 1.5, 1.75, 2.0]
    p = np.array([0.1, 0.85, 0.05])
    assert pol.step(1.0, p, 0.9).kind == COMMIT
    assert pol.step(1.0, p, 0.2).kind == ABSTAIN_QUALITY
    low = np.array([0.4, 0.3, 0.3])
    assert pol.step(1.0, low, 0.9) is None
    d = pol.step(2.0, low, 0.9)
    assert d.kind == TIMEOUT and d.target is None


def test_simulate_and_fit_theta():
    lengths = [1.0, 2.0]
    # trial 0: confident & right at 1s; trial 1: right, more confident at 2s; trial 2: unsure & WRONG
    probs = np.array(
        [
            [[0.9, 0.1], [0.95, 0.05]],
            [[0.2, 0.8], [0.05, 0.95]],
            [[0.55, 0.45], [0.6, 0.4]],
        ]
    )
    labels = np.array([0, 1, 1])
    pred, t = simulate(probs, labels, 0.85, lengths)
    assert pred.tolist() == [0, 1, -1] and t.tolist() == [1.0, 2.0, 2.0]
    # theta <= 0.6 commits the wrong trial 2 (acc 2/3); 0.65 is the smallest theta reaching 100%
    theta, table = fit_theta(
        probs, labels, lengths, 0.95, (0.5, 0.95, 0.05), 2, min_commit_rate=0.3
    )
    assert theta == pytest.approx(0.65)


# ---- no-SSVEP floor (LIVE 6 Nov: speller committed on noise, see GATES.md G6) ----------------------
def test_fit_null_floor_recovers_power_law():
    from neurohack.decoders.calibration import fit_null_floor, null_floor_at

    rng = np.random.default_rng(0)
    # winning noise score ~ 0.8 / L: the 95th percentile follows the same law
    rest = {L: 0.8 / L * rng.uniform(0.5, 1.0, 5000) for L in (1.0, 1.25, 1.5)}
    fl = fit_null_floor(rest, 0.95, 20)
    assert fl["exponent"] == pytest.approx(-1.0, abs=0.05)
    assert null_floor_at(fl, 4.0) == pytest.approx(0.8 / 4 * 0.975, rel=0.05)
    assert null_floor_at(None, 1.0) == 0.0


def test_fit_null_floor_single_length_and_too_few():
    from neurohack.decoders.calibration import fit_null_floor, null_floor_at

    fl = fit_null_floor({1.0: np.full(30, 0.6), 1.5: np.full(5, 0.1)}, 0.95, 20)  # 1.5 s skipped
    assert fl["exponent"] == -1.0 and null_floor_at(fl, 2.0) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        fit_null_floor({1.0: np.full(5, 0.6)}, 0.95, 20)


def test_policy_does_not_commit_below_floor():
    floor = {"coef": 0.8, "exponent": -1.0}
    pol = StoppingPolicy(theta=0.9, min_window_s=1.0, step_s=0.5, max_window_s=2.0, floor=floor)
    sure = np.array([0.97, 0.01, 0.01, 0.01])
    assert (
        pol.step(1.0, sure, 1.0, score_max=0.5) is None
    )  # confident softmax, but noise-level score
    d = pol.step(1.0, sure, 1.0, score_max=0.85)
    assert d.kind == COMMIT and d.target == 0
    assert pol.step(2.0, sure, 1.0, score_max=0.3).kind == TIMEOUT  # floor(2 s) = 0.4
    assert StoppingPolicy(theta=0.9).step(1.0, sure, 1.0, score_max=0.0).kind == COMMIT  # no floor


def test_simulate_respects_allowed_mask():
    probs = np.array([[[0.9, 0.1], [0.95, 0.05]]])
    allowed = np.array([[False, True]])
    pred, t = simulate(probs, np.array([0]), 0.85, [1.0, 2.0], allowed=allowed)
    assert pred.tolist() == [0] and t.tolist() == [2.0]
