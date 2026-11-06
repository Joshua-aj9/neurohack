"""Buffer, preprocessing, ITR, timing."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from neurohack.buffer import RingBuffer
from neurohack.eval.itr import itr_bits_per_min, wolpaw_bits
from neurohack.preprocessing import StreamFilter, car, design_stream_sos
from neurohack.timing import dejitter, effective_rate, fill_gaps, gap_report


def test_ringbuffer_wrap_and_window():
    rb = RingBuffer(2, 10)
    for k in range(5):
        d = np.vstack([np.arange(k * 4, k * 4 + 4)] * 2).astype(float)
        rb.append(d, d[0] / 100)
    assert rb.n_written == 20 and rb.n_available == 10
    data, ts = rb.latest(3)
    assert data[0].tolist() == [17, 18, 19]
    w = rb.window(0.12, 4)
    assert w is not None and w[0][0].tolist() == [12, 13, 14, 15]
    assert rb.window(0.18, 4) is None  # not yet available
    assert rb.window(0.01, 2) is None  # overwritten


def test_ringbuffer_oversized_append():
    rb = RingBuffer(1, 5)
    rb.append(np.arange(12, dtype=float)[None], np.arange(12, dtype=float))
    assert rb.latest(5)[0][0].tolist() == [7, 8, 9, 10, 11]
    assert rb.n_written == 12


def test_stream_filter_matches_batch():
    fs = 250.0
    sos = design_stream_sos(fs, 50.0, 30.0, (1.0, 90.0), 4)
    x = np.random.default_rng(0).standard_normal((3, 2000))
    sf = StreamFilter(sos, 3)
    chunks = [sf(x[:, i : i + 37]) for i in range(0, 2000, 37)]
    y_stream = np.concatenate(chunks, axis=1)
    zi = signal.sosfilt_zi(sos)[:, None, :] * x[None, :, 0:1]
    y_batch, _ = signal.sosfilt(sos, x, axis=1, zi=zi)
    np.testing.assert_allclose(y_stream, y_batch, atol=1e-10)


def test_notch_attenuates_50hz():
    fs = 250.0
    t = np.arange(5000) / fs
    sf = StreamFilter(design_stream_sos(fs, 50.0, 30.0, None, 4), 1)
    y = sf(np.sin(2 * np.pi * 50 * t)[None])
    assert np.std(y[0, 2500:]) < 0.05


def test_car_zero_mean():
    x = np.random.default_rng(1).standard_normal((4, 100))
    np.testing.assert_allclose(car(x).mean(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize(
    "n,p,expected", [(4, 1.0, 2.0), (2, 1.0, 1.0), (4, 0.25, 0.0), (2, 0.4, 0.0)]
)
def test_wolpaw_known_values(n, p, expected):
    assert wolpaw_bits(n, p) == pytest.approx(expected)


def test_wolpaw_formula_and_rate():
    n, p = 4, 0.9
    b = np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    assert wolpaw_bits(n, p) == pytest.approx(b)
    assert itr_bits_per_min(n, p, 3.0) == pytest.approx(b * 20)
    assert itr_bits_per_min(n, p, 1.0) == pytest.approx(b * 60)


def test_dejitter_recovers_clock():
    rng = np.random.default_rng(0)
    c = np.arange(1, 5001)
    true_t = 100.0 + c / 250.03
    noisy = true_t + rng.normal(0, 0.004, c.size)
    dj = dejitter(noisy, c, 250.0)
    assert np.max(np.abs(dj - true_t)) < 5e-4
    assert effective_rate(dj, c) == pytest.approx(250.03, abs=0.01)


def test_gap_report_and_fill():
    c = np.array([1, 2, 3, 6, 7, 8, 10])
    rep = gap_report(c)
    assert rep["n_lost"] == 3 and rep["n_expected"] == 10 and rep["max_gap"] == 2
    data = c[None].astype(float) * 2
    filled, cf, ts, mask = fill_gaps(data, c, c / 250.0, 250.0)
    assert cf.tolist() == list(range(1, 11))
    np.testing.assert_allclose(filled[0], np.arange(1, 11) * 2.0)
    assert mask.sum() == 3
