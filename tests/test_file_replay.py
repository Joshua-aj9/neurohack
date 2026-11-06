"""Day-of dataset readers (sources/file.py) and replay from MNE Raw with event markers."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from neurohack.sources.file import describe, read_raw_any
from neurohack.sources.replay import from_raw


def test_csv_and_npy(tmp_path):
    arr = np.random.default_rng(0).standard_normal((1000, 4))  # [n, ch] layout -> transposed
    np.savetxt(tmp_path / "x.csv", arr, delimiter=",")
    np.save(tmp_path / "x.npy", arr.T)
    for name in ["x.csv", "x.npy"]:
        raw = read_raw_any(tmp_path / name, srate=250.0)
        assert raw.get_data().shape == (4, 1000)
        np.testing.assert_allclose(raw.get_data()[0] * 1e6, arr[:, 0], atol=1e-9)
    with pytest.raises(ValueError):
        read_raw_any(tmp_path / "x.csv")  # srate required


def test_fif_and_replay_markers(tmp_path):
    fs = 250.0
    info = mne.create_info(["Oz", "Pz", "STI"], fs, ["eeg", "eeg", "stim"])
    data = np.zeros((3, 2500))
    data[:2] = np.random.default_rng(1).standard_normal((2, 2500)) * 1e-5
    data[2, [500, 1500]] = [1, 2]
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.save(tmp_path / "t_raw.fif", verbose=False)
    r2 = read_raw_any(tmp_path / "t_raw.fif")
    assert describe(r2).startswith("3 ch @ 250.0 Hz")
    rep = from_raw(r2, realtime=False, event_id={"left": 1, "right": 2})
    assert rep.info.ch_names == ["Oz", "Pz"]
    rep.start()
    got, markers = 0, []
    while not rep.finished:
        d, _ = rep.read()
        got += d.shape[1]
        markers += rep.read_markers()
    assert got == 2500
    assert [m for _, m in markers] == ["event|left", "event|right"]
    assert markers[0][0] == pytest.approx(2.0)
