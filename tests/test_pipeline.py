"""Sources + online pipeline: synthetic and replay go through the identical code path."""

from __future__ import annotations

import numpy as np

from neurohack.decoders.stopping import ABSTAIN_QUALITY, COMMIT
from neurohack.pipeline import OnlinePipeline
from neurohack.recorder import Recorder
from neurohack.sources.replay import ReplaySource, from_npz
from neurohack.sources.synthetic import SyntheticSource

FREQS = [8.57, 10.0, 12.0, 15.0]


def run_trials(pipe: OnlinePipeline, src, schedule, trial_s=4.5, rest_s=0.5):
    """schedule: list of target indices (synthetic) — drives the source stimulus if it has one."""
    out = []
    src.start()
    t = 0.0
    for tgt in schedule:
        # rest
        if isinstance(src, SyntheticSource):
            src.set_stimulus(None)
        while pipe.latest_time() < t + rest_s:
            pipe.pump()
        onset = t + rest_s
        if isinstance(src, SyntheticSource):
            src.set_stimulus(FREQS[tgt], onset=onset)
        pipe.start_trial(onset)
        dec = None
        while dec is None:
            pipe.pump()
            dec = pipe.update()
        out.append(dec)
        while pipe.latest_time() < onset + trial_s:
            pipe.pump()
        t = onset + trial_s
    src.stop()
    return out


def _cfg(cfgs, theta=0.5):
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfgs["decoder"].items()}
    d["stopping"] = dict(d["stopping"], theta=theta)
    d["calibration"] = dict(d["calibration"], temperature=0.02)
    return d


def test_synthetic_pipeline_decodes(cfgs, info):
    syn = dict(cfgs["synthetic"], ssvep=dict(cfgs["synthetic"]["ssvep"], amplitude_uv=3.0))
    src = SyntheticSource(info, syn, realtime=False)
    pipe = OnlinePipeline(src, _cfg(cfgs), cfgs["quality"], FREQS)
    sched = [0, 1, 2, 3, 2, 1]
    decs = run_trials(pipe, src, sched)
    commits = [d for d in decs if d.kind == COMMIT]
    assert len(commits) >= 4
    assert (
        sum(d.target == s for d, s in zip(decs, sched, strict=True) if d.kind == COMMIT)
        >= len(commits) - 1
    )


def test_abstains_on_emg(cfgs, info):
    src = SyntheticSource(info, cfgs["synthetic"], realtime=False)
    pipe = OnlinePipeline(src, _cfg(cfgs), cfgs["quality"], FREQS)
    src.artifacts = {"emg"}
    decs = run_trials(pipe, src, [1, 2])
    assert all(d.kind == ABSTAIN_QUALITY for d in decs)


def test_replay_identical_code_path(cfgs, info):
    """Synthetic run -> captured samples -> ReplaySource through the same OnlinePipeline -> identical
    per-window probabilities and decision."""
    syn = dict(cfgs["synthetic"], ssvep=dict(cfgs["synthetic"]["ssvep"], amplitude_uv=3.0))
    src = SyntheticSource(info, syn, realtime=False)
    chunks, tss = [], []
    orig_read = src.read

    def tapped_read():
        d, t = orig_read()
        chunks.append(d)
        tss.append(t)
        return d, t

    src.read = tapped_read
    pipe_a = OnlinePipeline(src, _cfg(cfgs, theta=0.9999), cfgs["quality"], FREQS)
    dec_a = run_trials(pipe_a, src, [3])[0]
    hist_a = pipe_a.last_history
    data, ts = np.concatenate(chunks, axis=1), np.concatenate(tss)

    rep = ReplaySource(info, data, ts, realtime=False)
    pipe_b = OnlinePipeline(rep, _cfg(cfgs, theta=0.9999), cfgs["quality"], FREQS)
    rep.start()
    pipe_b.start_trial(0.5)  # run_trials' first onset = rest_s
    dec_b = None
    while dec_b is None:
        pipe_b.pump()
        dec_b = pipe_b.update()
    assert dec_a == dec_b
    assert len(hist_a) == len(pipe_b.last_history) >= 1
    for ha, hb in zip(hist_a, pipe_b.last_history, strict=True):
        np.testing.assert_allclose(ha["probs"], hb["probs"], rtol=1e-12)
        assert ha["quality"] == hb["quality"]


def test_recorder_roundtrip(cfgs, info, tmp_path):
    src = SyntheticSource(info, cfgs["synthetic"], realtime=True)
    src.start()
    rec = Recorder(src, tmp_path)
    rec.start()
    import time

    time.sleep(0.5)
    rec.add_marker(123.0, '{"event":"x"}')
    path = rec.stop()
    src.stop()
    rp = from_npz(path, realtime=False)
    assert rp.info.ch_names == info.ch_names
    assert rp.data.shape[1] > 50
    assert len(rp._markers) == 1


def test_synthetic_drops_show_in_counter(cfgs, info):
    src = SyntheticSource(info, cfgs["synthetic"], realtime=False)
    src.drop_prob = 0.05
    src.start()
    d, t = src.generate(10000)
    cnt = d[info.picks("counter")[0]]
    loss = 1 - cnt.size / (cnt[-1] - cnt[0] + 1)
    assert 0.03 < loss < 0.07
