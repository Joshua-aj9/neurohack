"""G1 evidence: decision latency per window + long synthetic soak with memory tracking.  [SYNTHETIC]

uv run python scripts/g1_pipeline_bench.py                 # accelerated 30-min soak (virtual clock)
uv run python scripts/g1_pipeline_bench.py --realtime      # wall-clock 30-min soak
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import psutil

from neurohack.config import REPO_ROOT, load_config
from neurohack.decoders.calibration import fit_policy
from neurohack.decoders.fbcca import make_decoder
from neurohack.decoders.stopping import COMMIT
from neurohack.pipeline import OnlinePipeline
from neurohack.sources.base import unicorn_info_from_config
from neurohack.sources.synthetic import SyntheticSource


def latency_bench(dcfg: dict, freqs: list[float], srate: float, reps: int = 200) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for method in ("fbcca", "cca"):
        dec = make_decoder(dict(dcfg, method=method), freqs, srate)
        for L in (1.0, 2.0, 4.0):
            w = rng.standard_normal((len(dcfg["channels"]), int(L * srate)))
            dec.predict_proba(w)  # warm caches
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                dec.predict_proba(w)
                ts.append((time.perf_counter() - t0) * 1e3)
            out[f"{method}_{L:.0f}s"] = {
                "median_ms": float(np.median(ts)),
                "p99_ms": float(np.percentile(ts, 99)),
                "max_ms": float(np.max(ts)),
            }
    return out


def run_trial(
    pipe: OnlinePipeline,
    src: SyntheticSource,
    freq: float,
    st: dict,
    realtime: bool,
    timing: dict | None = None,
):
    """One rest + stimulus trial; records per-update() compute time into `timing` if given."""
    src.set_stimulus(None)
    t_rest = src.now() + st["rest_s"]
    while src.now() < t_rest:
        pipe.pump()
        if realtime:
            time.sleep(0.005)
    onset = src.now()
    src.set_stimulus(freq, onset=onset)
    pipe.start_trial(onset)
    dec = None
    while dec is None:
        pipe.pump()
        t0 = time.perf_counter()
        dec = pipe.update()
        if timing is not None:
            dt = (time.perf_counter() - t0) * 1e3
            timing["ms"].append(dt)
            timing["max"] = max(timing["max"], dt)
        if realtime:
            time.sleep(0.005)
    while src.now() < onset + st["stim_s"]:
        pipe.pump()
        if realtime:
            time.sleep(0.005)
    return dec, pipe.last_history


def calibrate(
    pipe: OnlinePipeline,
    src: SyntheticSource,
    freqs: list[float],
    st: dict,
    n_per_target: int,
    realtime: bool,
) -> dict:
    """Cued calibration block (theta>1 forces full-length windows), then fit T and theta."""
    stop_cfg = pipe.dcfg["stopping"]
    pipe.set_calibration(1.0, 1.01)
    labels = np.repeat(np.arange(len(freqs)), n_per_target)
    np.random.default_rng(2).shuffle(labels)
    scores = []
    for y in labels:
        _, hist = run_trial(pipe, src, freqs[y], st, realtime)
        scores.append([h["scores"] for h in hist])
    fit = fit_policy(np.asarray(scores), labels, pipe.lengths, stop_cfg)
    pipe.set_calibration(fit["temperature"], fit["theta"])
    return {k: v for k, v in fit.items() if k != "theta_table"}


def soak(minutes: float, realtime: bool, cfgs: dict, n_per_target: int = 10) -> dict:
    info = unicorn_info_from_config(cfgs["device"])
    syn = cfgs["synthetic"]
    src = SyntheticSource(info, syn, realtime=realtime)
    freqs = cfgs["stim"]["frequency_sets"][cfgs["stim"]["active_set"]]
    pipe = OnlinePipeline(src, cfgs["decoder"], cfgs["quality"], freqs)
    proc = psutil.Process()
    st = cfgs["stim"]["trial"]
    rng = np.random.default_rng(1)
    src.start()
    calib = calibrate(pipe, src, freqs, st, n_per_target, realtime)
    t_end = src.now() + minutes * 60
    rss, decisions, n_trials = [], [], 0
    timing = {"ms": deque(maxlen=20_000), "max": 0.0}  # bounded: the harness must not leak either
    next_mem = src.now()
    while src.now() < t_end:
        tgt = int(rng.integers(len(freqs)))
        dec, _ = run_trial(pipe, src, freqs[tgt], st, realtime, timing)
        decisions.append((tgt, dec.kind, dec.target, dec.window_s))
        n_trials += 1
        if src.now() >= next_mem:
            rss.append((src.now() - (t_end - minutes * 60), proc.memory_info().rss / 2**20))
            next_mem += 60
    src.stop()
    rss_arr = np.array(rss)
    # growth: compare mean RSS of the last 5 samples to samples 3-7 (after warm-up)
    early = rss_arr[2:7, 1].mean() if len(rss_arr) > 8 else rss_arr[0, 1]
    late = rss_arr[-5:, 1].mean()
    commits = [d for d in decisions if d[1] == COMMIT]
    acc = float(np.mean([d[0] == d[2] for d in commits])) if commits else float("nan")
    cm = np.array(pipe.compute_ms)
    return {
        "minutes_simulated": minutes,
        "realtime": realtime,
        "n_trials": n_trials,
        "commit_rate": len(commits) / max(1, n_trials),
        "accuracy_committed": acc,
        "mean_decision_s_committed": float(np.mean([d[3] for d in commits])) if commits else None,
        "decoder_compute_ms_median": float(np.median(cm)),
        "decoder_compute_ms_p99": float(np.percentile(cm, 99)),
        "update_ms_p99": float(np.percentile(timing["ms"], 99)),
        "update_ms_max": timing["max"],
        "calibration": calib,
        "rss_mb_trace": rss,
        "rss_growth_mb": float(late - early),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--realtime", action="store_true")
    args = ap.parse_args()
    cfgs = {n: load_config(n) for n in ["device", "stim", "decoder", "quality", "synthetic"]}
    freqs = cfgs["stim"]["frequency_sets"][cfgs["stim"]["active_set"]]
    res = {"label": "SYNTHETIC"}
    if not args.realtime:
        res["latency"] = latency_bench(cfgs["decoder"], freqs, cfgs["device"]["srate"])
    t0 = time.time()
    res["soak"] = soak(args.minutes, args.realtime, cfgs)
    res["soak"]["wall_s"] = time.time() - t0
    out = REPO_ROOT / "results" / f"g1_{'realtime' if args.realtime else 'accelerated'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    s = res["soak"]
    print(json.dumps({k: v for k, v in res.get("latency", {}).items()}, indent=1))
    print(
        f"soak: {s['minutes_simulated']} min ({'wall-clock' if args.realtime else 'virtual clock'}), "
        f"wall {s['wall_s']:.0f}s, trials {s['n_trials']}, commit {s['commit_rate']:.2f}, "
        f"acc(committed) {s['accuracy_committed']:.3f}, decoder p99 {s['decoder_compute_ms_p99']:.1f} ms, "
        f"update max {s['update_ms_max']:.1f} ms, RSS growth {s['rss_growth_mb']:+.1f} MB"
    )
    print(f"-> {Path(out).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
