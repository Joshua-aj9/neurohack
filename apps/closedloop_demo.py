"""Closed-loop template: alpha power (Oz/PO7/PO8) -> threshold/hysteresis/refractory -> audio tone.

    uv run python apps/closedloop_demo.py --synthetic-test            # CL gate, SYNTHETIC (silent: dummy audio)
    uv run python apps/closedloop_demo.py --source lsl                # LIVE: baseline/stim blocks from config
    uv run python apps/closedloop_demo.py --source lsl --adapt        # LIVE with person-specific thresholds

Live eyes-closed test: during each block, keep eyes OPEN, then CLOSE them for ~10 s when prompted
(the console prints the block schedule). In stimulation blocks a tone plays when alpha rises; in baseline
blocks triggers are logged as sham (no tone).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from neurohack.actuators.base import make_actuator
from neurohack.closedloop import ProtocolRunner
from neurohack.config import load_config
from neurohack.session import Session, add_electrodes_arg, resolve_electrodes
from neurohack.sources.base import unicorn_info_from_config
from neurohack.sources.lsl import LSLSource
from neurohack.sources.synthetic import SyntheticSource


def evaluate_synthetic(res: dict, st: dict, t_start: float) -> dict:
    """Each alpha episode must produce exactly one trigger within max_onset_delay_s; none outside."""
    trig = [t - t_start for t in res["trigger_times"]]
    episodes = st["alpha_on_s"]
    per_ep, outside = [], []
    for t in trig:
        hit = [i for i, (a, b) in enumerate(episodes) if a <= t <= b]
        (per_ep.append((hit[0], t)) if hit else outside.append(t))
    counts = [sum(1 for i, _ in per_ep if i == k) for k in range(len(episodes))]
    delays = [t - episodes[i][0] for i, t in per_ep]
    lat_ok = res["latency_ms_max"] is not None and res["latency_ms_max"] < st["max_latency_ms"]
    ok = (
        all(c == 1 for c in counts)
        and not outside
        and all(d <= st["max_onset_delay_s"] for d in delays)
        and lat_ok
    )
    return {
        "triggers_rel_s": trig,
        "triggers_per_episode": counts,
        "triggers_outside": outside,
        "onset_delays_s": delays,
        "latency_ok": lat_ok,
        "pass": ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["lsl", "synthetic"], default="synthetic")
    ap.add_argument("--synthetic-test", action="store_true")
    ap.add_argument("--adapt", action="store_true", help="thresholds from the first baseline block")
    ap.add_argument("--actuator", choices=["audio", "visual", "log_only"], default=None)
    add_electrodes_arg(ap)
    args = ap.parse_args()
    cfg, dev = load_config("closedloop"), load_config("device")
    pre, qcfg = load_config("decoder")["preprocessing"], load_config("quality")
    if args.adapt:
        cfg["controller"]["baseline_adapt"] = True
    if args.actuator:
        cfg["actuator"]["type"] = args.actuator
    st = cfg["synthetic_test"]
    synthetic = args.synthetic_test or args.source == "synthetic"
    if args.synthetic_test:
        os.environ.setdefault(
            "SDL_AUDIODRIVER", "dummy"
        )  # measure the play() path without making noise
        blocks, block_s = st["blocks"], st["block_s"]
    else:
        blocks, block_s = cfg["protocol"]["blocks"], cfg["protocol"]["block_s"]
    label = "SYNTHETIC" if synthetic else "LIVE"
    sess = Session(
        "closedloop", label=label, electrodes=resolve_electrodes(args.electrodes, label == "LIVE")
    )
    if synthetic:
        src = SyntheticSource(
            unicorn_info_from_config(dev), load_config("synthetic"), realtime=True
        )
        src.alpha_gain = st["rest_alpha_gain"]
    else:
        src = LSLSource(dev)
    act = make_actuator(cfg["actuator"])
    runner = ProtocolRunner(src, cfg, pre, qcfg, act, sess.log, realtime=True)
    print(
        f"[{label}] blocks {blocks} x {block_s:.0f} s; actuator={cfg['actuator']['type']}; session {sess.dir}"
    )
    if not synthetic:
        print("Eyes OPEN by default; close them for ~10 s in the middle of each block.")

    def on_tick(t_rel: float) -> None:
        if synthetic:
            on = any(a <= t_rel < b for a, b in st["alpha_on_s"])
            src.alpha_gain = st["on_alpha_gain"] if on else st["rest_alpha_gain"]

    import pylsl

    t_start = pylsl.local_clock()
    res = runner.run(blocks, block_s, on_tick)
    res["buffer_latency_ms"] = getattr(act, "buffer_latency_ms", None)
    res["feature_trace"] = [(t["t_data"] - t_start, t["feature"], t["block"]) for t in runner.ticks]
    if args.synthetic_test:
        ev = evaluate_synthetic(res, st, t_start)
        res["evaluation"] = ev
        lat = [t["latency_ms"] for t in runner.triggers]
        print(json.dumps({k: v for k, v in ev.items()}, indent=1))
        print(
            f"latency threshold->actuator (all triggers incl. sham): median {np.median(lat):.1f} ms, "
            f"max {np.max(lat):.1f} ms; audio buffer adds ~{res['buffer_latency_ms']:.1f} ms (+ OS output)"
        )
        print(f"\nGATE CL - {'PASS' if ev['pass'] else 'FAIL'}   [SYNTHETIC]")
    else:
        print(json.dumps({k: v for k, v in res.items() if k != "feature_trace"}, indent=1))
    sess.save_results(res)
    sess.close()
    act.close()
    print(f"Evidence: {sess.dir / 'results.json'}")


if __name__ == "__main__":
    main()
