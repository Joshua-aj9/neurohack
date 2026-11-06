"""G0 plumbing check: find the UnicornLSL stream, record N minutes, verify rate / gaps / channel map / markers.

    uv run python apps/stream_check.py --list                      # just list LSL streams
    uv run python apps/stream_check.py --minutes 10 --wifi off     # full G0 run (LIVE)
    uv run python apps/stream_check.py --minutes 1 --synthetic     # dry run without the headset (SYNTHETIC)

Writes sessions/<stamp>_stream_check/{eeg.npz, markers.jsonl, events.jsonl, results.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pylsl

from neurohack.config import load_config
from neurohack.markers import MarkerInlet, MarkerOutlet
from neurohack.preprocessing import StreamFilter
from neurohack.quality import assess
from neurohack.recorder import Recorder
from neurohack.session import Session, add_electrodes_arg, resolve_electrodes
from neurohack.sources.base import unicorn_info_from_config
from neurohack.sources.lsl import LSLSource, list_streams, stream_channel_labels
from neurohack.sources.synthetic import SyntheticSource
from neurohack.timing import dejitter, gap_report


def print_streams() -> None:
    streams = list_streams(2.0)
    if not streams:
        print(
            "NO LSL STREAMS FOUND. Start Unicorn Suite -> UnicornLSL -> select device -> Open -> Start."
        )
    for s in streams:
        print(
            f"  name={s.name()!r} type={s.type()!r} n_ch={s.channel_count()} srate={s.nominal_srate()} "
            f"fmt={s.channel_format()} source_id={s.source_id()!r} host={s.hostname()!r}"
        )
        labels = stream_channel_labels(pylsl.StreamInlet(s).info(timeout=2.0))
        if any(labels):
            print(f"    labels: {labels}")


def guess_channel_roles(data: np.ndarray) -> dict:
    """Heuristics to confirm the channel map from data alone (no assumptions about index order)."""
    roles = {}
    for i, x in enumerate(data):
        d = np.diff(x)
        if x.size > 10 and np.mean(d == 1) > 0.99:
            roles[i] = "counter"
        elif np.ptp(x) == 0:
            roles[i] = f"constant({x[0]:g})"
        elif 0 <= x.min() and x.max() <= 100 and np.mean(d == 0) > 0.95:
            roles[i] = "battery?"
        elif np.abs(np.median(x)) < 2.0 and np.std(x) < 1.0:
            roles[i] = "accel? (|x|<2 g)"
        else:
            roles[i] = "eeg/gyro"
    return roles


def analyse(
    data: np.ndarray,
    ts: np.ndarray,
    info,
    markers_sent: list[tuple[float, str]],
    markers_recv: list[tuple[float, str]],
    device_cfg: dict,
    qcfg: dict,
) -> dict:
    fs_nom = device_cfg["srate"]
    roles = guess_channel_roles(data)
    counter_idx = [i for i, r in roles.items() if r == "counter"]
    cfg_counter = device_cfg["channel_index"]["Counter"]
    res: dict = {
        "n_samples": int(ts.size),
        "duration_s": float(ts[-1] - ts[0]),
        "n_channels": int(data.shape[0]),
        "channel_roles_guess": {
            info.ch_names[i] if i < len(info.ch_names) else i: r for i, r in roles.items()
        },
        "counter_index_detected": counter_idx,
        "counter_index_config": cfg_counter,
    }
    res["rate_from_timestamps_hz"] = float((ts.size - 1) / (ts[-1] - ts[0]))
    if counter_idx:
        c = data[counter_idx[0]]
        res["gaps"] = gap_report(c)
        span = ts[-1] - ts[0]
        res["rate_from_counter_hz"] = float((c[-1] - c[0]) / span)
        dj = dejitter(ts, c, fs_nom)
        res["timestamp_jitter_ms_std"] = float(np.std(ts - dj) * 1e3)
        res["dejittered_rate_hz"] = float(1.0 / np.polyfit(c, dj, 1)[0])
    d = np.diff(ts)
    res["ts_interval_ms"] = {
        "median": float(np.median(d) * 1e3),
        "p99": float(np.percentile(d, 99) * 1e3),
        "max": float(d.max() * 1e3),
    }
    eeg_idx = info.picks("eeg")
    eeg = data[eeg_idx]
    res["eeg_stats_uV"] = {
        info.ch_names[i]: {"mean": float(data[i].mean()), "std": float(data[i].std())}
        for i in eeg_idx
    }
    last = int(fs_nom)
    acc, gyr = info.picks("accel"), info.picks("gyro")
    filt = StreamFilter.from_config(load_config("decoder")["preprocessing"], fs_nom, len(eeg_idx))(
        eeg
    )
    q = assess(
        eeg[:, -last:],
        fs_nom,
        [info.ch_names[i] for i in eeg_idx],
        qcfg,
        data[acc, -last:] if acc else None,
        data[gyr, -last:] if gyr else None,
        data[counter_idx[0], -last:] if counter_idx else None,
        filtered=filt[:, -last:],
    )
    res["quality_last_1s"] = q.as_dict()
    # marker round trip: sent (outlet time) vs received (inlet, clock-synced) — same machine => ~0
    sent = {v: t for t, v in markers_sent}
    lags = [(t_r - sent[v]) * 1e3 for t_r, v in markers_recv if v in sent]
    res["markers"] = {
        "sent": len(markers_sent),
        "received": len(lags),
        "lag_ms_max_abs": float(np.max(np.abs(lags))) if lags else None,
        "inside_eeg_span": int(
            sum(ts[0] <= sent[v] <= ts[-1] for _, v in markers_recv if v in sent)
        ),
    }
    return res


def verdict(res: dict, device_cfg: dict) -> tuple[str, list[str]]:
    notes, status = [], "PASS"
    tol = device_cfg["srate_tolerance_hz"]
    rate = res.get("rate_from_counter_hz", res["rate_from_timestamps_hz"])
    if abs(rate - device_cfg["srate"]) > tol:
        status = "FAIL"
        notes.append(f"rate {rate:.3f} Hz outside {device_cfg['srate']}+/-{tol}")
    else:
        notes.append(f"rate {rate:.3f} Hz within {device_cfg['srate']}+/-{tol}")
    if not res["counter_index_detected"]:
        status = "FAIL"
        notes.append("no counter channel detected (split-mode stream? use combined 'Data' stream)")
    else:
        if res["counter_index_detected"][0] != res["counter_index_config"]:
            notes.append(
                f"COUNTER INDEX MISMATCH: detected {res['counter_index_detected']} vs config "
                f"{res['counter_index_config']} -> fix configs/device.yaml channel_index"
            )
            status = "PARTIAL" if status == "PASS" else status
        else:
            notes.append(f"counter at index {res['counter_index_detected'][0]} (matches config)")
        loss = res["gaps"]["loss_frac"]
        if loss > 0.01:
            status = "FAIL"
        elif loss > 0.001 and status == "PASS":
            status = "PARTIAL"
        notes.append(
            f"counter loss {loss * 100:.3f}% ({res['gaps']['n_lost']} samples, "
            f"{res['gaps']['n_gap_events']} gap events, max gap {res['gaps']['max_gap']})"
        )
    m = res["markers"]
    max_lag = device_cfg["lsl"]["marker_max_lag_ms"]
    lag = m["lag_ms_max_abs"]
    if (
        m["received"] < m["sent"]
        or m["inside_eeg_span"] < m["received"]
        or lag is None
        or lag > max_lag
    ):
        status = "FAIL"
    notes.append(
        f"markers: sent {m['sent']} received {m['received']} inside EEG span {m['inside_eeg_span']}, "
        f"max |lag| {lag if lag is None else round(lag, 3)} ms (limit {max_lag} ms)"
    )
    return status, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--wifi", choices=["on", "off", "unknown"], default="unknown")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--marker-every-s", type=float, default=10.0)
    ap.add_argument("--wait-s", type=float, default=60.0, help="max wait for the first EEG sample")
    add_electrodes_arg(ap, required_for_live=False)  # G0 tests plumbing; contact does not matter
    args = ap.parse_args()
    print("LSL streams visible:")
    print_streams()
    if args.list:
        return
    device_cfg, qcfg = load_config("device"), load_config("quality")
    label = "SYNTHETIC" if args.synthetic else "LIVE"
    electrodes = resolve_electrodes(args.electrodes, label == "LIVE", required=False)
    sess = Session("stream_check", label=label, electrodes=electrodes)
    if args.synthetic:
        src = SyntheticSource(
            unicorn_info_from_config(device_cfg), load_config("synthetic"), realtime=True
        )
        src.drop_prob = 0.0002
    else:
        src = LSLSource(device_cfg)
    src.start()
    print(
        f"[{label}] recording from {src.info.name!r} ({src.info.n_channels} ch @ {src.info.srate} Hz) "
        f"for {args.minutes} min, wifi={args.wifi} -> {sess.dir}"
    )
    if src.info.meta:
        print(f"  stream meta: {src.info.meta}")
    # Own marker stream (unique source_id) + run id in every marker value, so a second copy of this
    # script running at the same time cannot cross-talk (seen on Fri 6 Nov: two G0 runs overlapped).
    run_id = f"{os.getpid()}_{int(time.time())}"
    outlet = MarkerOutlet(
        device_cfg["lsl"]["marker_stream_name"] + "_check",
        device_cfg["lsl"]["marker_stream_type"],
        source_id=f"stream_check_{run_id}",
    )
    time.sleep(0.5)
    inlet = MarkerInlet(source_id=outlet.source_id)
    rec = Recorder(src, sess.dir, marker_inlet=inlet)
    rec.start()
    sent: list[tuple[float, str]] = []
    # The clock starts at the first EEG sample, so no sync marker can precede the data.
    t_wait = time.time()
    next_wait_print = t_wait
    while rec.snapshot()[1].size == 0:
        now = time.time()
        if now - t_wait > args.wait_s:
            rec.stop()
            inlet.close()
            src.stop()
            print(
                f"GATE G0 - FAIL   [{label}]  no EEG samples within {args.wait_s:.0f} s "
                "(press Start in UnicornLSL, then re-run)"
            )
            sys.exit(1)
        if now >= next_wait_print:
            print("  waiting for the first EEG sample (is UnicornLSL acquiring?)", flush=True)
            next_wait_print += 5.0
        time.sleep(0.05)
    print(
        f"  data flowing after {time.time() - t_wait:.1f} s; {args.minutes:g}-min clock starts now",
        flush=True,
    )
    t_end = time.time() + args.minutes * 60
    next_marker = time.time() + 2.0
    next_print = time.time() + 10.0
    try:
        while time.time() < t_end:
            now = time.time()
            if now >= next_marker:
                v = json.dumps({"event": "sync", "run": run_id, "i": len(sent)})
                t = pylsl.local_clock()
                outlet.outlet.push_sample([v], t)
                sent.append((t, v))
                next_marker += args.marker_every_s
            if now >= next_print:
                data, ts = rec.snapshot()
                if ts.size > 2:
                    rate = (ts.size - 1) / (ts[-1] - ts[0])
                    print(
                        f"  t={ts[-1] - ts[0]:7.1f}s  samples={ts.size:7d}  ts-rate={rate:7.3f} Hz  "
                        f"markers sent={len(sent)}",
                        flush=True,
                    )
                else:
                    print("  ... no samples yet (is UnicornLSL acquiring?)", flush=True)
                next_print += 10.0
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("interrupted; analysing what was recorded")
    time.sleep(0.5)
    rec.stop()
    inlet.close()
    src.stop()
    data, ts = rec.snapshot()
    if ts.size < 100:
        print("FAIL: fewer than 100 samples recorded")
        sys.exit(1)
    res = analyse(data, ts, src.info, sent, rec._markers, device_cfg, qcfg)
    res["wifi"] = args.wifi
    status, notes = verdict(res, device_cfg)
    res["verdict"] = status
    res["notes"] = notes
    sess.save_results(res)
    sess.close()
    print(
        json.dumps(
            {
                k: res[k]
                for k in [
                    "n_samples",
                    "duration_s",
                    "rate_from_timestamps_hz",
                    "rate_from_counter_hz",
                    "dejittered_rate_hz",
                    "timestamp_jitter_ms_std",
                    "gaps",
                    "counter_index_detected",
                    "counter_index_config",
                    "channel_roles_guess",
                    "markers",
                    "ts_interval_ms",
                    "quality_last_1s",
                ]
                if k in res
            },
            indent=1,
        )
    )
    print(f"\nGATE G0 - {status}   [{label}]  wifi={args.wifi}")
    for n in notes:
        print(f"  - {n}")
    print(f"Evidence: {sess.dir / 'results.json'}")


if __name__ == "__main__":
    main()
