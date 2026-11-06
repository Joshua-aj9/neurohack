"""Contact check: 1 minute eyes open / eyes closed, then a measured per-channel report.

    uv run python apps/contact_check.py --electrodes dry        # LIVE (UnicornLSL running)
    uv run python apps/contact_check.py --electrodes saline     # LIVE, after wetting the electrodes
    uv run python apps/contact_check.py --source synthetic      # SYNTHETIC smoke test (instant)

Cues: ONE beep = close your eyes, TWO beeps = open your eyes. Sit still, relax the jaw.
Run it once per electrode condition and compare the reports (sessions/*_contact_check_*/results.json).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from neurohack.config import load_config
from neurohack.eval.contact import contact_report
from neurohack.session import Session, add_electrodes_arg, resolve_electrodes
from neurohack.sources.base import unicorn_info_from_config
from neurohack.sources.lsl import LSLSource
from neurohack.sources.synthetic import SyntheticSource


def beep(n: int, cc: dict) -> None:
    try:
        import winsound
    except ImportError:  # not Windows
        print("\a" * n, end="", flush=True)
        return
    for _ in range(n):
        winsound.Beep(int(cc["beep_hz"]), int(cc["beep_ms"]))
        time.sleep(cc["beep_gap_ms"] / 1e3)


def schedule(cc: dict) -> list[tuple[str, float]]:
    return [("settle", cc["settle_s"])] + [(b, cc["block_s"]) for b in cc["blocks"]]


def record_live(src: LSLSource, cc: dict) -> tuple[np.ndarray, np.ndarray]:
    chunks, labels = [], []
    for label, dur in schedule(cc):
        print(f"  {label:7s} {dur:4.0f} s", flush=True)
        if label == "closed":
            beep(1, cc)
        elif label == "open" and labels:
            beep(2, cc)
        t_end = time.perf_counter() + dur
        while time.perf_counter() < t_end:
            d, ts = src.read()
            if ts.size:
                chunks.append(d)
                labels.append(np.full(ts.size, label))
            time.sleep(0.01)
    beep(2, cc)
    return np.concatenate(chunks, axis=1), np.concatenate(labels)


def record_synthetic(src: SyntheticSource, cc: dict, st: dict) -> tuple[np.ndarray, np.ndarray]:
    chunks, labels = [], []
    fs = src.info.srate
    for label, dur in schedule(cc):
        src.alpha_gain = st["on_alpha_gain"] if label == "closed" else st["rest_alpha_gain"]
        d, _ = src.generate(int(dur * fs))
        chunks.append(d)
        labels.append(np.full(d.shape[1], label))
    return np.concatenate(chunks, axis=1), np.concatenate(labels)


def print_report(rep: dict, decision: list[str], tag: str) -> None:
    cols = [
        ("quality_score", "qual", "{:5.2f}"),
        ("raw_ptp_uv", "rawPtp", "{:7.0f}"),
        ("filtered_ptp_uv", "filtPtp", "{:7.0f}"),
        ("line_50hz_rms_uv", "50Hz", "{:6.1f}"),
        ("filtered_rms_uv", "filtRMS", "{:7.1f}"),
        ("car_rms_uv", "carRMS", "{:6.1f}"),
        ("alpha_ratio", "aRatio", "{:6.2f}"),
        ("alpha_p", "p", "{:6.3f}"),
        ("alpha_ratio_car", "aRcar", "{:6.2f}"),
        ("alpha_p_car", "pcar", "{:6.3f}"),
    ]
    print(f"\n{tag} per channel (uV; ptp = median over 1 s windows; alpha = closed/open 8-12 Hz)")
    print("ch    " + " ".join(f"{h:>{len(fmt.format(0))}}" for _, h, fmt in cols) + "  alpha")
    for ch, m in rep["channels"].items():
        star = "*" if ch in decision else " "
        vals = " ".join(fmt.format(m[k]) for k, _, fmt in cols)
        print(f"{ch:4s}{star} {vals}  {'YES' if m['alpha_reactive'] else '-'}")
    print("(* = decoder channel)")
    print(
        f"mean |r| between channels: {rep['mean_abs_corr_filtered']:.2f} filtered, "
        f"{rep['mean_abs_corr_car']:.2f} after CAR   segments {rep['n_segments']}"
    )
    for h in rep["hints"]:
        print(f"HINT: {h}")
    verdict = "EEG VISIBLE" if rep["eeg_visible"] else "NO ALPHA REACTIVITY"
    print(
        f"\nCONTACT CHECK - {verdict} on {rep['reactive_decision_channels'] or 'no'} "
        f"decoder channel(s)   [{tag}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["lsl", "synthetic"], default="lsl")
    add_electrodes_arg(ap)
    args = ap.parse_args()
    live = args.source == "lsl"
    electrodes = resolve_electrodes(args.electrodes, live)
    dev, qcfg, dcfg = load_config("device"), load_config("quality"), load_config("decoder")
    cc = qcfg["contact_check"]
    tag = "LIVE" if live else "SYNTHETIC"
    sess = Session("contact_check", label=tag, electrodes=electrodes)
    if live:
        src = LSLSource(dev)
        src.start()
        print(f"stream {src.info.meta.get('lsl_name')}: {sum(d for _, d in schedule(cc)):.0f} s")
        print("ONE beep = CLOSE eyes, TWO beeps = OPEN eyes. Start with eyes open, sit still.")
        data, labels = record_live(src, cc)
        src.stop()
    else:
        src = SyntheticSource(
            unicorn_info_from_config(dev), load_config("synthetic"), realtime=False
        )
        src.start()
        src.ssvep_gain = 0.0
        data, labels = record_synthetic(src, cc, load_config("closedloop")["synthetic_test"])
    eeg = src.info.picks("eeg")
    names = [src.info.ch_names[i] for i in eeg]
    np.savez_compressed(
        sess.dir / "contact_check.npz",
        data=data,
        labels=labels,
        ch_names=src.info.ch_names,
        srate=src.info.srate,
    )
    rep = contact_report(
        data[eeg], labels, src.info.srate, names, dcfg["channels"], cc, dcfg["preprocessing"], qcfg
    )
    sess.save_results(rep)
    sess.close()
    print_report(rep, dcfg["channels"], tag)
    print(f"Evidence: {sess.dir / 'results.json'}")
    if not live:
        print(json.dumps({k: rep[k] for k in ("eeg_visible", "reactive_decision_channels")}))


if __name__ == "__main__":
    main()
