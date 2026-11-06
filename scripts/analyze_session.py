"""Analyse a cued SSVEP calibration session (G3): accuracy vs window, PSD at Oz, calibration fit.

    uv run python scripts/analyze_session.py sessions/<stamp>_ssvep_calibrate
    uv run python scripts/analyze_session.py <session> --no-write-calibration

Writes <session>/analysis.json, <session>/calibration.json, figures into <session>/ and figures/g3_*.png,
and (unless --no-write-calibration) results/user_calibration.json which the speller loads.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy import signal

from neurohack.config import REPO_ROOT, frange, load_config
from neurohack.decoders.base import softmax
from neurohack.decoders.calibration import ece, fit_policy, reliability
from neurohack.decoders.cca import CCADecoder
from neurohack.decoders.fbcca import FBCCADecoder
from neurohack.decoders.stopping import simulate, summarize
from neurohack.eval import figures
from neurohack.eval.offline import accuracy_by_length, binomial_p, confusion, scores_by_length
from neurohack.markers import decode
from neurohack.preprocessing import StreamFilter
from neurohack.timing import dejitter

CURVE = frange(0.5, 4.0, 0.25)


def load_session(sdir: Path) -> dict:
    z = np.load(sdir / "eeg.npz", allow_pickle=False)
    cfg = {
        p.stem: yaml.safe_load(p.read_text(encoding="utf-8"))
        for p in (sdir / "config").glob("*.yaml")
    }
    markers = [
        json.loads(line)
        for line in (sdir / "markers.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    res = (
        json.loads((sdir / "results.json").read_text(encoding="utf-8"))
        if (sdir / "results.json").exists()
        else {}
    )
    return {
        "data": z["data"],
        "ts": z["timestamps"],
        "srate": float(z["srate"]),
        "ch_names": [str(c) for c in z["ch_names"]],
        "ch_types": [str(c) for c in z["ch_types"]],
        "cfg": cfg,
        "markers": markers,
        "label": res.get("label", "LIVE"),
        "electrodes": res.get("electrodes", "unknown"),
    }


def epochs(
    sess: dict, offset_s: float, length_s: float
) -> tuple[np.ndarray, np.ndarray, list[float], np.ndarray]:
    """Filtered EEG epochs [n, n_eeg, len] starting at trial_start + offset, labels, freqs, eeg filtered."""
    data, ts, fs = sess["data"], sess["ts"], sess["srate"]
    if "counter" in sess["ch_types"]:
        ts = dejitter(ts, data[sess["ch_types"].index("counter")], fs)
    eeg_idx = [i for i, t in enumerate(sess["ch_types"]) if t == "eeg"]
    filt = StreamFilter.from_config(sess["cfg"]["decoder"]["preprocessing"], fs, len(eeg_idx))(
        data[eeg_idx]
    )
    trials = [(m["t"], decode(m["value"])) for m in sess["markers"]]
    trials = [(t, d) for t, d in trials if d.get("event") == "trial_start"]
    freqs = trials[0][1]["freqs"] if trials else []
    n = int(round(length_s * fs))
    X, y = [], []
    for t, d in trials:
        i0 = int(np.searchsorted(ts, t + offset_s))
        if i0 + n <= filt.shape[1]:
            X.append(filt[:, i0 : i0 + n])
            y.append(int(d["target"]))
    return np.asarray(X), np.asarray(y), freqs, filt


def snr_at(
    f: np.ndarray, pxx: np.ndarray, f0: float, half_bw: float = 1.0, excl: float = 0.3
) -> float:
    k = np.argmin(np.abs(f - f0))
    neigh = (np.abs(f - f0) <= half_bw) & (np.abs(f - f0) > excl)
    return float(pxx[k] / pxx[neigh].mean())


def analyze(sdir: Path, write_calibration: bool = True) -> dict:
    sdir = Path(sdir)
    sess = load_session(sdir)
    dc = sess["cfg"]["decoder"]
    an = dc.get("analysis") or load_config("decoder")["analysis"]  # older sessions lack the section
    fs = sess["srate"]
    lat = dc["visual_latency_s"]
    X, y, freqs, _ = epochs(sess, lat, 4.0)
    if len(y) == 0:
        raise SystemExit("no complete trials found")
    eeg_names = [n for n, t in zip(sess["ch_names"], sess["ch_types"], strict=True) if t == "eeg"]
    label = sess["label"]
    chance = 1 / len(freqs)
    n_tg = len(freqs)
    sets = {"config": dc["channels"]}
    if "Pz" in eeg_names and "Pz" not in dc["channels"]:
        sets["config+Pz"] = ["Pz", *dc["channels"]]
    electrodes = sess["electrodes"]
    out: dict = {
        "label": label,
        "electrodes": electrodes,
        "n_trials": int(len(y)),
        "freqs": freqs,
        "chance": chance,
        "channel_sets": {},
    }
    stop_lengths = frange(
        dc["stopping"]["min_window_s"], dc["stopping"]["max_window_s"], dc["stopping"]["step_s"]
    )
    curves, best = {}, None
    for name, chans in sets.items():
        idx = [eeg_names.index(c) for c in chans]
        Xc = X[:, idx]
        fb = FBCCADecoder(freqs, fs, dc["fbcca"], dc["n_harmonics"])
        cc = CCADecoder(freqs, fs, dc["n_harmonics"])
        af = accuracy_by_length(scores_by_length(fb, Xc, CURVE, fs), y)
        ac = accuracy_by_length(scores_by_length(cc, Xc, CURVE, fs), y)
        curves[f"FBCCA {name}"] = af
        curves[f"CCA {name}"] = ac
        le3 = [i for i, L in enumerate(CURVE) if L <= an["pass_max_window_s"]]
        k3 = le3[int(np.argmax(af[le3]))]
        s_stop = scores_by_length(fb, Xc, stop_lengths, fs)
        # held-out estimate (2-fold) + final fit on all trials
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(y))
        halves = [perm[: len(y) // 2], perm[len(y) // 2 :]]
        preds, times, ys = [], [], []
        for a, b in [(0, 1), (1, 0)]:
            fit = fit_policy(s_stop[halves[a]], y[halves[a]], stop_lengths, dc["stopping"])
            pr, tt = simulate(
                softmax(s_stop[halves[b]] / fit["temperature"]),
                y[halves[b]],
                fit["theta"],
                stop_lengths,
                dc["stopping"]["on_timeout"],
            )
            preds.append(pr)
            times.append(tt)
            ys.append(y[halves[b]])
        pred, tdec, yy = np.concatenate(preds), np.concatenate(times), np.concatenate(ys)
        heldout = summarize(pred, tdec, yy, n_tg, gap_s=1.0)
        final = fit_policy(s_stop, y, stop_lengths, dc["stopping"])
        k2 = CURVE.index(2.0)
        row = {
            "channels": chans,
            "acc_fbcca": dict(zip(map(str, CURVE), af.round(3).tolist(), strict=True)),
            "acc_cca": dict(zip(map(str, CURVE), ac.round(3).tolist(), strict=True)),
            "best_fbcca_le3s": {
                "window_s": CURVE[k3],
                "acc": float(af[k3]),
                "p_binom": binomial_p(int(round(af[k3] * len(y))), len(y), chance),
            },
            "acc_fbcca_2s": float(af[k2]),
            "dynamic_stopping_heldout": heldout,
            "final_fit": {k: v for k, v in final.items() if k != "theta_table"},
            "confusion_heldout": confusion(pred, yy, n_tg).tolist(),
        }
        out["channel_sets"][name] = row
        # choose by held-out ITR, tie-break by 2 s accuracy
        key = (heldout["itr_bits_min"], float(af[k2]))
        if best is None or key > best[0]:
            best = (key, name, final, s_stop, pred, yy)
    _, best_name, best_fit, s_stop, pred, yy = best
    out["selected_channel_set"] = best_name
    calib = {
        "label": label,
        "electrodes": electrodes,
        "session": sdir.name,
        "channels": sets[best_name],
        "freqs": freqs,
        "temperature": best_fit["temperature"],
        "theta": best_fit["theta"],
        "ece_before": best_fit["ece_before"],
        "ece_after": best_fit["ece_after"],
    }
    out["calibration"] = calib

    # PSD at Oz per target (stimulus period only)
    oz = eeg_names.index("Oz") if "Oz" in eeg_names else 0
    pxx_by, snrs = {}, {}
    for j, f0 in enumerate(freqs):
        ep = X[y == j, oz]
        if len(ep) == 0:
            continue
        f, p = signal.periodogram(ep, fs=fs, window="hann", axis=-1)  # full-epoch resolution
        pm = p.mean(axis=0)
        pxx_by[f"target {f0:g} Hz"] = pm
        snrs[f"{f0:g}"] = snr_at(f, pm, f0, an["snr_half_bw_hz"], an["snr_exclude_hz"])
    out["oz_snr_at_stim_freq"] = snrs
    out["oz_clear_peaks"] = int(sum(v >= an["snr_min"] for v in snrs.values()))

    cfg_row = out["channel_sets"]["config"]
    best_acc = max(r["best_fbcca_le3s"]["acc"] for r in out["channel_sets"].values())
    if best_acc >= an["pass_acc"] and out["oz_clear_peaks"] >= n_tg - 1:
        v = "PASS"
    elif best_acc >= an["partial_acc"]:
        v = "PARTIAL"
    else:
        v = "FAIL"
    out["verdict"] = v

    lab = f"{label} ({electrodes}) {sdir.name}"
    figs = [
        figures.accuracy_vs_window(CURVE, curves, chance, lab, sdir / "acc_vs_window.png"),
        figures.psd_plot(f, pxx_by, freqs, f"{label} Oz", sdir / "psd_oz.png"),
        figures.confusion_plot(
            np.asarray(out["channel_sets"][best_name]["confusion_heldout"]),
            [f"{f:g}" for f in freqs],
            f"{label} held-out dyn. stopping",
            sdir / "confusion.png",
        ),
    ]
    pr_raw = softmax(s_stop.reshape(-1, n_tg))
    pr_cal = softmax(s_stop.reshape(-1, n_tg) / best_fit["temperature"])
    ypool = np.repeat(y, s_stop.shape[1])
    figs.append(
        figures.reliability_plot(
            reliability(pr_raw, ypool),
            reliability(pr_cal, ypool),
            ece(pr_raw, ypool),
            ece(pr_cal, ypool),
            f"{label} (in-sample)",
            sdir / "calibration.png",
        )
    )
    live = label == "LIVE"  # only LIVE sessions are exported to figures/ and results/
    if live:
        for p in figs:
            shutil.copy2(p, REPO_ROOT / "figures" / f"g3_{p.name}")
    (sdir / "analysis.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    (sdir / "calibration.json").write_text(
        json.dumps(calib, indent=2, default=float), encoding="utf-8"
    )
    if write_calibration and live:
        (REPO_ROOT / "results").mkdir(exist_ok=True)
        (REPO_ROOT / "results" / "user_calibration.json").write_text(
            json.dumps(calib, indent=2, default=float), encoding="utf-8"
        )
    print(f"[{label}, electrodes={electrodes}] {len(y)} trials, freqs {freqs}")
    for name, r in out["channel_sets"].items():
        h = r["dynamic_stopping_heldout"]
        print(
            f"  {name:10s} {r['channels']}: FBCCA 2s {r['acc_fbcca_2s']:.2f}, best<=3s "
            f"{r['best_fbcca_le3s']['acc']:.2f}@{r['best_fbcca_le3s']['window_s']}s "
            f"(p={r['best_fbcca_le3s']['p_binom']:.1e}) | held-out stopping acc {h['accuracy_committed']:.2f} "
            f"commit {h['commit_rate']:.2f} t {h['mean_decision_s']:.2f}s ITR {h['itr_bits_min']:.1f}"
        )
    print(f"  Oz SNR at stim freqs: { {k: round(v, 2) for k, v in snrs.items()} }")
    print(
        f"  selected: {best_name}  T={calib['temperature']:.4f} theta={calib['theta']:.3f} "
        f"ECE {calib['ece_before']:.3f}->{calib['ece_after']:.3f}"
    )
    print(
        f"\nGATE G3 - {v}   [{label}]   (config-channel best<=3s acc {cfg_row['best_fbcca_le3s']['acc']:.2f})"
    )
    print(f"Evidence: {sdir / 'analysis.json'} ; figures {sdir}/*.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--no-write-calibration", action="store_true")
    a = ap.parse_args()
    analyze(a.session, not a.no_write_calibration)


if __name__ == "__main__":
    main()
