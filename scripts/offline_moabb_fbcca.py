"""G2: FBCCA vs CCA offline on MOABB Kalunga2016 (formerly SSVEPExo) + synthetic sanity arm.

PUBLIC DATASET: Kalunga2016, 12 subjects, 256 Hz, stimuli 13/17/21 Hz (+ rest, excluded; chance = 1/3).
Onset convention (--onset): "dataset" (default) starts windows at MOABB's Kalunga2016.interval[0] = 2 s
after the event marker (the cue; subjects must shift gaze to the cued LED, the first ~1 s decodes poorly);
"marker" starts at marker + decoder.visual_latency_s. NOTE: "dataset" was adopted after a first pass with
"marker" (2 s acc FBCCA 0.633 / CCA 0.606); both are reported in GATES.md.
Pre-specified primary window = 2.0 s (binomial test and FBCCA-vs-CCA comparison).
Calibrated dynamic stopping is evaluated with 2-fold stratified CV per subject (fit T, theta on one half).

    uv run python scripts/offline_moabb_fbcca.py [--channels PO7 Oz PO8] [--subjects 1 2 3]
"""

from __future__ import annotations

import argparse
import json

import mne
import numpy as np
from sklearn.model_selection import StratifiedKFold

from neurohack.config import REPO_ROOT, frange, load_config
from neurohack.decoders.base import softmax
from neurohack.decoders.calibration import ece, fit_policy, reliability
from neurohack.decoders.cca import CCADecoder
from neurohack.decoders.fbcca import FBCCADecoder
from neurohack.decoders.stopping import simulate, summarize
from neurohack.eval import figures
from neurohack.eval.offline import accuracy_by_length, binomial_p, confusion, scores_by_length
from neurohack.preprocessing import StreamFilter, epochs_from_onsets
from neurohack.sources.base import unicorn_info_from_config
from neurohack.sources.synthetic import SyntheticSource

mne.set_log_level("ERROR")
RES, FIG = REPO_ROOT / "results", REPO_ROOT / "figures"
PRIMARY_S = 2.0
CURVE_LENGTHS = frange(0.5, 4.0, 0.25)


def load_subject(
    ds, subject: int, channels: list[str], pre_cfg: dict, epoch_s: float, offset_s: float
):
    """Causally filtered (same StreamFilter as online) epochs of stimulus trials for one subject."""
    stim = {k: v for k, v in ds.event_id.items() if k != "rest"}
    freqs = sorted(float(k) for k in stim)
    code_to_label = {stim[str(int(f))]: i for i, f in enumerate(freqs)}
    X, y = [], []
    for _, runs in ds.get_data(subjects=[subject])[subject].items():
        for _, raw in runs.items():
            fs = raw.info["sfreq"]
            ev = mne.find_events(raw)
            data = raw.get_data(picks=channels) * 1e6
            data = StreamFilter.from_config(pre_cfg, fs, len(channels))(data)
            keep = np.isin(ev[:, 2], list(code_to_label))
            onsets = ev[keep, 0] - raw.first_samp
            ep = epochs_from_onsets(data, onsets, 0, int(round(epoch_s * fs)))
            X.append(ep)
            y += [code_to_label[c] for c in ev[keep, 2]][: len(ep)]
    return np.concatenate(X), np.asarray(y), freqs, fs


def stopping_cv(
    scores: np.ndarray, y: np.ndarray, lengths: list[float], stop_cfg: dict, seed: int = 0
) -> dict:
    """2-fold CV: fit T + theta on train half, simulate dynamic stopping on the test half."""
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    preds, times, labels, p_raw, p_cal, fits = [], [], [], [], [], []
    for tr, te in skf.split(np.zeros(len(y)), y):
        fit = fit_policy(scores[tr], y[tr], lengths, stop_cfg)
        probs_te = softmax(scores[te] / fit["temperature"])
        pr, tt = simulate(probs_te, y[te], fit["theta"], lengths, stop_cfg["on_timeout"])
        preds.append(pr)
        times.append(tt)
        labels.append(y[te])
        n_len = scores.shape[1]
        p_raw.append(softmax(scores[te].reshape(-1, scores.shape[2])))
        p_cal.append(probs_te.reshape(-1, scores.shape[2]))
        fits.append({"temperature": fit["temperature"], "theta": fit["theta"]})
    pred, tdec, yy = np.concatenate(preds), np.concatenate(times), np.concatenate(labels)
    yy_pooled = np.concatenate([np.repeat(lab, n_len) for lab in labels])
    pr, pc = np.concatenate(p_raw), np.concatenate(p_cal)
    s = summarize(pred, tdec, yy, scores.shape[2], gap_s=1.0)
    return {
        **s,
        "ece_raw": ece(pr, yy_pooled),
        "ece_calibrated_heldout": ece(pc, yy_pooled),
        "folds": fits,
        "_pred": pred,
        "_y": yy,
        "_pr": pr,
        "_pc": pc,
        "_ypool": yy_pooled,
    }


def synthetic_arm(cfgs: dict, n_trials: int = 120) -> dict:
    """SYNTHETIC: accuracy vs window at the configured SNR and an amplitude sweep."""
    info = unicorn_info_from_config(cfgs["device"])
    dc = cfgs["decoder"]
    fs = info.srate
    freqs = cfgs["stim"]["frequency_sets"][cfgs["stim"]["active_set"]]
    eeg = info.picks("eeg")
    names = [info.ch_names[i] for i in eeg]
    di = [names.index(c) for c in dc["channels"]]
    out = {}
    base_amp = cfgs["synthetic"]["ssvep"]["amplitude_uv"]
    for amp in sorted({1.5, 3.0, base_amp}):
        syn = dict(cfgs["synthetic"], ssvep=dict(cfgs["synthetic"]["ssvep"], amplitude_uv=amp))
        src = SyntheticSource(info, syn, realtime=False)
        src.start()
        sf = StreamFilter.from_config(dc["preprocessing"], fs, len(eeg))
        sf(src.generate(int(2 * fs))[0][eeg])
        X, y = [], []
        off = int(round(dc["visual_latency_s"] * fs))
        for i in range(n_trials):
            lab = i % len(freqs)
            src.set_stimulus(freqs[lab], onset=src.now())
            d, _ = src.generate(int(4.5 * fs))
            X.append(sf(d[eeg])[di][:, : off + int(4.0 * fs)])
            y.append(lab)
            src.set_stimulus(None)
            sf(src.generate(int(1.0 * fs))[0][eeg])
        X, y = np.asarray(X), np.asarray(y)
        res = {}
        for name, dec in [
            ("FBCCA", FBCCADecoder(freqs, fs, dc["fbcca"], dc["n_harmonics"])),
            ("CCA", CCADecoder(freqs, fs, dc["n_harmonics"])),
        ]:
            res[name] = accuracy_by_length(
                scores_by_length(dec, X, CURVE_LENGTHS, fs, dc["visual_latency_s"]), y
            ).tolist()
        out[f"{amp:g}uV"] = res
    return {
        "lengths": CURVE_LENGTHS,
        "configured_amplitude_uv": base_amp,
        "n_trials": n_trials,
        "curves": out,
        "chance": 1 / len(freqs),
    }


def main() -> None:
    from moabb.datasets import Kalunga2016

    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--onset", choices=["dataset", "marker"], default="dataset")
    args = ap.parse_args()
    cfgs = {n: load_config(n) for n in ["device", "stim", "decoder", "synthetic"]}
    dc = cfgs["decoder"]
    channels = args.channels or dc["channels"]
    ds = Kalunga2016()
    subjects = args.subjects or ds.subject_list
    lat = float(ds.interval[0]) if args.onset == "dataset" else dc["visual_latency_s"]
    stop_lengths = frange(
        dc["stopping"]["min_window_s"], dc["stopping"]["max_window_s"], dc["stopping"]["step_s"]
    )

    per_subj, acc_f, acc_c = [], [], []
    all_pred, all_y, all_pr, all_pc, all_ypool = [], [], [], [], []
    for s in subjects:
        X, y, freqs, fs = load_subject(ds, s, channels, dc["preprocessing"], lat + 4.0, lat)
        fb = FBCCADecoder(freqs, fs, dc["fbcca"], dc["n_harmonics"])
        cc = CCADecoder(freqs, fs, dc["n_harmonics"])
        sf = scores_by_length(fb, X, CURVE_LENGTHS, fs, lat)
        sc = scores_by_length(cc, X, CURVE_LENGTHS, fs, lat)
        af, ac = accuracy_by_length(sf, y), accuracy_by_length(sc, y)
        acc_f.append(af)
        acc_c.append(ac)
        k = CURVE_LENGTHS.index(PRIMARY_S)
        n = len(y)
        nf, nc = int(round(af[k] * n)), int(round(ac[k] * n))
        s_stop = scores_by_length(fb, X, stop_lengths, fs, lat)
        cv = stopping_cv(s_stop, y, stop_lengths, dc["stopping"])
        all_pred.append(cv.pop("_pred"))
        all_y.append(cv.pop("_y"))
        all_pr.append(cv.pop("_pr"))
        all_pc.append(cv.pop("_pc"))
        all_ypool.append(cv.pop("_ypool"))
        row = {
            "subject": s,
            "n_trials": n,
            "acc_fbcca_2s": af[k],
            "acc_cca_2s": ac[k],
            "p_fbcca_2s": binomial_p(nf, n, 1 / 3),
            "p_cca_2s": binomial_p(nc, n, 1 / 3),
            "acc_fbcca_4s": af[-1],
            "acc_cca_4s": ac[-1],
            "mean_acc_fbcca_all_windows": float(af.mean()),
            "mean_acc_cca_all_windows": float(ac.mean()),
            "dynamic_stopping_cv": cv,
        }
        per_subj.append(row)
        print(
            f"S{s:02d} n={n:3d}  2s: FBCCA {af[k]:.2f} (p={row['p_fbcca_2s']:.1e})  CCA {ac[k]:.2f}  | "
            f"4s: FBCCA {af[-1]:.2f} CCA {ac[-1]:.2f} | stop: acc {cv['accuracy_committed']:.2f} "
            f"commit {cv['commit_rate']:.2f} t {cv['mean_decision_s']:.2f}s ITR {cv['itr_bits_min']:.1f}"
        )

    acc_f, acc_c = np.asarray(acc_f), np.asarray(acc_c)
    n_s = len(subjects)
    beats = sum(r["acc_fbcca_2s"] > r["acc_cca_2s"] for r in per_subj)
    ties = sum(r["acc_fbcca_2s"] == r["acc_cca_2s"] for r in per_subj)
    beats_mean = sum(
        r["mean_acc_fbcca_all_windows"] > r["mean_acc_cca_all_windows"] for r in per_subj
    )
    sig = sum(r["p_fbcca_2s"] < 0.01 for r in per_subj)
    pred, yy = np.concatenate(all_pred), np.concatenate(all_y)
    pr, pc, ypool = np.concatenate(all_pr), np.concatenate(all_pc), np.concatenate(all_ypool)
    summary = {
        "label": "PUBLIC DATASET (Kalunga2016) + SYNTHETIC arm",
        "channels": channels,
        "onset": args.onset,
        "window_offset_s": lat,
        "primary_window_s": PRIMARY_S,
        "chance": 1 / 3,
        "n_subjects": n_s,
        "fbcca_beats_cca_2s": beats,
        "ties_2s": ties,
        "fbcca_beats_cca_mean_over_windows": beats_mean,
        "fbcca_above_chance_p01_2s": sig,
        "mean_acc_fbcca_2s": float(acc_f[:, CURVE_LENGTHS.index(PRIMARY_S)].mean()),
        "mean_acc_cca_2s": float(acc_c[:, CURVE_LENGTHS.index(PRIMARY_S)].mean()),
        "pooled_stopping": summarize(pred, np.zeros(len(pred)), yy, 3, 1.0)
        | {"ece_raw": ece(pr, ypool), "ece_calibrated_heldout": ece(pc, ypool)},
        "curve_lengths": CURVE_LENGTHS,
        "curve_fbcca_mean": acc_f.mean(0).tolist(),
        "curve_cca_mean": acc_c.mean(0).tolist(),
        "per_subject": per_subj,
    }
    summary["pooled_stopping"].pop("mean_decision_s")
    summary["pooled_stopping"].pop("itr_bits_min")
    summary["mean_itr_bits_min_dynamic"] = float(
        np.mean([r["dynamic_stopping_cv"]["itr_bits_min"] for r in per_subj])
    )
    summary["mean_decision_s_dynamic"] = float(
        np.mean([r["dynamic_stopping_cv"]["mean_decision_s"] for r in per_subj])
    )
    summary["synthetic"] = synthetic_arm(cfgs)

    tag = args.tag or (
        ("_" + "-".join(channels) if args.channels else "")
        + ("_marker" if args.onset == "marker" else "")
    )
    lab = f"PUBLIC DATASET Kalunga2016, {n_s} subj, {'/'.join(channels)}"
    figures.accuracy_vs_window(
        CURVE_LENGTHS,
        {"FBCCA": acc_f.mean(0), "CCA": acc_c.mean(0)},
        1 / 3,
        lab,
        FIG / f"g2_acc_vs_window{tag}.png",
        {"FBCCA": acc_f, "CCA": acc_c},
    )
    figures.reliability_plot(
        reliability(pr, ypool),
        reliability(pc, ypool),
        ece(pr, ypool),
        ece(pc, ypool),
        "PUBLIC DATASET Kalunga2016 (held-out)",
        FIG / f"g2_calibration{tag}.png",
    )
    figures.confusion_plot(
        confusion(pred, yy, 3),
        ["13", "17", "21"],
        "PUBLIC DATASET Kalunga2016, dyn. stopping",
        FIG / f"g2_confusion{tag}.png",
    )
    syn = summary["synthetic"]
    key = f"{syn['configured_amplitude_uv']:g}uV"
    figures.accuracy_vs_window(
        CURVE_LENGTHS,
        {f"FBCCA {a}": np.asarray(c["FBCCA"]) for a, c in syn["curves"].items()},
        syn["chance"],
        "SYNTHETIC (amplitude sweep)",
        FIG / "g2_synthetic_acc.png",
    )
    out = RES / f"g2_moabb{tag}.json"
    RES.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    k2, k4 = CURVE_LENGTHS.index(2.0), CURVE_LENGTHS.index(4.0)
    print(
        f"\nFBCCA > CCA at 2s: {beats}/{n_s} (ties {ties}); over mean of windows: {beats_mean}/{n_s}"
    )
    print(
        f"FBCCA above chance p<0.01 at 2s: {sig}/{n_s}; mean acc 2s FBCCA {summary['mean_acc_fbcca_2s']:.3f} "
        f"CCA {summary['mean_acc_cca_2s']:.3f}"
    )
    print(
        f"dyn. stopping pooled held-out: {summary['pooled_stopping']}; mean decision "
        f"{summary['mean_decision_s_dynamic']:.2f}s; mean ITR {summary['mean_itr_bits_min_dynamic']:.1f} bits/min"
    )
    print(
        f"SYNTHETIC {key}: FBCCA 2s {syn['curves'][key]['FBCCA'][k2]:.3f}, 4s {syn['curves'][key]['FBCCA'][k4]:.3f}"
    )
    print(f"-> {out.relative_to(REPO_ROOT)}, figures/g2_*{tag}.png")


if __name__ == "__main__":
    main()
