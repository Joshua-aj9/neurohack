"""Electrode-contact report from an eyes-open / eyes-closed recording. Pure functions.

Answers three measured questions for a live setup (dry vs saline vs gel, different rooms):
  1. How much mains is there, and does it dominate the raw amplitude?  (raw vs decoder-filtered ptp)
  2. Is the noise mostly shared by all channels (reference / ground pads)?  (inter-channel correlation)
  3. Do the occipital channels record EEG at all?  (alpha reactivity: eyes closed vs open)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from neurohack.preprocessing import StreamFilter, car, detrend
from neurohack.quality import assess, psd


def _windows(n: int, size: int) -> list[slice]:
    return [slice(i, i + size) for i in range(0, n - size + 1, size)]


def mean_abs_corr(x: np.ndarray) -> float:
    """Mean |r| over all channel pairs (1 = every channel shows the same waveform)."""
    r = np.corrcoef(x)
    iu = np.triu_indices_from(r, k=1)
    return float(np.nanmean(np.abs(r[iu])))


def segment_band_power(
    x: np.ndarray, fs: float, spans: list[slice], seg: int, band: tuple[float, float]
) -> np.ndarray:
    """Band power per channel for every non-overlapping `seg`-sample segment of `spans`
    -> [n_segments, n_ch]."""
    out = []
    for sp in spans:
        y = x[:, sp]
        for w in _windows(y.shape[1], seg):
            f, p = psd(y[:, w], fs)
            m = (f >= band[0]) & (f <= band[1])
            out.append(p[:, m].sum(axis=1))
    return np.array(out).reshape(-1, x.shape[0])


def alpha_reactivity(
    x: np.ndarray,
    fs: float,
    open_spans: list[slice],
    closed_spans: list[slice],
    seg: int,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """(ratio, p) per channel: median closed / median open alpha power, and the one-sided
    Mann-Whitney U p-value for closed > open."""
    po = segment_band_power(x, fs, open_spans, seg, band)
    pc = segment_band_power(x, fs, closed_spans, seg, band)
    ratio = np.median(pc, axis=0) / np.maximum(np.median(po, axis=0), 1e-12)
    p = np.array(
        [
            stats.mannwhitneyu(pc[:, k], po[:, k], alternative="greater").pvalue
            for k in range(x.shape[0])
        ]
    )
    return ratio, p


def contact_report(
    raw: np.ndarray,
    labels: np.ndarray,
    fs: float,
    ch_names: list[str],
    decision_channels: list[str],
    cc_cfg: dict,
    pre_cfg: dict,
    q_cfg: dict,
) -> dict:
    """raw [n_eeg, n] microvolts; labels [n] in {"settle", "open", "closed"} per sample."""
    keep = labels != "settle"
    filt = StreamFilter.from_config(pre_cfg, fs, raw.shape[0])(raw)  # causal, as the decoder
    x_raw, x_f, lab = raw[:, keep], filt[:, keep], labels[keep]
    x_car = car(x_f)
    n1 = int(q_cfg["window_s"] * fs)

    # 1. quality-monitor metrics over 1 s windows (exactly what the viewer shows), medians
    wins = _windows(x_raw.shape[1], n1)
    reps = [assess(x_raw[:, w], fs, ch_names, q_cfg, filtered=x_f[:, w]) for w in wins]
    q_score = np.median([r.scores for r in reps], axis=0)
    line_uv = np.median([r.metrics["line_noise"] for r in reps], axis=0)  # raw mains RMS, uV
    filt_ptp = np.median([r.metrics["railing"] for r in reps], axis=0)
    raw_ptp = np.median([np.ptp(detrend(x_raw[:, w]), axis=1) for w in wins], axis=0)

    # 2. spans per condition, dropping the reaction time after each cue
    drop = int(cc_cfg["transition_s"] * fs)
    spans: dict[str, list[slice]] = {"open": [], "closed": []}
    edges = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1], True])
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if lab[a] in spans and b - (a + drop) > 0:
            spans[lab[a]].append(slice(a + drop, b))
    seg = int(cc_cfg["segment_s"] * fs)
    band = tuple(cc_cfg["alpha_band_hz"])
    ratio, p = alpha_reactivity(x_f, fs, spans["open"], spans["closed"], seg, band)
    ratio_car, p_car = alpha_reactivity(x_car, fs, spans["open"], spans["closed"], seg, band)

    def reactive(r: np.ndarray, pv: np.ndarray) -> np.ndarray:
        return (r >= cc_cfg["alpha_ratio_min"]) & (pv <= cc_cfg["alpha_p_max"])

    react = reactive(ratio, p) | reactive(ratio_car, p_car)
    di = [ch_names.index(c) for c in decision_channels if c in ch_names]
    corr = mean_abs_corr(x_f)
    hints = []
    if corr >= cc_cfg["common_mode_corr_warn"]:
        hints.append(
            "all channels share one waveform -> reference/ground (ear) pads: clean skin, "
            "press firmly, check the snaps"
        )
    if (
        np.median(raw_ptp) > q_cfg["railing"]["warn"]
        and np.median(filt_ptp) < q_cfg["railing"]["warn"]
    ):
        hints.append(
            "raw amplitude is mostly mains/drift: the decoder's filter removes it "
            "(the 50Hz column is the contact check)"
        )
    poor = [ch_names[i] for i in di if line_uv[i] >= q_cfg["line_noise"]["warn_uv"]]
    if poor:
        hints.append(f"mains above the contact threshold on {', '.join(poor)}: wet / re-seat them")
    if not react[di].any():
        hints.append(
            "no alpha reactivity on the decision channels: occipital contact (hair, saline)"
        )
    per_ch = {
        c: {
            "quality_score": float(q_score[i]),
            "raw_ptp_uv": float(raw_ptp[i]),
            "filtered_ptp_uv": float(filt_ptp[i]),
            "line_50hz_rms_uv": float(line_uv[i]),
            "filtered_rms_uv": float(x_f[i].std()),
            "car_rms_uv": float(x_car[i].std()),
            "alpha_ratio": float(ratio[i]),
            "alpha_p": float(p[i]),
            "alpha_ratio_car": float(ratio_car[i]),
            "alpha_p_car": float(p_car[i]),
            "alpha_reactive": bool(react[i]),
        }
        for i, c in enumerate(ch_names)
    }
    return {
        "channels": per_ch,
        "mean_abs_corr_filtered": corr,
        "mean_abs_corr_car": mean_abs_corr(x_car),
        "n_segments": {k: int(sum((s.stop - s.start) // seg for s in v)) for k, v in spans.items()},
        "reactive_decision_channels": [c for c in decision_channels if per_ch[c]["alpha_reactive"]],
        "eeg_visible": bool(react[di].any()),
        "hints": hints,
    }
