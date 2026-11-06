"""Live signal-quality viewer (idea B "trust layer" demo): traces, per-channel quality + reasons, spectrum,
and the eyes-open/closed alpha control signal.

    uv run python apps/quality_viewer.py --source lsl
    uv run python apps/quality_viewer.py --source synthetic     # SYNTHETIC demo

Synthetic keys: C jaw clench (EMG), M motion, F flat Oz, L line noise, E eyes closed (alpha up). ESC quits.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from neurohack.buffer import RingBuffer
from neurohack.closedloop import bandpower_feature
from neurohack.config import load_config
from neurohack.preprocessing import StreamFilter
from neurohack.quality import assess, psd
from neurohack.sources.base import unicorn_info_from_config
from neurohack.sources.lsl import LSLSource
from neurohack.sources.synthetic import SyntheticSource

TRACE_S = 5.0
COLORS = {
    "good": (60, 200, 90),
    "warn": (240, 170, 40),
    "bad": (230, 60, 60),
    "dim": (140, 140, 140),
    "text": (230, 230, 230),
    "bg": (12, 12, 16),
    "trace": (120, 180, 255),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["lsl", "synthetic"], default="lsl")
    ap.add_argument("--psd-channel", default="Oz")
    args = ap.parse_args()
    dev, qcfg, dcfg, cl = (
        load_config("device"),
        load_config("quality"),
        load_config("decoder"),
        load_config("closedloop"),
    )
    if args.source == "synthetic":
        src = SyntheticSource(
            unicorn_info_from_config(dev), load_config("synthetic"), realtime=True
        )
        src.ssvep_gain = 0.0
        src.alpha_gain = load_config("closedloop")["synthetic_test"][
            "rest_alpha_gain"
        ]  # start eyes open
    else:
        src = LSLSource(dev)
    src.start()
    info = src.info
    fs = info.srate
    eeg = info.picks("eeg")
    names = [info.ch_names[i] for i in eeg]
    acc, gyr, cnt = info.picks("accel"), info.picks("gyro"), info.picks("counter")
    raw = RingBuffer(info.n_channels, int(30 * fs))
    filt = RingBuffer(len(eeg), int(30 * fs))
    sf = StreamFilter.from_config(dcfg["preprocessing"], fs, len(eeg))
    fi = [names.index(c) for c in cl["feature"]["channels"]]
    psd_i = names.index(args.psd_channel)

    import pygame as pg

    pg.init()
    W, H = 1400, 860
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption(
        f"neurohack quality viewer [{'SYNTHETIC' if args.source == 'synthetic' else 'LIVE'}]"
    )
    font = pg.font.SysFont("segoeui", 18)
    big = pg.font.SysFont("segoeui", 26, bold=True)
    clock = pg.time.Clock()
    qmin, good = dcfg["stopping"]["quality_min"], qcfg["good_min"]
    alpha_hist: list[float] = []
    running = True
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                running = False
            if e.type == pg.KEYDOWN and isinstance(src, SyntheticSource):
                m = {pg.K_c: "emg", pg.K_m: "motion", pg.K_f: "flat", pg.K_l: "line"}
                if e.key in m:
                    src.artifacts ^= {m[e.key]}
                if e.key == pg.K_e:
                    src.alpha_gain = 3.0 if src.alpha_gain < 1 else 0.3
        data, ts = src.read()
        if ts.size:
            raw.append(data, ts)
            filt.append(sf(data[eeg]), ts)
        screen.fill(COLORS["bg"])
        nq = int(qcfg["window_s"] * fs)
        if raw.n_available >= nq:
            r, _ = raw.latest(nq)
            q = assess(
                r[eeg],
                fs,
                names,
                qcfg,
                r[acc] if acc else None,
                r[gyr] if gyr else None,
                r[cnt[0]] if cnt else None,
            )
            x, _ = filt.latest(int(TRACE_S * fs))
            row_h = 70
            for k, name in enumerate(names):
                y0 = 20 + k * row_h
                s = q.scores[k]
                col = (
                    COLORS["good"]
                    if s >= good
                    else (COLORS["warn"] if s >= qmin else COLORS["bad"])
                )
                pg.draw.rect(screen, col, (20, y0 + 10, 18, row_h - 25))
                screen.blit(font.render(f"{name} {s:.2f}", True, COLORS["text"]), (45, y0 + 20))
                tr = x[k, ::2]
                tr = tr - np.median(tr)
                scale = (row_h / 2 - 5) / max(30.0, float(np.percentile(np.abs(tr), 99)))
                pts = [
                    (150 + i * (700 / max(1, len(tr) - 1)), y0 + row_h / 2 - v * scale)
                    for i, v in enumerate(tr)
                ]
                if len(pts) > 1:
                    pg.draw.lines(screen, COLORS["trace"], False, pts, 1)
            for k, reason in enumerate(q.reasons[:6] or ["all channels OK"]):
                screen.blit(
                    font.render(reason, True, COLORS["warn"] if q.reasons else COLORS["good"]),
                    (20, 600 + k * 24),
                )
            # spectrum of the chosen channel (last 2 s)
            seg, _ = filt.latest(int(2 * fs))
            f, p = psd(seg[psd_i], fs)
            m = (f >= 1) & (f <= 45)
            lp = np.log10(p[m] + 1e-6)
            lp = (lp - lp.min()) / max(1e-6, np.ptp(lp))
            bw = 480 / m.sum()
            for i, (fr, v) in enumerate(zip(f[m], lp, strict=True)):
                pg.draw.rect(
                    screen, COLORS["trace"], (890 + i * bw, 330 - v * 280, max(1, bw - 1), v * 280)
                )
                if int(fr) % 10 == 0 and abs(fr - round(fr)) < 1e-6:
                    screen.blit(font.render(f"{fr:.0f}", True, COLORS["dim"]), (890 + i * bw, 335))
            screen.blit(
                font.render(f"{args.psd_channel} spectrum (log, 1-45 Hz)", True, COLORS["text"]),
                (890, 20),
            )
            # alpha control signal (idea B)
            n_feat = int(cl["feature"]["window_s"] * fs)
            xa, _ = filt.latest(n_feat)
            a = bandpower_feature(
                xa[fi], fs, tuple(cl["feature"]["band_hz"]), tuple(cl["feature"]["ref_band_hz"])
            )
            alpha_hist = (alpha_hist + [a])[-300:]
            on = a > cl["controller"]["threshold_on"]
            screen.blit(
                big.render(
                    f"relative alpha {a:.2f}  ->  {'EYES CLOSED' if on else 'eyes open'}",
                    True,
                    COLORS["good"] if on else COLORS["text"],
                ),
                (890, 400),
            )
            pts = [(890 + i * 480 / 300, 640 - v * 200) for i, v in enumerate(alpha_hist)]
            if len(pts) > 1:
                pg.draw.lines(screen, COLORS["trace"], False, pts, 2)
            for thr, c in [
                (cl["controller"]["threshold_on"], COLORS["good"]),
                (cl["controller"]["threshold_off"], COLORS["dim"]),
            ]:
                pg.draw.line(screen, c, (890, 640 - thr * 200), (1370, 640 - thr * 200), 1)
            lbl = "SYNTHETIC" if isinstance(src, SyntheticSource) else "LIVE"
            screen.blit(
                font.render(
                    f"[{lbl}] keys: C clench  M motion  F flat Oz  L mains  E eyes"
                    if lbl == "SYNTHETIC"
                    else f"[{lbl}] {info.meta.get('lsl_name', '')}",
                    True,
                    COLORS["dim"],
                ),
                (20, 830),
            )
        pg.display.flip()
        clock.tick(30)
        time.sleep(0.001)
    src.stop()
    pg.quit()


if __name__ == "__main__":
    main()
