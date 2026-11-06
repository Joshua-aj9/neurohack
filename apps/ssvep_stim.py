"""G4 stimulus timing test: 4 flickering targets, fullscreen vsync, log every flip, report drops/jitter.

    uv run python apps/ssvep_stim.py                 # 60 s timing test on the main display (LIVE)
    uv run python apps/ssvep_stim.py --seconds 10 --windowed   # quick look (not a valid G4 run)

Refuses to run if the measured refresh rate differs from configs/stim.yaml. ESC aborts.
Pushes an LSL marker on the first flip of each trial (stim_start / stim_end).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from neurohack.config import load_config
from neurohack.markers import MarkerOutlet
from neurohack.session import Session
from neurohack.stimulus import Display, cross_layout, frame_stats, grey, luminance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument(
        "--set", default=None, help="frequency set name (default: stim.yaml active_set)"
    )
    args = ap.parse_args()
    cfg, dev = load_config("stim"), load_config("device")
    tcfg = cfg["timing_test"]
    seconds = args.seconds or tcfg["duration_s"]
    freqs = cfg["frequency_sets"][args.set or cfg["active_set"]]
    phases = [p * np.pi for p in cfg["phases_pi"]]

    disp = Display(cfg, windowed=args.windowed)
    ok, msg = disp.check_refresh()
    print(msg)
    if not ok:
        disp.close()
        print(
            "REFUSING TO RUN: set the display to the configured refresh rate (Windows Settings > Display > "
            "Advanced display) or change stim.yaml refresh_hz to match, then re-run."
        )
        sys.exit(2)
    pg = disp.pg
    sess = Session("ssvep_stim", label="LIVE")
    outlet = MarkerOutlet(
        dev["lsl"]["marker_stream_name"] + "_stim", dev["lsl"]["marker_stream_type"]
    )
    lay = cfg["layout"]
    rects = [
        pg.Rect(*r) for r in cross_layout(*disp.size, lay["target_size_px"], lay["margin_frac"])
    ]
    font = pg.font.SysFont(None, lay["font_size"])
    labels = [font.render(f"{f:g} Hz", True, lay["text_rgb"]) for f in freqs]

    flips: list[float] = []
    t_flip = disp.flip()
    t0 = t_flip + disp.period  # stimulus time zero = first stimulus flip
    first = True
    aborted = False
    while True:
        t_next = t_flip + disp.period  # predicted time of the flip that will show this frame
        t = t_next - t0
        if t >= seconds:
            break
        for e in pg.event.get():
            if e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE or e.type == pg.QUIT:
                aborted = True
        if aborted:
            break
        disp.screen.fill(lay["background_rgb"])
        for r, f, ph, lab in zip(rects, freqs, phases, labels, strict=True):
            disp.screen.fill(grey(luminance(t, f, ph, cfg["mode"])), r)
            disp.screen.blit(lab, lab.get_rect(midtop=(r.centerx, r.bottom + 8)))
        t_flip = disp.flip()
        flips.append(t_flip)
        if first:
            outlet.push("stim_start", timestamp=t_flip, freqs=freqs)
            sess.log("stim_start", t=t_flip, freqs=freqs)
            first = False
    outlet.push("stim_end")
    disp.close()
    fl = np.asarray(flips)
    np.save(sess.dir / "flip_times.npy", fl)
    st = frame_stats(fl, disp.measured_hz, tcfg["drop_factor"])
    st.update(
        configured_hz=cfg["refresh_hz"],
        api_refresh_hz=disp.api_refresh_hz,
        windowed=args.windowed,
        aborted=aborted,
        freqs=freqs,
        mode=cfg["mode"],
    )
    if (
        st["dropped_frac"] < tcfg["pass_drop_frac"]
        and st["interval_std_ms"] < tcfg["pass_interval_std_ms"]
    ):
        v = "PASS"
    elif st["dropped_frac"] < tcfg["partial_drop_frac"]:
        v = "PARTIAL"
    else:
        v = "FAIL"
    if args.windowed or aborted or st["duration_s"] < 0.95 * tcfg["duration_s"]:
        v += " (NOT A VALID G4 RUN: windowed/aborted/short)"
    st["verdict"] = v
    sess.save_results(st)
    sess.close()
    print(json.dumps(st, indent=1))
    print(f"\nGATE G4 - {v}   [LIVE]")
    print(f"Evidence: {sess.dir / 'results.json'}  (flip_times.npy alongside)")


if __name__ == "__main__":
    main()
