"""G3 cued SSVEP calibration: 4 targets x N trials x (cue, stim, rest). Records EEG + markers, then analyses.

    uv run python apps/ssvep_calibrate.py                         # LIVE: UnicornLSL, 10 trials/target
    uv run python apps/ssvep_calibrate.py --source synthetic      # dry run, simulated SSVEP (SYNTHETIC)
    uv run python apps/ssvep_calibrate.py --set non_alpha         # alternative frequency set

Keys: SPACE = continue after a break, ESC = abort (data so far is saved and analysed).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_session import analyze  # noqa: E402

from neurohack.config import load_config  # noqa: E402
from neurohack.markers import MarkerOutlet, encode  # noqa: E402
from neurohack.recorder import Recorder  # noqa: E402
from neurohack.session import Session, add_electrodes_arg, resolve_electrodes  # noqa: E402
from neurohack.sources.base import unicorn_info_from_config  # noqa: E402
from neurohack.sources.lsl import LSLSource  # noqa: E402
from neurohack.sources.synthetic import SyntheticSource  # noqa: E402
from neurohack.stimulus import Display, cross_layout, grey, luminance  # noqa: E402


def wait_space(disp, font, text: str, lay: dict) -> bool:
    pg = disp.pg
    while True:
        disp.screen.fill(lay["background_rgb"])
        img = font.render(text, True, lay["text_rgb"])
        disp.screen.blit(img, img.get_rect(center=(disp.size[0] // 2, disp.size[1] // 2)))
        disp.flip()
        for e in pg.event.get():
            if e.type == pg.KEYDOWN and e.key == pg.K_SPACE:
                return True
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["lsl", "synthetic"], default="lsl")
    ap.add_argument("--trials-per-target", type=int, default=10)
    ap.add_argument("--set", default=None)
    ap.add_argument("--break-every", type=int, default=10)
    ap.add_argument("--windowed", action="store_true")
    add_electrodes_arg(ap)
    ap.add_argument(
        "--participant", default="joshua", help="participant ID used in the session name"
    )
    args = ap.parse_args()
    stim, dev = load_config("stim"), load_config("device")
    set_name = args.set or stim["active_set"]
    freqs = stim["frequency_sets"][set_name]
    phases = [p * np.pi for p in stim["phases_pi"]]
    tr = stim["trial"]
    lay = stim["layout"]
    label = "SYNTHETIC" if args.source == "synthetic" else "LIVE"
    electrodes = resolve_electrodes(args.electrodes, label == "LIVE")

    disp = Display(stim, windowed=args.windowed)
    ok, msg = disp.check_refresh()
    print(msg)
    if not ok:
        disp.close()
        print("REFUSING TO RUN: refresh rate mismatch (see apps/ssvep_stim.py).")
        sys.exit(2)
    pg = disp.pg

    if args.source == "synthetic":
        src = SyntheticSource(
            unicorn_info_from_config(dev), load_config("synthetic"), realtime=True
        )
    else:
        src = LSLSource(dev)
    src.start()
    sess = Session(
        f"ssvep_calibrate_{set_name}_{args.participant}", label=label, electrodes=electrodes
    )
    rec = Recorder(src, sess.dir)
    rec.start()
    outlet = MarkerOutlet(dev["lsl"]["marker_stream_name"], dev["lsl"]["marker_stream_type"])

    def mark(event: str, t: float, **fields) -> None:
        v = encode(event, **fields)
        rec.add_marker(t, v)
        outlet.outlet.push_sample([v], t)
        sess.log(event, t=t, **fields)

    rects = [
        pg.Rect(*r) for r in cross_layout(*disp.size, lay["target_size_px"], lay["margin_frac"])
    ]
    font = pg.font.SysFont(None, lay["font_size"])
    order = np.repeat(np.arange(len(freqs)), args.trials_per_target)
    np.random.default_rng(int(time.time())).shuffle(order)
    sess.log("protocol", freqs=freqs, order=order.tolist(), trial=tr, set=set_name)
    aborted = False
    if not wait_space(disp, font, "Look at the target marked in red. Press SPACE to start.", lay):
        aborted = True

    def draw(t_stim: float | None, cue: int | None) -> None:
        disp.screen.fill(lay["background_rgb"])
        for j, (r, f, ph) in enumerate(zip(rects, freqs, phases, strict=True)):
            level = 0.5 if t_stim is None else luminance(t_stim, f, ph, stim["mode"])
            disp.screen.fill(grey(level), r)
            if cue == j:
                pg.draw.rect(disp.screen, lay["cue_rgb"], r.inflate(24, 24), width=8)

    for k, tgt in enumerate(order if not aborted else []):
        if (
            k
            and k % args.break_every == 0
            and not wait_space(
                disp, font, f"Break ({k}/{len(order)}). Blink, relax. SPACE to continue.", lay
            )
        ):
            aborted = True
            break
        # cue
        t_end = disp.flip() + tr["cue_s"]
        while disp.flip() < t_end:
            draw(None, int(tgt))
        # stimulation: time zero = first stimulus flip
        t_flip = disp.flip()
        t0 = t_flip + disp.period
        first = True
        while True:
            t = (t_flip + disp.period) - t0
            if t >= tr["stim_s"]:
                break
            draw(t, int(tgt))
            t_flip = disp.flip()
            if first:
                mark("trial_start", t_flip, trial=k, target=int(tgt), freq=freqs[tgt], freqs=freqs)
                if isinstance(src, SyntheticSource):
                    src.set_stimulus(freqs[tgt], phase=phases[tgt], onset=t_flip)
                first = False
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    aborted = True
        mark("trial_end", t_flip, trial=k)
        if isinstance(src, SyntheticSource):
            src.set_stimulus(None)
        if aborted:
            break
        t_end = t_flip + tr["rest_s"]
        while disp.flip() < t_end:
            draw(None, None)
    disp.close()
    time.sleep(0.5)  # let the last samples arrive
    rec.stop()
    src.stop()
    sess.save_results(
        {
            "participant": args.participant,
            "aborted": aborted,
            "n_trials_planned": int(len(order)),
            "freqs": freqs,
            "set": set_name,
        }
    )
    sess.close()
    print(f"saved {sess.dir}")
    analyze(sess.dir, write_calibration=True)


if __name__ == "__main__":
    main()
