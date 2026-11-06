"""SSVEP communication aid (G6): 4 flickering targets = 3 LLM suggestions + "more", or letter groups.

    uv run python apps/speller/main.py --mode llm                       # LIVE (needs G3 calibration)
    uv run python apps/speller/main.py --mode baseline
    uv run python apps/speller/main.py --mode llm --source synthetic    # demo without headset (SYNTHETIC)
    uv run python apps/speller/main.py --mode baseline --copy "hi"      # copy-spelling: MEASURED accuracy
    uv run python apps/speller/main.py --mode baseline --copy "hi" --cue   # + red cue on the right target

Synthetic source keys: 1-4 or arrow keys = look at top/right/bottom/left target, 0 = look away,
C = toggle jaw clench (EMG artifact -> abstention), M = toggle head motion. ESC = quit (saves session).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from logic import (  # noqa: E402
    Speller,
    acceptable_targets,
    label_for,
    label_lines,
    sentence_metrics,
)

from neurohack.config import REPO_ROOT, load_config  # noqa: E402
from neurohack.decoders.calibration import fit_null_floor, fit_policy  # noqa: E402
from neurohack.decoders.stopping import ABSTAIN_QUALITY, COMMIT  # noqa: E402
from neurohack.llm import LLMClient  # noqa: E402
from neurohack.markers import MarkerOutlet  # noqa: E402
from neurohack.pipeline import OnlinePipeline  # noqa: E402
from neurohack.session import Session, add_electrodes_arg, resolve_electrodes  # noqa: E402
from neurohack.sources.base import unicorn_info_from_config  # noqa: E402
from neurohack.sources.lsl import LSLSource  # noqa: E402
from neurohack.sources.synthetic import SyntheticSource  # noqa: E402
from neurohack.stimulus import Display, cross_layout, frame_stats, grey, luminance  # noqa: E402
from neurohack.worker import DecoderWorker  # noqa: E402


def synthetic_calibration(cfgs: dict, freqs: list[float], n_per_target: int) -> dict:
    """Fit T/theta on simulated cued trials (virtual clock, a few seconds). SYNTHETIC."""
    info = unicorn_info_from_config(cfgs["device"])
    src = SyntheticSource(info, cfgs["synthetic"], realtime=False)
    pipe = OnlinePipeline(src, cfgs["decoder"], cfgs["quality"], freqs)
    pipe.set_calibration(1.0, 1.01)
    src.start()
    labels = np.repeat(np.arange(len(freqs)), n_per_target)
    scores = []
    for y in labels:
        src.set_stimulus(None)
        t = src.now() + 1.0
        while src.now() < t:
            pipe.pump()
        onset = src.now()
        src.set_stimulus(freqs[y], onset=onset)
        pipe.start_trial(onset)
        dec = None
        while dec is None:
            pipe.pump()
            dec = pipe.update()
        scores.append([h["scores"] for h in pipe.last_history])
    floor = None
    nf = cfgs["decoder"]["calibration"].get("null_floor")
    if nf and nf.get("enabled"):  # no-flicker stretch -> no-SSVEP floor, as for a LIVE calibration
        src.set_stimulus(None)
        t0 = src.now()
        while src.now() < t0 + nf["min_windows"] * 0.25 + max(nf["lengths"]) + 1.0:
            pipe.pump()
        rest = {}
        for L in nf["lengths"]:
            ws = [pipe.window(t0 + k * 0.25, L) for k in range(nf["min_windows"])]
            rest[float(L)] = np.array([pipe.decoder.scores(w).max() for w in ws if w is not None])
        floor = fit_null_floor(rest, nf["quantile"], nf["min_windows"])
    fit = fit_policy(
        np.asarray(scores), labels, pipe.lengths, cfgs["decoder"]["stopping"], floor=floor
    )
    return {
        "label": "SYNTHETIC",
        "channels": cfgs["decoder"]["channels"],
        "freqs": freqs,
        "temperature": fit["temperature"],
        "theta": fit["theta"],
        "null_floor": floor,
    }


class App:
    def __init__(self, args) -> None:
        self.args = args
        self.cfgs = {
            n: load_config(n)
            for n in ["device", "stim", "decoder", "quality", "llm", "speller", "synthetic"]
        }
        c, st = self.cfgs["speller"], self.cfgs["stim"]
        self.sc = c
        self.freqs = st["frequency_sets"][args.set or st["active_set"]]
        self.phases = [p * np.pi for p in st["phases_pi"]]
        self.synthetic = args.source == "synthetic"
        self.label = "SYNTHETIC" if self.synthetic else "LIVE"
        self.electrodes = resolve_electrodes(args.electrodes, not self.synthetic)

        calib_path = REPO_ROOT / c["calibration_file"]
        if self.synthetic:
            calib = synthetic_calibration(
                self.cfgs, self.freqs, c["synthetic_calibration_trials_per_target"]
            )
        elif calib_path.exists():
            calib = json.loads(calib_path.read_text(encoding="utf-8"))
            if calib.get("label") != "LIVE" or [float(f) for f in calib["freqs"]] != [
                float(f) for f in self.freqs
            ]:
                raise SystemExit(
                    f"{calib_path} is {calib.get('label')} / freqs {calib['freqs']} - "
                    f"run apps/ssvep_calibrate.py (LIVE, set {self.freqs}) first"
                )
        else:
            raise SystemExit(f"no {calib_path}: run apps/ssvep_calibrate.py first (G3)")
        self.calib = calib
        cal_el = calib.get("electrodes", "unknown")
        if not self.synthetic and cal_el != self.electrodes:
            print(
                f"WARNING: calibration was recorded with electrodes={cal_el}, this run is "
                f"{self.electrodes}. Recalibrate (apps/ssvep_calibrate.py) for reliable T/theta."
            )
        dcfg = dict(self.cfgs["decoder"], channels=calib["channels"])
        info_src = (
            SyntheticSource(
                unicorn_info_from_config(self.cfgs["device"]), self.cfgs["synthetic"], realtime=True
            )
            if self.synthetic
            else LSLSource(self.cfgs["device"])
        )
        self.src = info_src
        if not self.synthetic:
            self.src.start()  # resolve now so the pipeline sees the real channel layout
            self.src.start = lambda: None  # worker.start() must not re-open
        self.pipe = OnlinePipeline(self.src, dcfg, self.cfgs["quality"], self.freqs, record=True)
        if not calib.get("null_floor"):
            print(
                "WARNING: calibration has no no-SSVEP floor: the speller will commit on noise. "
                "Re-run scripts/analyze_session.py on the calibration session."
            )
        self.pipe.set_calibration(calib["temperature"], calib["theta"], calib.get("null_floor"))
        self.dec_channels = list(calib["channels"])
        self.goal = args.copy.lower().strip() if args.copy else None
        if self.goal is not None and not set(self.goal) <= set(c["letters"] + " "):
            raise SystemExit(f"--copy may only contain {c['letters']!r} and spaces")
        self.cue = bool(args.cue)
        if self.cue and self.goal is None:
            raise SystemExit("--cue needs --copy (the cue marks the target that spells the phrase)")
        self.worker = DecoderWorker(self.pipe)

        self.speller = Speller(args.mode, c["letters"], c["return_to_suggestions_after_letter"])
        self.llm = LLMClient(self.cfgs["llm"]) if args.mode == "llm" else None
        self.pool = cf.ThreadPoolExecutor(1)
        self.fut: cf.Future | None = None
        self.sess = Session(
            f"speller_{args.mode}_{args.participant}", label=self.label, electrodes=self.electrodes
        )
        self.sess.log("calibration", **calib)
        self.outlet = MarkerOutlet(self.cfgs["device"]["lsl"]["marker_stream_name"] + "_speller")
        self.contexts = [args.context] if args.context else c["contexts"]
        self.ctx_i = 0
        self.results: list[dict] = []
        self.gaze: int | None = None
        self.stim_flips: list[float] = []
        self.flip_runs: list[list[float]] = []
        self._label_cache: dict = {}
        self._reset_counters()

    # ---------------------------------------------------------------------------------
    def _reset_counters(self) -> None:
        self.t_first: float | None = None
        self.n = {"commit": 0, "undo": 0, "abstain": 0, "timeout": 0, "correct": 0}

    @property
    def context(self) -> str:
        return self.contexts[self.ctx_i % len(self.contexts)]

    def request_suggestions(self) -> None:
        if self.llm is not None and self.speller.needs_suggestions():
            self.fut = self.pool.submit(self.llm.suggest, self.context, self.speller.text, 3)

    def handle_keys(self, pg) -> bool:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                return False
            if e.type == pg.KEYDOWN and self.synthetic:
                keymap = {
                    pg.K_1: 0,
                    pg.K_2: 1,
                    pg.K_3: 2,
                    pg.K_4: 3,
                    pg.K_UP: 0,
                    pg.K_RIGHT: 1,
                    pg.K_DOWN: 2,
                    pg.K_LEFT: 3,
                }
                if e.key in keymap:
                    self.gaze = keymap[e.key]
                elif e.key == pg.K_0:
                    self.gaze = None
                elif e.key == pg.K_c:
                    self.src.artifacts ^= {"emg"}
                elif e.key == pg.K_m:
                    self.src.artifacts ^= {"motion"}
        return True

    # ---------------------------------------------------------------------------------
    def run(self) -> None:
        st, ui, tm = self.cfgs["stim"], self.sc["ui"], self.sc["timing"]
        disp = Display(st, windowed=self.args.windowed)
        ok, msg = disp.check_refresh()
        print(msg)
        if not ok:
            disp.close()
            raise SystemExit("refresh rate mismatch (see apps/ssvep_stim.py)")
        pg = disp.pg
        W, H = disp.size
        rects = [
            pg.Rect(*r)
            for r in cross_layout(W, H, st["layout"]["target_size_px"], st["layout"]["margin_frac"])
        ]
        f_text = pg.font.SysFont("segoeui", ui["text_font_size"])
        f_lab = pg.font.SysFont("segoeui", ui["label_font_size"], bold=True)
        f_small = pg.font.SysFont("segoeui", ui["small_font_size"])
        self.worker.start()
        phase, t_phase_end, msg_text, chosen = "fetch", 0.0, "", None
        self.request_suggestions()
        t_flip = disp.flip()
        t0_stim = 0.0
        running = True
        t_stop = t_flip + 60 * self.args.max_minutes if self.args.max_minutes else float("inf")
        while running:
            running = self.handle_keys(pg) and t_flip < t_stop
            if self.worker.error:
                raise self.worker.error
            now = t_flip + disp.period
            # ---- phase transitions -------------------------------------------------------
            if phase == "fetch":
                if self.fut is None or self.fut.done():
                    if self.fut is not None:
                        s = self.fut.result()
                        self.speller.set_suggestions(s.items)
                        self.sess.log(
                            "llm",
                            source=s.source,
                            latency_s=s.latency_s,
                            items=s.items,
                            error=s.error,
                            context=self.context,
                            partial=self.speller.text,
                        )
                        self.fut = None
                    phase, t_phase_end = "gap", now + tm["gap_s"]
            elif phase in ("gap", "feedback", "message") and now >= t_phase_end:
                if phase == "feedback" and self.speller.sent_text is not None:
                    self.finish_sentence(now)
                    phase, t_phase_end = "result", now + tm["result_s"]
                elif phase == "feedback" and self.speller.needs_suggestions():
                    phase = "fetch"  # request already started at commit time (on_decision)
                elif phase in ("feedback", "message"):
                    # static new labels before every flicker, so gaze can settle (see speller.yaml)
                    phase, t_phase_end, chosen = "gap", now + tm["gap_s"], None
                else:
                    phase, t0_stim, chosen = "stim", -1.0, None
            elif phase == "result" and now >= t_phase_end:
                if self.args.once:
                    break
                self.ctx_i += 1
                self.speller.reset()
                self._reset_counters()
                self.request_suggestions()
                phase = "fetch"
            elif phase == "stim":
                while not self.worker.decisions.empty():
                    dec, hist = self.worker.decisions.get_nowait()
                    self.on_decision(dec, hist, now)
                    if self.synthetic:
                        self.src.set_stimulus(None)
                    if dec.kind == COMMIT:
                        chosen = dec.target
                        phase, t_phase_end = "feedback", now + tm["feedback_s"]
                    else:
                        q = self.worker.quality
                        mine = q.reasons_for(self.dec_channels) if q else []
                        reasons = "; ".join(mine[:2]) or "low quality on " + "/".join(
                            self.dec_channels
                        )
                        msg_text = (
                            f"Not selected - signal quality ({reasons})"
                            if dec.kind == ABSTAIN_QUALITY
                            else "No selection - look at one target"
                        )
                        phase, t_phase_end = "message", now + tm["message_s"]
            # ---- draw --------------------------------------------------------------------
            disp.screen.fill(st["layout"]["background_rgb"])
            targets = self.speller.targets()
            cue_j = (
                min(acceptable_targets(self.speller, self.goal))
                if self.cue and phase in ("gap", "stim")
                else None
            )
            for j, (r, f, ph) in enumerate(zip(rects, self.freqs, self.phases, strict=True)):
                if phase == "stim" and t0_stim >= 0:
                    level = luminance(now - t0_stim, f, ph, st["mode"])
                else:
                    level = 0.5
                lab = label_for(targets[j], self.sc["labels"])
                if not lab:
                    level = 0.1
                disp.screen.fill(grey(level), r)
                if phase == "feedback" and chosen == j:
                    pg.draw.rect(disp.screen, ui["highlight_rgb"], r.inflate(24, 24), width=10)
                elif j == cue_j:  # diagnostic cue, drawn exactly like the G3 calibration cue
                    pg.draw.rect(disp.screen, st["layout"]["cue_rgb"], r.inflate(24, 24), width=8)
                img = self.label_surface(
                    pg,
                    f_lab,
                    label_lines(targets[j], self.sc["labels"], ui["label_letters_per_line"]),
                    r.width - 2 * ui["label_pad_px"],
                )
                disp.screen.blit(img, img.get_rect(center=r.center))
                if self.synthetic and self.gaze == j:
                    pg.draw.circle(disp.screen, (80, 160, 255), r.midbottom, 10)
            self.draw_center(disp, f_text, f_small, phase, msg_text)
            self.draw_quality(disp, f_small)
            t_flip = disp.flip()
            if phase == "stim":
                self.stim_flips.append(t_flip)
            elif self.stim_flips and len(self.stim_flips) > 1:
                self.flip_runs.append(self.stim_flips)
                self.stim_flips = []
            else:
                self.stim_flips = []
            if phase == "stim" and t0_stim < 0:
                t0_stim = t_flip  # first stimulus flip = trial onset
                self.worker.start_trial(t_flip)
                self.sess.log(
                    "trial_start",
                    t=t_flip,
                    targets=[label_for(t, self.sc["labels"]) for t in targets],
                    expected=sorted(acceptable_targets(self.speller, self.goal))
                    if self.goal is not None
                    else None,
                    cued=cue_j,
                )
                self.outlet.push(
                    "trial_start",
                    timestamp=t_flip,
                    targets=[label_for(t, self.sc["labels"]) for t in targets],
                )
                if self.t_first is None:
                    self.t_first = t_flip
                if self.synthetic and self.gaze is not None:
                    self.src.set_stimulus(
                        self.freqs[self.gaze], self.phases[self.gaze], onset=t_flip
                    )
        self.worker.stop()
        disp.close()
        if self.pipe.save_recording(self.sess.dir / "eeg.npz"):
            print(f"raw EEG saved: {self.sess.dir / 'eeg.npz'}")
        fstats = self.frame_summary(disp.measured_hz)
        print(f"stimulus frames during decoding: {fstats}")
        self.sess.save_results(
            {
                "mode": self.args.mode,
                "copy_goal": self.goal,
                "cued": self.cue,
                "calibration": self.calib,
                "sentences": self.results,
                "stim_frame_stats": fstats,
            }
        )
        self.sess.close()
        print(json.dumps(self.results, indent=1))
        print(f"[{self.label}] session saved: {self.sess.dir}")

    def frame_summary(self, hz: float) -> dict:
        """Dropped frames / jitter pooled over all stimulation periods (decoder running in parallel)."""
        runs = [np.asarray(r) for r in self.flip_runs + [self.stim_flips] if len(r) > 1]
        if not runs:
            return {}
        stats = [frame_stats(r, hz, self.cfgs["stim"]["timing_test"]["drop_factor"]) for r in runs]
        d = np.concatenate([np.diff(r) for r in runs])
        n_drop = sum(s["n_dropped_frames"] for s in stats)
        n_exp = sum(round(s["duration_s"] * hz) for s in stats)
        return {
            "n_stim_periods": len(runs),
            "n_dropped_frames": int(n_drop),
            "dropped_frac": n_drop / max(1, n_exp),
            "interval_std_ms": float(d.std() * 1e3),
            "interval_max_ms": float(d.max() * 1e3),
        }

    def on_decision(self, dec, hist, now: float) -> None:
        targets = self.speller.targets()
        rec = {
            "kind": dec.kind,
            "target": dec.target,
            "p_max": dec.p_max,
            "window_s": dec.window_s,
            "quality": dec.quality,
            "gaze_synthetic": self.gaze if self.synthetic else None,
            "targets": [label_for(t, self.sc["labels"]) for t in targets],
            "history": hist,
        }
        if dec.kind == COMMIT:
            self.n["commit"] += 1
            if self.goal is not None:
                ok = acceptable_targets(self.speller, self.goal)
                rec.update(expected=sorted(ok), correct=dec.target in ok)
                self.n["correct"] += int(dec.target in ok)
            action = self.speller.select(dec.target)
            if action == "undo":
                self.n["undo"] += 1
            if self.speller.sent_text is None and self.speller.needs_suggestions():
                self.request_suggestions()  # overlaps the LLM call with the feedback highlight
            rec.update(action=action, text=self.speller.text)
        elif dec.kind == ABSTAIN_QUALITY:
            self.n["abstain"] += 1
            q = self.worker.quality
            rec["reasons"] = q.reasons if q else []
            rec["reasons_decoder"] = q.reasons_for(self.dec_channels) if q else []
        else:
            self.n["timeout"] += 1
        self.sess.log("decision", **rec)

    def finish_sentence(self, now: float) -> None:
        m = sentence_metrics(
            self.speller.sent_text or "",
            self.t_first or now,
            now,
            self.n["commit"],
            self.n["undo"],
            self.n["abstain"],
            self.n["timeout"],
            n_correct=self.n["correct"] if self.goal is not None else None,
        )
        m.update(
            copy_goal=self.goal,
            cued=self.cue,
            context=self.context,
            mode=self.args.mode,
            label=self.label,
            participant=self.args.participant,
            electrodes=self.electrodes,
        )
        self.results.append(m)
        self.sess.log("sentence", **m)
        self.last_metrics = m
        print(
            f"[{self.label}] {self.args.mode}{' CUED' if self.cue else ''}: {m['text']!r} {m['wpm']:.2f} wpm, {m['n_commits']} selections, "
            f"{m['n_undo']} undo, accuracy {m['p_est']:.2f} ({m['p_source']}), "
            f"{m['n_abstain']} abstain, {m['n_timeout']} timeout, ITR {m['itr_bits_min']:.1f} bits/min"
        )

    def draw_center(self, disp, f_text, f_small, phase: str, msg_text: str) -> None:
        ui = self.sc["ui"]
        W, H = disp.size
        cx, cy = W // 2, H // 2
        top = (
            (f'Copy{" (cued)" if self.cue else ""}: "{self.goal}"')
            if self.goal is not None
            else f'Caregiver: "{self.context}"'
        )
        ctx = f_small.render(top, True, ui["dim_rgb"])
        disp.screen.blit(ctx, ctx.get_rect(center=(cx, cy - 70)))
        txt = f_text.render((self.speller.text or "") + "_", True, ui["text_rgb"])
        disp.screen.blit(txt, txt.get_rect(center=(cx, cy)))
        if phase == "message":
            m = f_small.render(msg_text, True, ui["warn_rgb"])
            disp.screen.blit(m, m.get_rect(center=(cx, cy + 60)))
        elif phase == "fetch":
            m = f_small.render("thinking...", True, ui["dim_rgb"])
            disp.screen.blit(m, m.get_rect(center=(cx, cy + 60)))
        elif phase == "result":
            lm = self.last_metrics
            m = f_small.render(
                f"sent: {lm['words']} words in {lm['minutes'] * 60:.0f} s = {lm['wpm']:.2f} words/min "
                f"({lm['n_commits']} selections, ITR {lm['itr_bits_min']:.1f} bits/min) "
                f"[{self.label}]",
                True,
                ui["highlight_rgb"],
            )
            disp.screen.blit(m, m.get_rect(center=(cx, cy + 60)))
        mode = f_small.render(
            f"{self.args.mode.upper()} mode  |  {self.label}  |  selections {self.n['commit']}  "
            f"abstain {self.n['abstain']}  timeout {self.n['timeout']}"
            + (
                f"  correct {self.n['correct']}/{self.n['commit']}" if self.goal is not None else ""
            ),
            True,
            ui["dim_rgb"],
        )
        disp.screen.blit(mode, mode.get_rect(center=(cx, cy + 100)))

    def label_surface(self, pg, font, lines: list[str], max_w: int):
        """Outlined, word-wrapped label lines as one surface; cached (rendered once per label, not
        once per 165 Hz frame)."""
        key = (tuple(lines), max_w)
        if key in self._label_cache:
            return self._label_cache[key]
        ui = self.sc["ui"]
        wrapped: list[str] = []
        for line in lines:
            cur = ""
            for word in line.split(" "):
                trial = f"{cur} {word}".strip()
                if cur and font.size(trial)[0] > max_w:
                    wrapped.append(cur)
                    cur = word
                else:
                    cur = trial
            wrapped.append(cur)
        o = ui["label_outline_px"]
        imgs = []
        for line in wrapped:
            fill = font.render(line, True, ui["text_rgb"])
            edge = font.render(line, True, ui["label_outline_rgb"])
            surf = pg.Surface((fill.get_width() + 2 * o, fill.get_height() + 2 * o), pg.SRCALPHA)
            for dx in (-o, 0, o):
                for dy in (-o, 0, o):
                    if dx or dy:
                        surf.blit(edge, (o + dx, o + dy))
            surf.blit(fill, (o, o))
            imgs.append(surf)
        w = max((i.get_width() for i in imgs), default=1)
        h = sum(i.get_height() for i in imgs) or 1
        out = pg.Surface((w, h), pg.SRCALPHA)
        y = 0
        for i in imgs:
            out.blit(i, ((w - i.get_width()) // 2, y))
            y += i.get_height()
        self._label_cache[key] = out
        return out

    def draw_quality(self, disp, f_small) -> None:
        ui, pg = self.sc["ui"], disp.pg
        q = self.worker.quality
        if q is None:
            return
        qmin = self.cfgs["decoder"]["stopping"]["quality_min"]
        good = self.cfgs["quality"]["good_min"]
        s = ui["quality_box_px"]
        x0, y0 = 16, 16
        shown = [
            (n, sc) for n, sc in zip(q.ch_names, q.scores, strict=True) if n in self.dec_channels
        ]
        for i, (name, score) in enumerate(shown):
            col = (
                ui["highlight_rgb"]
                if score >= good
                else (ui["warn_rgb"] if score >= qmin else ui["bad_rgb"])
            )
            r = pg.Rect(x0 + i * (s + 30), y0, s, s)
            pg.draw.rect(disp.screen, col, r)
            lab = f_small.render(name, True, ui["dim_rgb"])
            disp.screen.blit(lab, lab.get_rect(midtop=(r.centerx, r.bottom + 2)))
        for k, reason in enumerate(q.reasons_for(self.dec_channels)[:3]):
            img = f_small.render(reason, True, ui["warn_rgb"])
            disp.screen.blit(img, (x0, y0 + s + 30 + k * 24))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm", "baseline"], default="llm")
    ap.add_argument("--source", choices=["lsl", "synthetic"], default="lsl")
    ap.add_argument("--set", default=None)
    ap.add_argument(
        "--context", default=None, help="caregiver prompt (default: cycle speller.yaml contexts)"
    )
    ap.add_argument("--once", action="store_true", help="stop after one sentence is sent")
    ap.add_argument(
        "--copy",
        default=None,
        help="copy-spelling: phrase to type (lowercase letters/spaces); every selection is scored",
    )
    ap.add_argument(
        "--cue",
        action="store_true",
        help="with --copy: red frame on the target to look at (diagnostic; G3-style cued trials)",
    )
    ap.add_argument("--windowed", action="store_true")
    add_electrodes_arg(ap)
    ap.add_argument(
        "--participant", default="joshua", help="participant ID used in the session name"
    )
    ap.add_argument(
        "--max-minutes", type=float, default=None, help="stop after this long (unattended runs)"
    )
    args = ap.parse_args()
    App(args).run()


if __name__ == "__main__":
    main()
