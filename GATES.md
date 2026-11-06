# GATES.md — live status board

Labels: **SYNTHETIC** | **PUBLIC DATASET** | **LIVE (Joshua)**. Only measured numbers appear here.

| Gate | Status | Label | Updated |
|---|---|---|---|
| G1 Pipeline | **PASS** | SYNTHETIC | Thu 5 Nov |
| G2 Offline decoder | **PASS** (see onset caveat) | PUBLIC DATASET + SYNTHETIC | Thu 5 Nov |
| G0 Plumbing | WAITING FOR JOSHUA — `apps/stream_check.py` ready | LIVE | — |
| G4 Stimulus timing | WAITING FOR JOSHUA — `apps/ssvep_stim.py` ready | LIVE | — |
| G5 LLM | WAITING FOR JOSHUA — `scripts/llm_bench.py` | LIVE | — |
| G3 Live SSVEP | WAITING FOR JOSHUA — `apps/ssvep_calibrate.py` | LIVE | — |
| G6 Full loop | BUILT, synthetic/headless-tested; WAITING FOR G3 + JOSHUA — `apps/speller/main.py` | LIVE | — |
| CL Closed-loop template | **PASS** synthetic part (4/4 runs); live eyes-closed test WAITING FOR JOSHUA | SYNTHETIC | Thu 5 Nov |
| G7 ErrP (stretch) | not started (only after G6 passes) | — | — |

## Joshua: run these in order and paste the output
All commands run from `C:\dev
eurohack` in PowerShell. Each one prints a `GATE Gx - ...` line and the evidence path.
LIVE apps require `--electrodes gel|dry|saline|none`. It is stored in every `results.json`, and the speller
warns if you calibrated under a different condition. G0 does not need it (plumbing only).

1. **G0** (headset on, UnicornLSL acquiring in combined "Data" mode). Run it twice: once with Wi-Fi on, once off.
   ```powershell
   uv run python apps/stream_check.py --list
   uv run python apps/stream_check.py --minutes 10 --wifi on
   uv run python apps/stream_check.py --minutes 10 --wifi off
   ```
   Also paste the `channel_roles_guess` block. It settles whether index 14 or 15 is the counter.
2. **G4** (no headset needed, fullscreen; look at the screen; ESC aborts). `configs/stim.yaml` is set to 165 Hz, which `Win32_VideoController` reports for your panel.
   ```powershell
   uv run python apps/ssvep_stim.py
   ```
3. **G5.** In LM Studio, load the model you want; the config suggests `google/gemma-4-e4b`, so edit `configs/llm.yaml` `model:` if you choose another. Then:
   ```powershell
   uv run python scripts/llm_bench.py --wifi off
   ```
4. **G3** (PO7/Oz/PO8 in good contact, plus Pz if you can, dry or gel; 40 trials, about 5 min including breaks).
   Check `apps/quality_viewer.py --source lsl` is green first:
   ```powershell
   uv run python apps/ssvep_calibrate.py --electrodes dry      # or gel / saline
   ```
   Analysis runs automatically. It writes `results/user_calibration.json`, which the speller needs.
5. **G6** (after G3 passes). Type the same 5-word reply in both modes, then do a 10-minute run with one deliberate jaw clench:
   ```powershell
   uv run python apps/speller/main.py --mode baseline --once --electrodes dry
   uv run python apps/speller/main.py --mode llm --once --electrodes dry
   uv run python apps/speller/main.py --mode llm --max-minutes 10 --electrodes dry
   ```
6. **CL live** (eyes open; close them for about 10 s in the middle of each 60 s block; 4 blocks):
   ```powershell
   uv run python apps/closedloop_demo.py --source lsl --adapt --electrodes dry
   ```

---

```
GATE G1 — PASS   [SYNTHETIC]
Evidence: uv run pytest            -> 58 passed (incl. synthetic vs replay identical-code-path test:
                                      same per-window probabilities, quality and decision)
          uv run python scripts/g1_pipeline_bench.py
            -> decision compute per window (3 ch, 4 targets, 250 Hz): FBCCA median/p99/max
               1 s 3.5/5.0/6.7 ms, 2 s 3.9/5.3/8.2 ms, 4 s 6.4/9.6/11.2 ms  (< 50 ms)
               CCA 4 s 0.7/1.1/1.1 ms
            -> 30-min soak on the virtual clock (450k samples, 360 trials, 12 s wall): RSS flat at
               112.95 MB at every 1-min sample (growth +0.0 MB); full update() max 7.0 ms.
               tracemalloc over a 10-min soak after warm-up: net +1 KB.
            -> in-soak calibration (40 cued trials): T=0.060, theta=0.815, ECE 0.617 -> 0.007;
               held-out committed accuracy 0.922 at commit rate 1.00 (synthetic SNR 5 uV)
          uv run python scripts/g1_pipeline_bench.py --realtime     (real 30 min, 2009 s wall incl. calibration)
            -> 360 trials, decoder p99 9.2 ms, full update() max 12.7 ms; RSS 111.70 MB at min 0 ->
               112.04 MB from min ~25 to 30 (growth metric +0.2 MB; no trend at the end)
            -> calibration T=0.049, theta=0.905, ECE 0.609 -> 0.007; held-out committed acc 0.914,
               commit rate 1.00, mean decision 1.07 s (synthetic SNR 5 uV)
          -> results/g1_accelerated.json, results/g1_realtime.json
Decision: continue
Next: G0 on hardware [JOSHUA].
```

```
GATE G2 — PASS   [PUBLIC DATASET: MOABB Kalunga2016, 12 subj, 13/17/21 Hz, chance 1/3] + [SYNTHETIC]
Evidence: uv run python scripts/offline_moabb_fbcca.py      (channels PO7/Oz/PO8, primary window 2.0 s)
  FBCCA > CCA at 2 s: 7/12 (2 ties, 3 losses); averaged over 0.5–4 s windows: 10/12
  FBCCA above chance (binomial, p<0.01) at 2 s: 10/12
  mean acc at 2 s: FBCCA 0.732 vs CCA 0.689
  calibrated dynamic stopping, 2-fold CV per subject (fit T+theta on one half, test on the other):
    held-out ECE 0.392 (T=1) -> 0.019 (fitted T); committed acc 0.902 at commit rate 0.56;
    mean decision 2.56 s; mean ITR 11.2 bits/min. Target accuracy 0.95 NOT reached on held-out data
    (theta fitted on ~24 trials per fold overfits); expect the same live -> use >= 40 calibration trials.
  SYNTHETIC (5 uV SSVEP, configs/synthetic.yaml): FBCCA 0.958 at 2 s, 1.000 at 4 s (>= 95%)
    amplitude sweep: 1.5 uV ~chance, 3.0 uV 0.71 @ 2 s -> figures/g2_synthetic_acc.png
  -> results/g2_moabb.json, figures/g2_acc_vs_window.png, g2_calibration.png, g2_confusion.png
Decision: continue
Next: G0 on hardware.
```

```
GATE CL — PASS (synthetic part)   [SYNTHETIC]
Evidence: uv run python apps/closedloop_demo.py --synthetic-test   (x4 runs: 1 + 3 in results/cl_repeats.log)
  protocol: baseline 30 s + stimulation 30 s; simulated eyes-closed episodes at 10–20 s and 40–50 s
  (alpha gain 0.3 -> 3.0); feature = relative alpha 8–12 / 2–30 Hz on Oz/PO7/PO8, EMA 0.5
  every run: exactly 1 trigger per episode, onset delay 0.30–0.47 s, 0 triggers outside episodes;
  baseline-block trigger logged as sham (no actuation), stimulation-block trigger fired the audio actuator
  software latency (newest sample in the crossing window -> actuator.trigger returned):
    max 8.3–9.4 ms over the 4 runs (< 100 ms); audio mixer buffer adds ~5.8 ms + OS output (not measured)
  every controller tick (~19 Hz) logged with feature, quality, armed state and trigger to events.jsonl
  -> sessions/*_closedloop/results.json (local), configs/closedloop.yaml
  NOTE: thresholds 0.55/0.40 were set from the SYNTHETIC rest distribution (rest 0.23±0.07, p99.5 0.46;
  first run with 0.35/0.25 gave 5 false triggers). Live: use --adapt (baseline mean + 4/2 SD).
Decision: continue (template only until a track needs it)
Next: live eyes-closed test [JOSHUA].
```

```
GATE G6 — not yet evaluated (built; SYNTHETIC/headless functional checks only, no performance claims)
Evidence of function (SYNTHETIC, SDL dummy display, simulated gaze):
  baseline mode: 19 decisions in 36 s, letters typed via 3-level group narrowing;
  llm mode (static fallback suggestions, no LM Studio): suggestion selection + "more"/undo flow;
  EMG artifact on: every trial ended as abstain_quality with reason "all: high EMG (jaw clench / frown?)"
  speller logs per decision: kind, target, p_max, window, quality, reasons, action, text, full window history;
  per sentence: words/min, selections, undo, abstain, timeout, ITR; stimulus frame stats during decoding
Next: LIVE run after G3 [JOSHUA].
```

**G2 caveats and side results (PUBLIC DATASET):**
- **Onset convention, chosen after the first pass.** The first pass started windows at the event marker + 0.14 s. Results: 2 s accuracy FBCCA 0.633 / CCA 0.606; FBCCA > CCA on 5/12 (3 ties); above chance on 7/12 (`results/g2_moabb_marker.json`).
  - A window-offset scan showed the first ~1 s after the marker decodes poorly. Signal persists to 6 s, consistent with subjects shifting gaze to the cued LED.
  - I then adopted MOABB's documented `Kalunga2016.interval[0]` = 2 s as the onset. That value is defined by the dataset, not tuned on our results, but the choice was made after seeing the first pass.
- **Channels matter more than anything else here:**

  | channels | FBCCA 2 s | CCA 2 s | FBCCA > CCA | above chance | stop-CV committed acc / commit rate / ITR |
  |---|---|---|---|---|---|
  | PO7/Oz/PO8 (Unicorn subset) | 0.732 | 0.689 | 7/12 | 10/12 | 0.902 / 0.56 / 11.2 bits/min |
  | POz/PO7/Oz/PO8 | 0.851 | 0.756 | 12/12 | 12/12 | 0.915 / 0.75 / 19.6 bits/min |
  | all 8 occipital | 0.892 | 0.760 | 11/12 (1 tie) | 12/12 | 0.932 / 0.77 / 20.9 bits/min |

  → For the Unicorn, test `[Pz, PO7, Oz, PO8]` against `[PO7, Oz, PO8]` on Joshua's G3 calibration data. Pz is more anterior than POz, so the gain may be smaller. `scripts/analyze_session.py` does this comparison.
