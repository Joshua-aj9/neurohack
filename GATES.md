# GATES.md — live status board

Labels: **SYNTHETIC** | **PUBLIC DATASET** | **LIVE (Joshua)**. Only measured numbers appear here.

| Gate | Status | Label | Updated |
|---|---|---|---|
| G1 Pipeline | **PASS** | SYNTHETIC | Thu 5 Nov |
| G2 Offline decoder | **PASS** (see onset caveat) | PUBLIC DATASET + SYNTHETIC | Thu 5 Nov |
| G0 Plumbing | **PASS** | LIVE | Fri 6 Nov 00:27 |
| G4 Stimulus timing | **PASS** | LIVE | Fri 6 Nov 00:30 |
| G5 LLM | **PASS with caveat**: latency PASS with Wi-Fi off; the only strict 20/20+ JSON run had Wi-Fi on | LIVE | Fri 6 Nov 00:54 |
| Contact check (pre-G3) | dry and saline: **EEG visible** on PO7/Oz/PO8. Saline cut occipital mains ~30–50× (≈500 → 11–17 µV), alpha ratio 4.6–6.4. Decoder now uses a 47–53 Hz band-stop; quality monitor now scores the filtered signal + mains as contact (saline: 89% of windows pass) | LIVE | Fri 6 Nov 17:40 |
| G3 Live SSVEP | **PASS**: PO7/Oz/PO8 FBCCA 0.97 @ 2 s, 1.00 @ 2.5 s (40 trials); held-out dyn. stopping acc 0.95, ITR 38.7 bits/min | LIVE (saline) | Fri 6 Nov 18:03 |
| G6 Full loop | BUILT, synthetic/headless-tested; G3 done -> WAITING FOR JOSHUA — `apps/speller/main.py` | LIVE | — |
| CL Closed-loop template | **PASS** synthetic part (4/4 runs); live eyes-closed test WAITING FOR JOSHUA | SYNTHETIC | Thu 5 Nov |
| G7 ErrP (stretch) | not started (only after G6 passes) | — | — |

## Joshua: run these in order and paste the output
All commands run from `C:\dev\neurohack` in PowerShell. Each one prints a `GATE Gx - ...` line and the evidence path.
LIVE apps require `--electrodes gel|dry|saline|none`. It is stored in every `results.json`, and the speller
warns if you calibrated under a different condition. G0 does not need it (plumbing only).

1. **G0** (UnicornLSL acquiring in combined "Data" mode; the headset need not be worn). Run the commands
   ONE AT A TIME, never in two terminals at once. The Wi-Fi-off run is the one that counts.
   ```powershell
   uv run python apps/stream_check.py --minutes 10 --wifi off
   ```
   The counter index is already settled (15, confirmed 6 Nov).
2. **G4** (no headset needed, fullscreen; look at the screen; ESC aborts). `configs/stim.yaml` is set to 165 Hz, which `Win32_VideoController` reports for your panel.
   ```powershell
   uv run python apps/ssvep_stim.py
   ```
3. **G5.** In LM Studio, load the model you want; the config suggests `google/gemma-4-e4b`, so edit `configs/llm.yaml` `model:` if you choose another. Then:
   ```powershell
   uv run python scripts/llm_bench.py --wifi off
   ```
4. **G3** (DONE, PASS with saline). PO7/Oz/PO8 wetted with saline, plus Pz only if wetted too; 40 trials, ~5 min.
   First run `apps/contact_check.py --electrodes <same>`; it must say `EEG VISIBLE`. (The quality viewer
   shows unwetted channels red for 50 Hz / poor contact; that is expected.)
   ```powershell
   uv run python apps/ssvep_calibrate.py --electrodes saline   # or gel / dry
   ```
   Analysis runs automatically. It writes `results/user_calibration.json`, which the speller needs.
5. **G6** (after G3 passes). Type the same 5-word reply in both modes, then do a 10-minute run with one deliberate jaw clench:
   ```powershell
   uv run python apps/speller/main.py --mode baseline --once --electrodes saline
   uv run python apps/speller/main.py --mode llm --once --electrodes saline
   uv run python apps/speller/main.py --mode llm --max-minutes 10 --electrodes saline
   ```
6. **CL live** (eyes open; close them for about 10 s in the middle of each 60 s block; 4 blocks):
   ```powershell
   uv run python apps/closedloop_demo.py --source lsl --adapt --electrodes saline
   ```

---

```
GATE G3 — PASS   [LIVE (Joshua), saline on PO7/Oz/PO8, Fri 6 Nov 17:58]
Evidence: uv run python apps/ssvep_calibrate.py --electrodes saline
  -> sessions/20261106_175818_ssvep_calibrate_default_joshua_saline/analysis.json (local only);
     figures/g3_*.png; results/user_calibration.json
  40 trials, targets 8.57/10/12/15 Hz, chance 0.25; refresh measured 165.19 Hz (825-frame probe)
  PO7/Oz/PO8 (selected): FBCCA 0.97 @ 2 s, 1.00 @ 2.5 s (binomial p=8.3e-25)
    held-out dynamic stopping (2-fold): committed acc 0.95, commit rate 0.95, mean 1.39 s, ITR 38.7 bits/min
    quality gate passes 87% of 1 s windows
  Pz+PO7/Oz/PO8: FBCCA 1.00 @ 1.5 s; held-out acc 0.90, commit 1.00, 1.06 s, ITR 40.0
    but Pz was not wetted (mains median 3062 uV): quality gate passes 0% -> NOT ELIGIBLE
  Oz SNR at stimulus frequency: 8.57 Hz 3.06, 10 Hz 9.65, 12 Hz 24.1, 15 Hz 22.0 (4/4 clear peaks)
  final fit on all trials: T=0.0452, theta=0.885, ECE 0.649 -> 0.023 (in-sample)
Fix found on the way: analyze_session first selected Pz+PO7/Oz/PO8 by ITR (40.0 vs 38.7). With the
  live quality gate the speller would have abstained on every selection. Selection now requires the
  gate to pass in >= 50% of the recording's 1 s windows (decoder.yaml analysis.min_quality_pass_frac).
  The same session was re-analysed: user_calibration.json now uses PO7/Oz/PO8.
Refresh check fix: 120-frame startup probes on the G4 flip times ranged 163.3-166.8 Hz (18% outside
  +-1 Hz -> false MISMATCH twice); 825-frame probes 164.5-165.5 Hz (0% outside). Tolerance unchanged.
Next: G6 speller runs (saline, re-wet first).
```

```
CONTACT CHECK — EEG VISIBLE on PO7, Oz, PO8   [LIVE (Joshua), dry electrodes, Fri 6 Nov 17:16]
Evidence: uv run python apps/contact_check.py --electrodes dry
  -> sessions/20261106_171602_contact_check_dry/ (results.json, contact_check.npz; local only)
  protocol: 5 s settle, open/closed/open/closed x 15 s, first 2 s after each cue dropped, 2 s segments
  mains: 50 Hz band RMS 474-2279 uV raw (occipital 473-534 uV); peak at 49.75 Hz, not 50.0
  outside 45-55 Hz the raw signal is EEG-sized: 3-7 uV RMS per band 4-45 Hz on every channel
    (Pz: 183 uV at 0.5-4 Hz = drift; Pz is not a decoder channel)
  old notch (50.0 Hz, Q30) left 103-116 uV of mains on PO7/Oz/PO8; filtered ptp 340-1456 uV
  mean |r| between channels 0.97 after filtering (the residual mains is shared), 0.75 after CAR
  alpha closed/open (8-12 Hz power), no CAR / CAR: PO7 1.52 (p=0.005) / 2.67, Oz 1.19 (p=0.20) / 1.75
    (p=0.002), PO8 2.71 / 5.33 (p<0.001); criterion ratio >= 1.5 and p <= 0.05 (HEURISTIC, not validated)
  DC offsets 298-684 mV (Pz highest)
Mains filter, chosen on this recording (scratch analysis, causal filters as online):
  occipital 45-55 Hz RMS: notch Q30 103-116 uV; notch Q10 35-40; notch Q5 18-20;
  band-stop 47-53 order 2: 1.9-2.0 uV (Oz 8-45 Hz RMS 8.0 uV in every variant); order 4: 1.4-1.6 uV
  -> configs/decoder.yaml now uses band-stop 47-53 Hz order 2 (gain at 48 Hz -8 dB, all other SSVEP
     harmonics of the default set > -1 dB); notch disabled
  G2 rerun with the band-stop [PUBLIC DATASET, Kalunga2016 13/17/21 Hz, so the 48 Hz loss is not tested]:
     mean acc 2 s FBCCA 0.753 (was 0.732), CCA 0.707 (was 0.689); above chance 12/12 (was 10/12);
     FBCCA > CCA 7/12 (3 ties); held-out committed acc 0.886 (was 0.902) at commit rate 0.61 (was 0.56);
     ECE 0.398 -> 0.016; ITR 11.8 bits/min (was 11.2) -> results/g2_moabbbandstop.json, g2_bandstop.log
Quality monitor on the same recording (1 s windows, decoder channels >= quality_min 0.5):
  scoring the raw signal (current): 0% of 62 windows pass (railing + 50 Hz on every channel)
  scoring the decoder-filtered signal: 97% pass; remaining flags are EMG, mostly on Pz
  -> changed after the saline run (see below)
```

```
CONTACT CHECK — EEG VISIBLE on PO7, Oz, PO8   [LIVE (Joshua), saline on PO7/Oz/PO8, ear pads re-seated, 17:39]
Evidence: uv run python apps/contact_check.py --electrodes saline
  -> sessions/20261106_173943_contact_check_saline/ (local only)
  decoder channels (wetted), raw 48-52 Hz RMS per 1 s window (min / median / max):
    PO7  8.7 / 10.9 / 12.9 uV   Oz  9.0 / 10.6 / 13.8   PO8  15.3 / 17.2 / 20.7
    same channels dry: PO7 498 / 522 / 659   Oz 449 / 476 / 613   PO8 521 / 534 / 663
    -> no overlap: the worst saline window is >20x below the best dry window
  raw ptp 79-102 uV (dry 1359-1521); filtered RMS 11.0-11.6 uV (dry 105-117)
  after the band-stop, 45-55 Hz RMS 1.2-1.7 uV; Oz filtered spectrum peaks at 11.75-12.25 Hz (alpha)
  alpha closed/open, no CAR: PO7 5.43, Oz 4.61, PO8 6.41 (all p<0.001; dry 1.52 / 1.19 / 2.71)
  unwetted channels (Fz, C3, Cz, C4, Pz) got WORSE: 50 Hz 2.5-6.8 mV (dry 0.74-2.3 mV); 90-125 Hz
    60-148 uV after filtering. Cause not determined from this data.
  CAR now HURTS the decoder channels (RMS 11 -> 54 uV): it mixes in the bad channels. Decoder keeps car: false.
  mains peak moved to 50.0 Hz (dry run: 49.75 Hz): the band-stop covers both
  mean |r| between channels 0.43 filtered (dry 0.97)
Quality monitor on this recording (decoder channels >= quality_min 0.5, 62 x 1 s windows):
  raw (current): 10% pass (PO8 50 Hz ratio; PO7 near threshold)   decoder-filtered: 89% pass
Decision for G3: wet PO7/Oz/PO8 with saline (Pz too only if it is actually wetted; it was not usable here).
```

```
QUALITY MONITOR CHANGE (Fri 6 Nov ~17:50, agreed with Joshua)   [LIVE recordings re-scored offline]
  flatline / railing / EMG now use the decoder-filtered signal (band-stop + bandpass); clipping and
  mains use the raw signal. Mains became a CONTACT check: absolute 48-52 Hz RMS per 1 s window,
  warn 50 uV / bad 200 uV (HEURISTIC, set between the two recordings: saline 8.7-20.7, dry 449-663 uV).
  Decoder channels PO7/Oz/PO8 >= quality_min 0.5, 62 x 1 s windows:
    saline: 10% (old) -> 89% (new); 85% >= good_min 0.7. Unwetted Fz/C3/Cz/C4/Pz flagged "50 Hz (poor contact?)".
    dry:     0% (old) ->  0% (new); the only reason is now mains/poor contact (no false "huge amplitude").
    -> with these thresholds a DRY setup always abstains. Dry still showed alpha reactivity, so if dry is
       ever needed, G3-dry accuracy should decide whether to relax warn_uv/bad_uv, not this heuristic.
  Tests: 78 passed (2 new quality tests: 500 uV mains -> contact flag, no railing; 15 uV mains -> no penalty).
  CL synthetic gate rerun: PASS. G1 not rerun (quality now also reads the filtered buffer; no timing check yet).
```

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

```
GATE G5 — PASS with caveat   [LIVE, google/gemma-4-e4b, plain JSON mode (structured_output: false)]
Evidence:
  run A, Wi-Fi OFF, 20 calls x 4 suggestions (00:49): p50 0.620 s, p90 0.673 s, max 0.710 s;
    19/20 strict JSON. Call 3 dropped the closing ']' ('..."Just a little"}') -> fell back to the
    static list (the repair below did not exist yet)  -> sessions/20261106_004939_llm_bench
  run B, Wi-Fi ON (Joshua had re-enabled it), 100 calls: 100/100 strict JSON, p50 0.615 s,
    p90 0.705 s, max 0.802 s; script verdict PASS  -> sessions/20261106_005312_llm_bench
  malformed JSON in plain mode today: 1 of 160 calls (run A 20 + diagnostics 40 + run B 100)
  Added: repair_suggestions() recovers strings from an unclosed/truncated list (tested on the exact
    live reply); the bench now reports strict JSON (gate criterion) and usable-after-repair separately.
Caveat: by the letter (20 calls, all valid JSON, Wi-Fi off) no single run meets every condition:
  A had Wi-Fi off but 19/20 strict; B had 100/100 but Wi-Fi on. Inference is local (localhost), so
  Wi-Fi is not expected to change it, but that was not measured. For a by-the-letter PASS:
  `uv run python scripts/llm_bench.py --wifi off`.
Decision: continue (does not block G3/G6)
Next: G3.
```

```
GATE G5 — first attempt, superseded: PARTIAL   [LIVE, google/gemma-4-e4b in LM Studio, Wi-Fi off]
Evidence: uv run python scripts/llm_bench.py --wifi off   (00:40, structured_output: true, 4 suggestions)
  20/20 valid JSON, p50 1.079 s, p90 1.164 s, max 1.195 s  (PASS needs p90 < 1 s)
  -> sessions/20261106_004027_llm_bench/results.json
Diagnosis (LIVE diagnostic calls, same model, not the gate script; Wi-Fi state not re-checked):
  decode speed ~40 tok/s in every condition; json_schema mode makes the model emit ~45 completion
  tokens for 4 suggestions vs ~25 in plain mode (prompt 157 tokens either way)
  20 calls each, plain mode: 4 suggestions 20/20 valid, p50 644 / p90 696 / max 745 ms;
                             3 suggestions (what the speller uses) 20/20 valid, p50 512 / p90 573 ms
Remediation: configs/llm.yaml structured_output: false (parser + static fallback still catch bad
  output). Speller now starts the LLM call at commit time, overlapping the 0.6 s feedback highlight.
  Bug found in the G5 log and fixed: "Yes, can I" + "can I have tea?" typed "can I" twice ->
  clean() now strips any word overlap between the end of the reply and the start of a suggestion.
Decision: remediation (same model, cheaper output format)
Next: rerun the official benchmark to confirm: uv run python scripts/llm_bench.py --wifi off
```

```
GATE G4 — PASS   [LIVE, built-in panel, fullscreen, stimulus only]
Evidence: uv run python apps/ssvep_stim.py   (60 s, 4 targets 8.57/10/12/15 Hz, sampled sinusoid)
  measured refresh 165.16 Hz (config 165 +/- 1; Windows API 165)
  dropped frames 8 / ~9,900 = 0.08% (PASS < 0.5%); 1 interval > 2 frames
  frame-interval SD 0.41 ms (PASS < 1 ms); median 6.055 ms, p99 7.03 ms, max 12.29 ms
  -> sessions/20261106_002941_ssvep_stim/results.json (+ flip_times.npy)
  Scope: stimulus alone. With the decoder thread running, the speller logs its own frame stats
  per session (results.json "stim_frame_stats"); check them at G6.
Decision: continue
Next: G5 (LM Studio benchmark).
```

```
GATE G0 — PASS   [LIVE, UN-2024.06.14, Wi-Fi OFF, single run, headset not worn]
Evidence: uv run python apps/stream_check.py --minutes 10 --wifi off   (00:16:54, nothing else running)
  rate from device counter 250.306 Hz over 601.7 s (within 250 +/- 1)
  counter-gap loss 0.000% (150,608 of 150,608 samples, 0 gap events, 0 backwards)
  counter index 15 confirmed (matches config)
  markers: 60 sent, 60 received, 60 inside the EEG span, max |lag| 0.011 ms
  LSL timestamp jitter vs counter fit 7.8 ms SD; sample interval p99 24.1 ms, max 61.2 ms
  quality block all red = expected with the headset off the head (not part of G0)
  -> sessions/20261106_001654_stream_check/results.json
  The "Stream transmission broke off" line printed after the verdict was a shutdown-order message
  (marker outlet closed before the recorder's inlet); fixed: inlet.close() before exit.
Decision: continue
Next: G4 (stimulus timing).
```

```
GATE G0 — earlier attempt, superseded (rerun needed)   [LIVE, UN-2024.06.14, headset not worn]
Evidence: two 10-min runs of apps/stream_check.py, started 00:00:11 (--wifi on) and 00:01:19 (--wifi off).
  They OVERLAPPED for ~9 min, so both "wifi" labels describe the same Wi-Fi state for most of the run.
  Measured (both runs):
    rate from device counter 250.383 / 250.379 Hz over 588 / 601 s (within 250 +/- 1)
    counter-gap loss 0.000% (147,194 and 150,481 samples, 0 gap events); 0 backwards steps
    counter = index 15 (detected from data, matches config); battery 14; validation 16 = constant 1
    LSL timestamp jitter vs counter fit: 14.2 / 12.7 ms SD; sample interval p99 24 ms, max 66.9 ms
      (Bluetooth delivers in bursts -> offline analysis de-jitters with the counter)
    battery channel 66.7 -> 60.0 over the runs (moves in ~6.7 steps)
  Why both printed FAIL (test-harness bugs, not device faults; fixed in apps/stream_check.py):
    run 1: UnicornLSL started ~12.6 s after the script (first counter = 1), so 2 of 60 sync markers
      preceded the first EEG sample
    run 2: both runs published a marker stream with the same name; run 2's inlet resolved run 1's stream
      (68 s "lag" = the start-time difference; "transmission broke off" at 00:10:13 = run 1 ending)
  Fix: per-run marker stream (unique source_id) + run id in every marker; clock starts at the first
    EEG sample; marker |lag| limit 5 ms (device.yaml). Verified: two synthetic runs at once -> each
    3/3 own markers, lag <= 0.03 ms. Regression test tests/test_markers.py.
Decision: continue (the plumbing numbers meet PASS; only the marker check and the Wi-Fi-off condition are unproven)
Next: ONE 10-min run, Wi-Fi off, nothing else running:
  uv run python apps/stream_check.py --minutes 10 --wifi off
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
