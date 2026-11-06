# Claude Code handoff: Oxford Neurotech Hackathon prep repo

Read this whole file before doing anything. Then write a plan of 20 lines or fewer, and start executing straight away. Only pause where a gate below says **[JOSHUA]**: those steps need the headset, the screen, or LM Studio on Joshua's machine.

---

## 0. Context

- **Event:** Oxford Neurotech Hackathon, Sat 7 to Sun 8 Nov 2026, Mathematical Institute, Oxford.
  - Four tracks span the neurotech stack, from peripheral interfaces to sub-cortical implants.
  - Track challenges are **unknown until the day**.
  - Judges are Oxford PIs plus one industry judge per track; there is also a startup prize.
  - A Denison-style closed-loop challenge (EEG feature → stimulation, scored from logs) is plausible.
- **Primary idea (A):** an SSVEP communication aid for people with ALS or locked-in syndrome.
  - The user looks at flickering targets. Each target is a next-word or phrase suggestion from a **local** LLM (LM Studio).
  - FBCCA decoding (no training needed) with **calibrated, confidence-gated dynamic stopping**.
  - A **live signal-quality monitor** makes the decoder abstain when the signal is bad.
  - Stretch goal: an error-related potential (ErrP) detector that auto-undoes wrong selections.
  - Headline demo metric: words/min and ITR with LLM suggestions vs letter-by-letter spelling, measured live on several people.
- **Fallback idea (B):** a dry-EEG "trust layer" product (the quality monitor plus reliability scoring), with eyes-open/closed alpha as the control signal.
- **Hardware and software (Windows 11):**
  - i7-14650HX, 64 GB RAM, RTX 5050 8 GB.
  - g.tec **Unicorn Hybrid Black**: 8 EEG channels at 250 Hz, plus accelerometer, gyroscope, battery, sample counter and validation channels.
  - The headset connects through the CSR8510 A10 USB dongle and is paired. The internal Intel Bluetooth radio is deliberately DISABLED. **Do not touch drivers, Bluetooth settings or power settings.**
  - Data path: Unicorn Suite → **UnicornLSL** app → LSL.
  - Default electrode order (confirm against the Unicorn manual on GitHub, `unicorn-bi`): Fz, C3, Cz, C4, Pz, PO7, Oz, PO8.
  - LM Studio local server, OpenAI-compatible, at `http://localhost:1234/v1`. Joshua loads the model.
- **Rules on pre-written code are unknown.** Keep a hard split:
  - `src/neurohack/` is generic, transferable infrastructure.
  - `apps/` holds thin, idea-specific code.
  - The README must have a "Built before the event" section listing exactly what was prebuilt.

## 1. Non-negotiables

1. **Never report a number you did not measure.** Label every result **SYNTHETIC**, **PUBLIC DATASET** or **LIVE (Joshua)**. Never present synthetic results as live ones.
2. Gates report **PASS / PARTIAL / FAIL** with evidence (command run, output, figure path). Keep `GATES.md` as a live status board.
3. **Scope-cut over effort-increase.** If time runs short, cut in this order: G7 (ErrP) → UI polish → closed-loop extras. **Never cut G0–G3.**
4. Hardware-dependent steps: write the script and exact run instructions, then **stop and wait** for Joshua to paste the output. Never mock hardware results.
5. **Do not invent protocol details.** For the Unicorn serial fallback, work only from the official `UnicornBluetoothProtocol.pdf` in `unicorn-bi/Unicorn-Suite-Hybrid-Black` on GitHub.
6. **Code style:**
   - Type hints, small modules, pure functions where possible.
   - `pytest` for tests, `ruff` for linting.
   - Configs in YAML, no magic numbers in app code.
   - Follow the Good Research Code Handbook style.
   - Create `CLAUDE.md` with these conventions so later sessions inherit them.
7. Windows-native: PowerShell commands, no WSL (Bluetooth and LSL run on the host). Large data and recordings stay out of git.

## 2. Environment

- Python 3.11 venv via `uv` (install uv if missing).
- Dependencies: `numpy scipy scikit-learn mne pylsl pyxdf pygame-ce moabb openai pyyaml pytest ruff`. Add `pyserial` only if the serial fallback is built.
- Verify `import pylsl` works (the wheel bundles liblsl) and print versions into `ENV.md`.

## 3. Repo layout

```
neurohack/
  README.md  CLAUDE.md  GATES.md  ENV.md  pyproject.toml  .gitignore
  configs/   device.yaml  stim.yaml  decoder.yaml  quality.yaml  llm.yaml  closedloop.yaml
  src/neurohack/
    sources/      base.py (Source ABC: start/stop/read -> data[n_ch,n], timestamps[n])
                  lsl.py  synthetic.py  replay.py (npz/xdf/MOABB at real-time pace)
                  file.py (MNE readers: EDF/BDF/FIF/.mat/.csv, for datasets handed out on the day)
                  serial_unicorn.py (CONDITIONAL, see G0)
    buffer.py         thread-safe ring buffer
    preprocessing.py  stateful causal SOS filters (50 Hz UK notch, bandpass), CAR, window extraction
    quality.py        per-channel quality score in [0,1] plus reasons
    decoders/         base.py (predict_proba(window) -> probs), cca.py, fbcca.py,
                      calibration.py (temperature scaling), stopping.py (dynamic stopping)
    markers.py        LSL marker outlet and helpers
    recorder.py       records EEG and marker inlets to a session folder (npz + jsonl) on LSL time
    timing.py         de-jitter timestamps using the sample counter and nominal 250 Hz
    actuators/        base.py (Actuator.trigger(event)), audio.py (pygame.mixer), visual.py, log_only.py
    closedloop.py     feature -> controller (threshold/hysteresis/refractory) -> actuator, with a
                      baseline/stimulation protocol runner and full logs
    llm.py            LM Studio client: JSON suggestions, timeout, fallback (static n-gram/phrase list)
    session.py        session folder: config snapshot, events.jsonl, results.json
    eval/             itr.py (Wolpaw), offline.py, figures.py
  apps/
    stream_check.py       G0
    quality_viewer.py     live per-channel quality and spectra (the trust-layer demo for idea B)
    ssvep_stim.py         G4 flicker and frame-timing test
    ssvep_calibrate.py    G3 cued calibration protocol
    speller/              G6 main demo
    closedloop_demo.py    alpha-power -> audio template for a closed-loop track
    errp/                 G7 stretch
  scripts/  offline_moabb_fbcca.py  analyze_session.py  make_pitch_figures.py
  tests/
```

## 4. Key technical specs

**FBCCA (Chen et al., 2015)**
- Sub-bands m = 1..5, bandpass [8m, 88] Hz (Chebyshev type I, zero-phase on the decision window; pad to limit edge effects).
- Reference signals: sin and cos at k·f for k = 1..Nh. Keep k·f below the sub-band upper edge and below Nyquist (125 Hz).
- Weights w(m) = m^(-1.25) + 0.25. Score per target = Σ w(m)·ρ_m², where ρ_m is the largest canonical correlation in sub-band m.
- Channels default to PO7, Oz, PO8 (configurable).
- Implement plain CCA too, as a baseline.

**Calibrated probabilities and dynamic stopping**
- Convert scores to probabilities with softmax(scores / T), with T fitted on calibration data (minimise NLL). Report ECE (expected calibration error) before and after.
- Dynamic stopping: the window grows from 1.0 s in 0.25 s steps up to a maximum (default 4 s).
  - **Commit** a selection when p_max ≥ θ.
  - **Abstain** if quality on the decision channels is below the threshold.
  - **Time out** at the maximum window.
  - Choose θ on calibration data to reach the target accuracy (default 95%).

**Quality monitor**, per channel over a rolling 1 s window, with all thresholds in `quality.yaml`:
- flatline (std below threshold)
- railing / saturation
- 50 Hz line-noise ratio
- EMG band power ratio (20–45 Hz vs 1–20 Hz)
- motion from accelerometer and gyroscope
- counter gaps (dropped samples)

The output is a score and human-readable reasons, for example "Oz: high EMG (jaw clench?)".

**Wolpaw ITR** (bits/selection):
B = log2 N + P·log2 P + (1−P)·log2((1−P)/(N−1)). Multiply by selections per minute, including gaps between selections.

**Stimulus**
- pygame-ce, fullscreen, vsync. Read the actual refresh rate and refuse to run if it differs from `stim.yaml`.
- Default mode is sampled sinusoidal luminance, L(t) = 0.5·(1 + sin(2πft + φ)), evaluated on each frame's flip time. This allows any frequency.
- Default frequency set at 60 Hz: 8.57, 10, 12, 15. Make it configurable, with a second set that avoids the 10 Hz alpha band, because alpha can bias the decoder toward 10 Hz.
- Push an LSL marker on the first flip of each trial and log every flip time.

**Speller UX (G6)**
- Four targets: three LLM suggestions plus "more ▸". "More" opens letter-group mode, where the alphabet is split into four groups and selecting a group splits it further. There are also undo and send.
- A context prompt (for example a caregiver asking "How are you feeling today?") plus the partial sentence go to the LLM.
- A baseline mode does letter-group spelling only. Both modes log selections, errors, times, words/min and ITR.
- The UI always shows the quality status, and shows abstentions clearly.

## 5. Gates (in execution order)

| Gate | Who | PASS | PARTIAL | FAIL → action |
|---|---|---|---|---|
| **G1 Pipeline** | CC | Synthetic and replay sources run through the identical code path; pytest green; decision compute < 50 ms per window; 30-min synthetic run with no memory growth | n/a | Fix before anything else |
| **G2 Offline decoder** | CC | FBCCA on MOABB Kalunga2016 (formerly SSVEPExo; check the current class name; stimulus classes only, chance = 1/3): beats CCA on most subjects; above chance by binomial test (p < 0.01) on most subjects; accuracy-vs-window curve saved; ≥ 95% on synthetic | Above chance but FBCCA ≤ CCA | Bug, not idea: debug filters, references, channels |
| **G0 Plumbing [JOSHUA]** | CC writes `stream_check.py` | Stream found; measured rate 250 ± 1 Hz over 10 min; counter-gap loss ≤ 0.1%; channel map and counter index confirmed; recorder saves EEG and markers in sync; repeated with Wi-Fi OFF | Gaps 0.1–1% (fill by interpolation, log it) | No stream by Fri 12:00 → build `serial_unicorn.py` from the official PDF; if that fails → idea B on replayed data |
| **G4 Stimulus timing [JOSHUA]** | CC writes `ssvep_stim.py` | 60 s run: dropped frames < 0.5%; frame-interval std < 1 ms; refresh matches config | Drops 0.5–2% | Switch to PsychoPy for the stimulus |
| **G5 LLM [JOSHUA]** | CC writes the client and benchmark | 4 suggestions as valid JSON, p90 latency < 1 s over 20 calls, Wi-Fi off | p90 1–2 s | Smaller model, or the static fallback list |
| **G3 Live SSVEP [JOSHUA]**, the real go/no-go | CC writes `ssvep_calibrate.py` + analysis | 4 targets × 10 trials × 4 s on Joshua (gel at PO7/Oz/PO8): ≥ 90% at ≤ 3 s windows; clear PSD peaks at the stimulus frequencies at Oz | 70–90%: use 3 targets, a longer maximum window, or the non-alpha frequency set | < 70% after one round of electrode fixes → pivot to idea B |
| **G6 Full loop [JOSHUA]** | CC builds `apps/speller` | Joshua types a 5-word reply in LLM mode faster (words/min) than in baseline mode, both logged; 10-min run with no crash; abstention triggers on a deliberate jaw clench | Works but no speed gain | Demo baseline mode plus the quality monitor |
| **CL Closed-loop template** | CC (+ Joshua for the live test) | Synthetic: triggers correctly with hysteresis and refractory period; software latency from threshold crossing to audio < 100 ms; live eyes-closed test triggers on alpha rise; baseline/stimulation protocol logs | n/a | Keep it as a template only |
| **G7 ErrP (stretch) [JOSHUA]** | CC, only after G6 passes | Calibration with induced errors (system deliberately wrong on ~25% of selections), Fz/Cz/C3/C4/Pz, 1–10 Hz, 0–800 ms epochs, timestamps de-jittered with the counter, shrinkage LDA: cross-validated AUC ≥ 0.70 | AUC 0.6–0.7: show as an offline result only | Drop it |

## 6. Schedule (now: Thu 5 Nov evening)

- **Thu night:** environment, G1, G2, and `stream_check.py` ready for Joshua's G0 run.
- **Fri morning:** G4, G5, quality monitor and viewer.
- **Fri 12:00:** G0 decision point.
- **Fri afternoon:** G3 (go/no-go), then G6.
- **Fri evening:** CL template, `make_pitch_figures.py` (accuracy vs window, PSD at Oz, confusion matrix, calibration curve, words/min comparison); G7 only if everything else has passed.
- **Fri 22:00: code freeze.** Write `DAY_OF.md`:
  - a startup checklist (dongle in the same USB port, Unicorn Suite, UnicornLSL, LM Studio server, Wi-Fi off, run `stream_check`);
  - first-hour challenge triage (map each track to the reusable modules);
  - a one-command demo launch;
  - a judge-testing protocol (consent line, ~3 min per person, logged).

## 7. Reporting format (every gate)

```
GATE Gx — PASS | PARTIAL | FAIL   [SYNTHETIC | PUBLIC DATASET | LIVE]
Evidence: <command> → <key numbers> → <figure/log path>
Decision: <continue | remediation | pivot>
Next: <one line>
```

Start now. First output: the plan, then environment setup.
