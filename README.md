# neurohack — Oxford Neurotech Hackathon (7–8 Nov 2026)

**Idea A:** an SSVEP communication aid for people with ALS or locked-in syndrome.
- The user looks at one of 4 flickering targets. Each target is a next-phrase suggestion from a **local** LLM (LM Studio), or "more" for letter-group spelling.
- Decoding uses **FBCCA**, which needs no training.
- Selections commit only when a **calibrated** probability crosses a threshold (dynamic stopping).
- A **live signal-quality monitor** makes the decoder abstain when the signal is bad.

**Idea B (fallback):** a dry-EEG "trust layer": the quality monitor plus reliability scoring, with eyes-open/closed alpha as the control signal.

Hardware: g.tec Unicorn Hybrid Black (8 EEG @ 250 Hz) → Unicorn Suite → UnicornLSL → LSL. Windows 11, Python 3.11 (uv).

Status of every gate, with evidence: **[GATES.md](GATES.md)**. Day-of runbook: **[DAY_OF.md](DAY_OF.md)**.

## Quick start (PowerShell)
```powershell
uv sync
uv run pytest -m "not slow"
uv run python apps/speller/main.py --mode llm --source synthetic   # full demo, no headset (SYNTHETIC)
```

## Layout
| path | what |
|---|---|
| `src/neurohack/` | **generic, transferable infrastructure** (works for any LSL/file EEG source and any task) |
| `apps/` | thin, idea-specific apps (G0 stream check, G4 stimulus test, G3 calibration, G6 speller, quality viewer, closed-loop demo) |
| `scripts/` | offline analyses and benchmarks (G1, G2, G5, session analysis, pitch figures) |
| `configs/*.yaml` | every tunable; apps contain no magic numbers |
| `results/`, `figures/` | small gate evidence (tracked); `sessions/`, `data/` are local only (git-ignored) |

## Built before the event
All of the following was written on **Thu 5 Nov 2026 (evening)**, before the hackathon, and is generic unless marked *idea-specific*.

**`src/neurohack/` (generic library)**
- `sources/`
  - `base.py`: Source ABC and StreamInfo.
  - `synthetic.py`: a Unicorn-like synthetic EEG with SSVEP, alpha and artifacts.
  - `replay.py`: replays npz/xdf/MNE Raw at real-time pace, with markers.
  - `lsl.py`: LSL inlet.
  - `file.py`: MNE readers for EDF/BDF/GDF/FIF/BrainVision/EEGLAB/.mat/.csv/.npy.
- `buffer.py`: thread-safe ring buffer.
- `preprocessing.py`: stateful causal notch and bandpass, CAR, epoching.
- `quality.py`: per-channel quality score with human-readable reasons. Checks are flatline, railing, clipping, 50 Hz, EMG, motion and counter gaps.
- `decoders/`
  - `cca.py`: CCA.
  - `fbcca.py`: FBCCA.
  - `calibration.py`: temperature scaling and ECE.
  - `stopping.py`: confidence-gated dynamic stopping with quality abstention, and θ fitting.
- `pipeline.py`: the online Source → filter → quality + decoder → stopping chain.
- `worker.py`: runs the decoder in a background thread.
- `markers.py`: LSL marker outlet and inlet.
- `recorder.py`: EEG and markers to npz/jsonl on LSL time.
- `timing.py`: counter-based de-jitter, gap report, gap filling.
- `session.py`: session folders with a config snapshot, events.jsonl and results.json.
- `stimulus.py`: pygame vsync display with a refresh check, sampled-sinusoid luminance, and frame statistics.
- `actuators/`: audio, visual and log-only actuators.
- `closedloop.py`: band-power feature → threshold/hysteresis/refractory controller → actuator. Includes a baseline/stimulation protocol runner and adaptive thresholds.
- `llm.py`: LM Studio client with JSON parsing, timeout and a static fallback.
- `eval/`
  - `itr.py`: Wolpaw ITR.
  - `offline.py`: accuracy vs window, binomial test, confusion matrix.
  - `figures.py`: standard figures.

**`apps/`**
- Generic tools:
  - `stream_check.py` (G0)
  - `ssvep_stim.py` (G4)
  - `quality_viewer.py`
  - `closedloop_demo.py`
- *Idea-specific*:
  - `ssvep_calibrate.py` (G3 protocol)
  - `speller/` (G6: `logic.py` state machine and `main.py` UI)
- `errp/` is a placeholder only (G7, not built).

**`scripts/`**
- `g1_pipeline_bench.py`
- `offline_moabb_fbcca.py`
- `llm_bench.py`
- `analyze_session.py`
- `make_pitch_figures.py`

**`tests/`**: pytest suite covering every module above.

**Added Fri 6 Nov (morning, still before the event):** `--electrodes gel|dry|saline|none`. It is required for LIVE
calibration, speller and closed-loop runs and is recorded in every session's `results.json` (`session.py`). The
speller warns when the calibration condition differs from the current run.

**Added Fri 6 Nov (afternoon, still before the event):** contact check. `src/neurohack/eval/contact.py` (generic):
mains amplitude, raw vs filtered peak-to-peak, inter-channel correlation (shared reference noise) and eyes
closed/open alpha reactivity. `apps/contact_check.py` runs the 1-minute protocol. `configs/quality.yaml`
gains a `contact_check` section. Tests are in `tests/test_contact.py`. `preprocessing.design_stream_sos`
gains an optional Butterworth band-stop. The decoder now uses 47–53 Hz instead of the 50 Hz notch, chosen
from Joshua's LIVE dry recording (see GATES.md). `quality.assess` takes an optional `filtered` window: flatline,
railing and EMG are scored on the decoder-filtered signal, and mains becomes an absolute-µV contact check on the
raw signal (thresholds from the dry and saline LIVE recordings).

Pre-event results (all SYNTHETIC or PUBLIC DATASET; none are LIVE) are in `GATES.md`, `results/` and `figures/`.

## Conventions
See [CLAUDE.md](CLAUDE.md). Every reported number carries a label: **SYNTHETIC**, **PUBLIC DATASET** or **LIVE**.
