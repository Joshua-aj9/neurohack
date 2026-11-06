# DAY_OF.md — runbook (Sat 7 – Sun 8 Nov, Mathematical Institute)

## 1. Startup checklist (in order)
1. Plug the CSR8510 dongle into the **same USB port** as during pairing. Do not touch drivers, Bluetooth or power settings.
2. Laptop on mains power. Close Teams, OneDrive, browsers and game launchers, which cause frame drops.
3. Fit the headset. Clean the skin behind both ears and stick on the reference/ground pads (fresh ones if possible).
   **Wet PO7, Oz, PO8 with saline** (and Pz if G3 selected it): part the hair, work the pins to the scalp, no drips
   between electrodes. LIVE 6 Nov: saline cut mains on those channels from ~500 to 11–17 µV; dry electrodes fail
   the contact check, so the decoder abstains. Re-wet between long runs. Pass the matching
   `--electrodes saline` (or gel / dry) to every LIVE app.
4. Start **Unicorn Suite → UnicornLSL**. Select the device, **Open**, choose combined ("Data") mode, then **Start**.
5. Start **LM Studio** → Developer → load the G5-chosen model → **Start Server** (port 1234).
6. **Wi-Fi OFF** using the Wi-Fi toggle only. Do NOT use airplane mode: it can also switch off the Bluetooth dongle.
7. Check the stream:
   ```powershell
   cd C:\dev\neurohack
   uv run python apps/stream_check.py --minutes 1 --wifi off
   ```
   Expect `GATE G0 - PASS`, rate 250 ± 1 Hz, loss ≤ 0.1%.
8. Check signal quality by eye. PO7/Oz/PO8 should be green; unwetted channels may stay red ("50 Hz, poor
   contact"), which is expected. Clench your jaw and the reason line should name EMG.
   ```powershell
   uv run python apps/quality_viewer.py --source lsl
   ```
   Then the measured version (1 min; one beep = close eyes, two beeps = open). It needs
   `CONTACT CHECK - EEG VISIBLE` on at least one decoder channel before G3 or the demo:
   ```powershell
   uv run python apps/contact_check.py --electrodes dry   # or gel / saline
   ```
9. Set the display refresh to what `configs/stim.yaml` says (`refresh_hz`). The stimulus apps refuse to run on a mismatch.

## 2. One-command demo launch
```powershell
uv run python apps/speller/main.py --mode llm --electrodes dry   # or gel / saline: must match how you calibrated
```
- **Fallback without the headset:** add `--source synthetic` (labelled SYNTHETIC on screen). Keys 1–4 look at a target, C triggers a jaw clench, which demonstrates abstention.
- **Fallback for idea B:** `uv run python apps/quality_viewer.py --source lsl`.

## 3. First-hour challenge triage
Challenges are revealed on the day. Map each track to existing modules before writing anything new:

| Track (likely shape) | Reuse |
|---|---|
| Peripheral interfaces (EMG / ENG / wearables / dataset given) | `sources/file.py` (load anything) → `preprocessing.py` → `quality.py` (EMG/motion/line checks) → `eval/offline.py`; `closedloop.py` with an EMG band-power feature |
| Non-invasive BCI (EEG) | `pipeline.py` + `decoders/` (CCA/FBCCA, calibration, dynamic stopping), `speller/`, `ssvep_calibrate.py`, `eval/itr.py` |
| Closed-loop neuromodulation (Denison-style: EEG feature → stimulation, scored from logs) | `closedloop.py` `ProtocolRunner` (baseline/stim blocks, sham logging, latency in every log line) + `actuators/`. Swap `bandpower_feature` for the required feature. Logs are already in `events.jsonl` |
| Implants / sub-cortical (LFP, DBS, spikes; offline data) | `sources/file.py` + `replay.py` (stream recorded data through the identical online code path), `closedloop.py` for beta-burst-style adaptive DBS (band = 13–30 Hz, hysteresis/refractory already there) |
| Startup prize | Idea B trust layer: `quality.py` + `quality_viewer.py` ("the decoder knows when not to trust the signal") |

Rule: if a challenge needs < 4 h of new code on top of these, take it; otherwise run idea A.

## 4. Judge-testing protocol (~3–4 min per person, everything logged)
1. **Consent line (read aloud):** "This is a research demo with a consumer EEG headset and flickering lights. It is non-invasive and you can stop at any time. Do you have epilepsy, a history of seizures, or sensitivity to flashing lights?"
   - **If yes, do not run the flicker.** Show the quality viewer instead.
   - "Your EEG and your results are stored anonymously on this laptop under a participant code."
2. Headset on. Gel at PO7/Oz/PO8 (about 60 s). `quality_viewer` must show green.
3. Quick calibration, 3 trials per target, about 75 s. More trials give a more reliable T/θ fit; the minimum workable number is untested:
   ```powershell
   uv run python apps/ssvep_calibrate.py --trials-per-target 3 --participant J01 --electrodes dry
   ```
   This overwrites `results/user_calibration.json`. Joshua's own calibration stays in his session folder; copy `calibration.json` back from there afterwards.
4. One sentence in LLM mode:
   ```powershell
   uv run python apps/speller/main.py --mode llm --once --participant J01 --electrodes dry
   ```
5. Note words/min and ITR from the screen. Everything is logged in `sessions/*_J01`.
6. Afterwards, run `uv run python scripts/make_pitch_figures.py`. Only LIVE sessions go into the live figures.

## 5. Freeze rules
- **Fri 22:00 code freeze.** Day-of changes are config only (`configs/*.yaml`), plus challenge-specific code in a new `apps/<challenge>/`.
- Report every number with its label (SYNTHETIC / PUBLIC DATASET / LIVE).
