# G7 ErrP detector (stretch) — NOT BUILT

Only start after G6 passes. Plan from the handoff:
- **Calibration:** the system is deliberately wrong on about 25% of selections (induced errors).
- **Channels:** Fz/Cz/C3/C4/Pz, bandpass 1–10 Hz.
- **Epochs:** 0–800 ms after feedback onset, with timestamps de-jittered via the counter (`neurohack.timing.dejitter`).
- **Classifier:** shrinkage LDA (`sklearn LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`).
- **PASS:** cross-validated AUC ≥ 0.70. AUC 0.6–0.7 → show as an offline result only.

Reusable already: `recorder.py`, `markers.py`, `timing.py`, `preprocessing.epochs_from_onsets`, `speller/logic.py` (undo).
