# CLAUDE.md — conventions for this repo (Oxford Neurotech Hackathon, 7–8 Nov 2026)

## Hard rules
- **Never report a number you did not measure.** Every result is labelled **SYNTHETIC**, **PUBLIC DATASET** or **LIVE (Joshua)**. Never present synthetic results as live ones.
- Gates (see `GATES.md`) report **PASS / PARTIAL / FAIL** with evidence: command → key numbers → figure/log path.
- Hardware steps (headset, screen, LM Studio): write the script and exact PowerShell run instructions, then **stop and wait** for Joshua to paste the output. Never mock hardware results.
- **Do not touch drivers, Bluetooth, or power settings.** The CSR8510 dongle is used; the Intel BT radio is disabled on purpose.
- Unicorn serial protocol: only from the official `UnicornBluetoothProtocol.pdf` (unicorn-bi/Unicorn-Suite-Hybrid-Black). Do not invent protocol details.
- Scope cuts, in this order: G7 (ErrP) → UI polish → closed-loop extras. **Never cut G0–G3.**

## Layout split (pre-written-code rules are unknown)
- `src/neurohack/` = generic, transferable infrastructure (sources, buffer, preprocessing, quality, decoders, closed loop, LLM client, eval).
- `apps/` = thin, idea-specific code. `scripts/` = offline analyses / figure generation.
- README "Built before the event" must list exactly what was prebuilt; update it when adding modules.

## Code style (Good Research Code Handbook)
- Python 3.11, type hints, small modules, pure functions where possible; state only where needed (filters, buffers, sources).
- Arrays are `data[n_channels, n_samples]` float64, timestamps `[n_samples]` on the LSL clock (`pylsl.local_clock()`).
- Configs in `configs/*.yaml`; **no magic numbers in app code** — load via `neurohack.config.load_config(name)`.
- `pytest` for tests (`uv run pytest`), `ruff` for lint (`uv run ruff check .`). Both must be green before reporting a gate.
- Windows-native: PowerShell commands, no WSL. Large data / recordings (`data/`, `sessions/`) stay out of git.

## Commands
```powershell
uv sync                                   # install
uv run pytest -m "not slow"               # fast tests
uv run ruff check . ; uv run ruff format --check .
uv run python scripts/g1_pipeline_bench.py
uv run python scripts/offline_moabb_fbcca.py
```
