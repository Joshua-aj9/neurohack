"""Session folder: config snapshot, events.jsonl, results.json."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from neurohack.config import CONFIG_DIR, REPO_ROOT


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


ELECTRODE_CONDITIONS = ("gel", "dry", "saline", "none")  # none = headset not worn


def add_electrodes_arg(ap, required_for_live: bool = True) -> None:
    """Add --electrodes to an argparse parser (recorded in every session's results.json)."""
    note = " (required for LIVE runs)" if required_for_live else ""
    ap.add_argument(
        "--electrodes",
        choices=ELECTRODE_CONDITIONS,
        default=None,
        help=f"electrode contact condition{note}",
    )


def resolve_electrodes(value: str | None, live: bool, required: bool = True) -> str:
    """Electrode condition to record. LIVE runs must state it unless `required` is False."""
    if value is not None:
        return value
    if not live:
        return "n/a"
    if required:
        raise SystemExit(
            "LIVE run: say how the electrodes are prepared: --electrodes gel|dry|saline|none"
        )
    return "unknown"


class Session:
    def __init__(
        self, name: str, root: Path | None = None, label: str = "LIVE", electrodes: str = "n/a"
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{electrodes}" if electrodes not in ("n/a", "unknown") else ""
        self.dir = (root or REPO_ROOT / "sessions") / f"{stamp}_{name}{suffix}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config").mkdir(exist_ok=True)
        for p in CONFIG_DIR.glob("*.yaml"):
            shutil.copy2(p, self.dir / "config" / p.name)
        self.label = label  # SYNTHETIC | PUBLIC DATASET | LIVE
        self.electrodes = electrodes
        self._events = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self.log(
            "session_start", name=name, label=label, electrodes=electrodes, wall_time=time.time()
        )

    def log(self, event: str, t: float | None = None, **fields) -> None:
        import pylsl

        rec = {"t": pylsl.local_clock() if t is None else t, "event": event, **fields}
        self._events.write(json.dumps(rec, default=_default) + "\n")
        self._events.flush()

    def save_results(self, results: dict, name: str = "results.json") -> Path:
        path = self.dir / name
        path.write_text(
            json.dumps(
                {"label": self.label, "electrodes": self.electrodes, **results},
                indent=2,
                default=_default,
            ),
            encoding="utf-8",
        )
        return path

    def close(self) -> None:
        if not self._events.closed:
            self.log("session_end", wall_time=time.time())
            self._events.close()


def read_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
