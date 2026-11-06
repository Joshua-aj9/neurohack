"""YAML config loading. All tunables live in `configs/*.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load `configs/<name>.yaml` (name without extension) as a dict."""
    path = (config_dir or CONFIG_DIR) / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all(config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load every YAML in the config dir, keyed by stem."""
    d = config_dir or CONFIG_DIR
    return {p.stem: load_config(p.stem, d) for p in sorted(d.glob("*.yaml"))}


def frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range robust to rounding (stop included if within step/2)."""
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(n)]
