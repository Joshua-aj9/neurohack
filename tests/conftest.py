from __future__ import annotations

import numpy as np
import pytest

from neurohack.config import load_config
from neurohack.sources.base import unicorn_info_from_config
from neurohack.sources.synthetic import SyntheticSource


@pytest.fixture(scope="session")
def cfgs() -> dict:
    return {
        n: load_config(n)
        for n in ["device", "stim", "decoder", "quality", "llm", "closedloop", "synthetic"]
    }


@pytest.fixture
def info(cfgs):
    return unicorn_info_from_config(cfgs["device"])


@pytest.fixture
def synth(cfgs, info) -> SyntheticSource:
    return SyntheticSource(info, cfgs["synthetic"], realtime=False)


def sine(freq: float, n: int, fs: float, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / fs + phase)
