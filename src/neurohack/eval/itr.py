"""Wolpaw information transfer rate."""

from __future__ import annotations

import math


def wolpaw_bits(n: int, p: float) -> float:
    """Bits per selection: log2 N + P log2 P + (1-P) log2((1-P)/(N-1)). Clipped at 0 below chance."""
    if n < 2:
        return 0.0
    if p <= 1.0 / n:
        return 0.0
    b = math.log2(n)
    if p < 1.0:
        b += p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
    return b


def itr_bits_per_min(n: int, p: float, seconds_per_selection: float) -> float:
    """Bits/min given the mean time per selection *including* gaps between selections."""
    return wolpaw_bits(n, p) * 60.0 / seconds_per_selection
