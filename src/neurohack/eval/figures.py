"""Standard figures. Every figure carries its data label (SYNTHETIC / PUBLIC DATASET / LIVE) in the title."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"axes.titlesize": 9})
BLUE, ORANGE, GREY = "#2f6fb0", "#d9822b", "#8a8a8a"


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def accuracy_vs_window(
    lengths,
    curves: dict[str, np.ndarray],
    chance: float,
    label: str,
    path: Path,
    per_subject: dict[str, np.ndarray] | None = None,
) -> Path:
    """curves: name -> mean accuracy [n_lens]; per_subject: name -> [n_subj, n_lens] (thin lines)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [BLUE, ORANGE, "#3a9a5b", "#9a3a8a"]
    for (name, acc), c in zip(curves.items(), colors, strict=False):
        if per_subject and name in per_subject:
            for row in per_subject[name]:
                ax.plot(lengths, row, color=c, alpha=0.15, lw=0.8)
        ax.plot(lengths, acc, color=c, lw=2.2, marker="o", ms=3, label=name)
    ax.axhline(chance, color=GREY, ls="--", lw=1, label=f"chance ({chance:.2f})")
    ax.set_xlabel("window length (s)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Accuracy vs window — {label}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def psd_plot(
    f: np.ndarray,
    pxx_by_cond: dict[str, np.ndarray],
    stim_freqs: list[float],
    label: str,
    path: Path,
    fmax: float = 40.0,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    m = f <= fmax
    for name, p in pxx_by_cond.items():
        ax.semilogy(f[m], p[m], lw=1.4, label=name)
    for sf in stim_freqs:
        ax.axvline(sf, color=GREY, ls=":", lw=1)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"PSD — {label}")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def confusion_plot(cm: np.ndarray, class_names: list[str], label: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion — {label}")
    return _save(fig, path)


def reliability_plot(
    rel_before: dict, rel_after: dict, ece_before: float, ece_after: float, label: str, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    for rel, name, c in [
        (rel_before, f"T=1 (ECE {ece_before:.3f})", ORANGE),
        (rel_after, f"fitted T (ECE {ece_after:.3f})", BLUE),
    ]:
        m = rel["count"] > 0
        ax.plot(rel["confidence"][m], rel["accuracy"][m], marker="o", color=c, label=name)
    ax.set_xlabel("confidence (max p)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Calibration — {label}")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def bar_compare(names: list[str], values: list[float], ylabel: str, label: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(names, values, color=[GREY, BLUE][: len(names)] + [BLUE] * max(0, len(names) - 2))
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    return _save(fig, path)
