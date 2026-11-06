"""Readers for datasets that might be handed out on the day: EDF/BDF/FIF/BrainVision/EEGLAB/.mat/.csv.

All return an MNE Raw (data in volts) so they plug into `replay.from_raw` and MNE tooling.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np


def read_raw_any(
    path: str | Path,
    srate: float | None = None,
    ch_names: list[str] | None = None,
    mat_key: str | None = None,
    unit_scale: float = 1e-6,
    preload: bool = True,
) -> mne.io.BaseRaw:
    """Read a file into MNE Raw.

    For .mat/.csv (no metadata), `srate` is required; data is assumed [n_ch, n] or [n, n_ch]
    (the longer axis is taken as time) in microvolts unless `unit_scale` says otherwise.
    """
    path = Path(path)
    ext = path.suffix.lower()
    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".gdf": mne.io.read_raw_gdf,
        ".fif": mne.io.read_raw_fif,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
    }
    if ext in readers:
        return readers[ext](str(path), preload=preload, verbose=False)
    if ext in (".mat", ".csv", ".txt", ".npy"):
        if srate is None:
            raise ValueError(f"{ext} files need srate=")
        arr = _load_array(path, mat_key)
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        names = ch_names or [f"ch{i}" for i in range(arr.shape[0])]
        info = mne.create_info(names, srate, ch_types="eeg")
        return mne.io.RawArray(arr * unit_scale, info, verbose=False)
    raise ValueError(f"unsupported file type: {ext}")


def _load_array(path: Path, mat_key: str | None) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".csv", ".txt"):
        try:
            return np.loadtxt(path, delimiter=",")
        except ValueError:
            return np.loadtxt(path, delimiter=",", skiprows=1)
    from scipy.io import loadmat

    m = loadmat(str(path))
    keys = [k for k in m if not k.startswith("__")]
    key = mat_key or max(keys, key=lambda k: np.asarray(m[k]).size)
    return np.asarray(m[key], dtype=float).squeeze()


def describe(raw: mne.io.BaseRaw) -> str:
    """One-paragraph summary for first-hour triage of an unknown dataset."""
    ann = raw.annotations
    desc = sorted(set(ann.description)) if len(ann) else []
    stim = mne.pick_types(raw.info, stim=True)
    return (
        f"{len(raw.ch_names)} ch @ {raw.info['sfreq']} Hz, {raw.n_times / raw.info['sfreq']:.1f} s; "
        f"channels={raw.ch_names[:20]}{'...' if len(raw.ch_names) > 20 else ''}; "
        f"annotations={desc[:20]}; stim_channels={[raw.ch_names[i] for i in stim]}"
    )
