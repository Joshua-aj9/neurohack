"""Filter-bank CCA (Chen et al. 2015, J Neural Eng 12:046008). Training-free.

Sub-band m (1..M): Chebyshev-I bandpass [low_m, high], low_m = start + step*(m-1) (default 8m Hz),
zero-phase filtered on the decision window with odd-reflection padding.
Score per target = sum_m w(m) * rho_m^2, w(m) = m^-a + b (a=1.25, b=0.25).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from neurohack.decoders.base import Decoder
from neurohack.decoders.cca import max_canon_corr, reference


def design_filterbank(srate: float, cfg: dict) -> list[np.ndarray]:
    nyq = srate / 2
    high = min(cfg["subband_high_hz"], nyq - 1)
    bank = []
    for m in range(1, cfg["n_subbands"] + 1):
        lo = cfg["subband_low_start_hz"] + cfg["subband_low_step_hz"] * (m - 1)
        if lo >= high:
            raise ValueError(f"sub-band {m} low edge {lo} >= high edge {high}")
        ws_lo = max(lo - cfg["stop_margin_hz"], 0.5)
        ws_hi = min(high + cfg["stop_margin_high_hz"], nyq - 0.5)
        order, wn = signal.cheb1ord(
            [lo, high], [ws_lo, ws_hi], cfg["gpass_db"], cfg["gstop_db"], fs=srate
        )
        bank.append(
            signal.cheby1(order, cfg["ripple_db"], wn, btype="bandpass", fs=srate, output="sos")
        )
    return bank


def subband_weights(n_subbands: int, a: float, b: float) -> np.ndarray:
    m = np.arange(1, n_subbands + 1)
    return m ** (-a) + b


class FBCCADecoder(Decoder):
    def __init__(
        self,
        freqs: list[float],
        srate: float,
        fb_cfg: dict,
        n_harmonics: int = 5,
        temperature: float = 1.0,
    ) -> None:
        self.freqs = list(freqs)
        self.srate = srate
        self.n_harmonics = n_harmonics
        self.temperature = temperature
        self.cfg = fb_cfg
        if min(self.freqs) < fb_cfg["subband_low_start_hz"]:
            raise ValueError(
                f"lowest stimulus {min(self.freqs)} Hz is below the first sub-band edge "
                f"{fb_cfg['subband_low_start_hz']} Hz; lower decoder.fbcca.subband_low_start_hz"
            )
        self.bank = design_filterbank(srate, fb_cfg)
        self.weights = subband_weights(fb_cfg["n_subbands"], fb_cfg["weight_a"], fb_cfg["weight_b"])
        self.high = min(fb_cfg["subband_high_hz"], srate / 2)
        self.pad = int(round(fb_cfg["pad_s"] * srate))

    def subband_rhos(self, window: np.ndarray) -> np.ndarray:
        """rho[m, target]: max canonical correlation per sub-band and target."""
        n = window.shape[1]
        padlen = min(self.pad, n - 1)
        refs = [reference(f, n, self.srate, self.n_harmonics, self.high).T for f in self.freqs]
        rho = np.zeros((len(self.bank), len(self.freqs)))
        for m, sos in enumerate(self.bank):
            xm = signal.sosfiltfilt(sos, window, axis=1, padtype="odd", padlen=padlen).T
            for j, ref in enumerate(refs):
                rho[m, j] = max_canon_corr(xm, ref)
        return rho

    def scores(self, window: np.ndarray) -> np.ndarray:
        return self.weights @ (self.subband_rhos(window) ** 2)


def make_decoder(decoder_cfg: dict, freqs: list[float], srate: float) -> Decoder:
    """Factory from decoder.yaml."""
    from neurohack.decoders.cca import CCADecoder

    t = decoder_cfg.get("calibration", {}).get("temperature", 1.0)
    if decoder_cfg["method"] == "cca":
        return CCADecoder(freqs, srate, decoder_cfg["n_harmonics"], t)
    if decoder_cfg["method"] == "fbcca":
        return FBCCADecoder(freqs, srate, decoder_cfg["fbcca"], decoder_cfg["n_harmonics"], t)
    raise ValueError(decoder_cfg["method"])
