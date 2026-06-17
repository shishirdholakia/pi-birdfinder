#!/usr/bin/env python3
"""Impulse/birdcall alignment helpers for ARU event localization."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from aru_io import UnitFiles, audio_info, read_mono_segment


@dataclass
class EventPick:
    unit: str
    sample: int
    time_s: float
    abs_time: datetime
    snr: float
    quality: str


@dataclass
class FineAlignment:
    unit: str
    ref_unit: str
    tau_s: float
    tau_sigma_s: float
    tau_ci95_s: Tuple[float, float]
    peak_score: float
    secondary_ratio: float
    lags_s: np.ndarray
    score: np.ndarray
    ref_start_sample: int
    other_start_sample: int
    ref_start_time: datetime
    other_start_time: datetime
    event_sample_other: int


def band_filter(x: np.ndarray, fs: int, highpass_hz: Optional[float] = None, lowpass_hz: Optional[float] = None, bandpass_hz: Optional[Tuple[float, float]] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 16:
        return x.astype(np.float32)
    nyq = 0.5 * fs
    if bandpass_hz is not None:
        lo, hi = bandpass_hz
        lo = max(1.0, float(lo)) / nyq
        hi = min(float(hi), nyq * 0.98) / nyq
        if lo >= hi:
            return x.astype(np.float32)
        sos = signal.butter(4, [lo, hi], btype="bandpass", output="sos")
    elif highpass_hz is not None and highpass_hz > 0:
        sos = signal.butter(4, float(highpass_hz) / nyq, btype="highpass", output="sos")
    elif lowpass_hz is not None and lowpass_hz > 0:
        sos = signal.butter(4, min(float(lowpass_hz), nyq * 0.98) / nyq, btype="lowpass", output="sos")
    else:
        return (x - np.mean(x)).astype(np.float32)
    try:
        return signal.sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return signal.sosfilt(sos, x).astype(np.float32)


def robust_envelope(x: np.ndarray, fs: int, smooth_ms: float = 5.0) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=float))
    win = max(1, int(round(smooth_ms * 1e-3 * fs)))
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def pick_impulse_in_unit(unit: str, uf: UnitFiles, clock: Any, guess_offset_s: Optional[float], search_half_s: float = 0.5, highpass_hz: float = 300.0, bandpass_hz: Optional[Tuple[float, float]] = None) -> EventPick:
    fs, nframes, dur_s = audio_info(uf.flac_path)
    if guess_offset_s is None:
        guess_offset_s = 0.5 * dur_s
    start_s = max(0.0, float(guess_offset_s) - float(search_half_s))
    seg, fs, start_sample = read_mono_segment(uf.flac_path, start_s, 2.0 * search_half_s)
    filt = band_filter(seg, fs, highpass_hz=highpass_hz, bandpass_hz=bandpass_hz)
    env = robust_envelope(filt, fs, smooth_ms=3.0)
    if env.size == 0:
        sample = int(round(guess_offset_s * fs))
        return EventPick(unit, sample, sample / fs, clock.time_from_sample(sample), 0.0, "empty")
    idx = int(np.argmax(env))
    med = float(np.median(env))
    mad = 1.4826 * float(np.median(np.abs(env - med))) + 1e-12
    snr = float((env[idx] - med) / mad)
    sample = int(start_sample + idx)
    return EventPick(unit, sample, sample / fs, clock.time_from_sample(sample), snr, "ok" if snr > 8 else "low_snr")


def gcc_phat_curve(sig: np.ndarray, refsig: np.ndarray, fs: int, max_tau_s: float) -> Tuple[np.ndarray, np.ndarray]:
    n = max(len(sig), len(refsig))
    if n <= 8:
        return np.zeros(1), np.zeros(1)
    x = np.pad(np.asarray(sig, dtype=float), (0, n - len(sig)))
    y = np.pad(np.asarray(refsig, dtype=float), (0, n - len(refsig)))
    x -= np.mean(x)
    y -= np.mean(y)
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    X = np.fft.rfft(x, n=nfft)
    Y = np.fft.rfft(y, n=nfft)
    R = X * np.conj(Y)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    cc = np.fft.irfft(R / denom, n=nfft)
    cc = np.concatenate((cc[-(nfft // 2):], cc[: nfft // 2]))
    mid = nfft // 2
    max_shift = int(round(min(max_tau_s * fs, mid - 1)))
    score = cc[mid - max_shift: mid + max_shift + 1]
    lags = np.arange(-max_shift, max_shift + 1, dtype=float) / fs
    return lags, score


def tau_from_curve(lags: np.ndarray, score: np.ndarray, temperature: float = 12.0) -> Tuple[float, float, Tuple[float, float], float, float]:
    if score.size == 0:
        return 0.0, 0.005, (-0.01, 0.01), 0.0, 1.0
    k = int(np.argmax(score))
    # parabolic interpolation around peak
    tau = float(lags[k])
    if 0 < k < score.size - 1:
        y0, y1, y2 = float(score[k-1]), float(score[k]), float(score[k+1])
        den = y0 - 2*y1 + y2
        if abs(den) > 1e-12:
            delta = 0.5 * (y0 - y2) / den
            step = float(lags[1] - lags[0]) if len(lags) > 1 else 0.0
            tau += float(np.clip(delta, -1, 1)) * step
    # Convert correlation curve to a soft likelihood. This is intentionally rough but stable.
    s = np.asarray(score, dtype=float)
    z = (s - np.max(s)) / max(1e-6, float(np.std(s)))
    w = np.exp(np.clip(2.0 * z, -60, 0))
    w /= np.sum(w) if np.sum(w) > 0 else 1.0
    mean = float(np.sum(w * lags))
    sigma = float(np.sqrt(max(1e-12, np.sum(w * (lags - mean) ** 2))))
    cdf = np.cumsum(w)
    lo = float(np.interp(0.025, cdf, lags))
    hi = float(np.interp(0.975, cdf, lags))
    # secondary ratio: highest peak outside +/- 3 samples of main peak
    mask = np.ones_like(s, dtype=bool)
    mask[max(0, k-3):min(len(s), k+4)] = False
    secondary = float(np.max(s[mask])) if np.any(mask) else 0.0
    ratio = secondary / max(float(s[k]), 1e-12)
    return tau, max(sigma, 1.0 / 48000.0), (lo, hi), float(s[k]), float(ratio)


def fine_align_impulse(unit: str, ref_unit: str, unit_files: Dict[str, UnitFiles], clocks: Dict[str, Any], ref_pick: EventPick, max_tau_s: float = 0.10, final_half_s: float = 0.25, highpass_hz: float = 300.0, bandpass_hz: Optional[Tuple[float, float]] = None) -> FineAlignment:
    ref_uf = unit_files[ref_unit]
    uf = unit_files[unit]
    fs, _, _ = audio_info(ref_uf.flac_path)
    other_est_sample = clocks[unit].sample_from_time(ref_pick.abs_time)
    ref_start = max(0, int(round(ref_pick.sample - final_half_s * fs)))
    other_start = max(0, int(round(other_est_sample - final_half_s * fs)))
    ref_seg, fs_ref, _ = read_mono_segment(ref_uf.flac_path, ref_start / fs, 2.0 * final_half_s)
    other_seg, fs_other, _ = read_mono_segment(uf.flac_path, other_start / fs, 2.0 * final_half_s)
    if fs_ref != fs_other:
        raise RuntimeError(f"Sample-rate mismatch {unit}={fs_other} vs {ref_unit}={fs_ref}")
    ref_f = band_filter(ref_seg, fs, highpass_hz=highpass_hz, bandpass_hz=bandpass_hz)
    other_f = band_filter(other_seg, fs, highpass_hz=highpass_hz, bandpass_hz=bandpass_hz)
    lags, score = gcc_phat_curve(other_f, ref_f, fs, max_tau_s=max_tau_s)
    tau, sigma, ci95, peak, sec_ratio = tau_from_curve(lags, score)
    other_event_sample = int(round(other_start + final_half_s * fs + tau * fs))
    return FineAlignment(
        unit=unit,
        ref_unit=ref_unit,
        tau_s=tau,
        tau_sigma_s=sigma,
        tau_ci95_s=ci95,
        peak_score=peak,
        secondary_ratio=sec_ratio,
        lags_s=lags,
        score=score,
        ref_start_sample=ref_start,
        other_start_sample=other_start,
        ref_start_time=clocks[ref_unit].time_from_sample(ref_start),
        other_start_time=clocks[unit].time_from_sample(other_start),
        event_sample_other=other_event_sample,
    )
