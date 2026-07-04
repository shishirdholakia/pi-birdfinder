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
class ImpulseCandidate:
    unit: str
    sample: int
    time_s: float
    abs_time: datetime
    snr: float
    prominence: float
    score: float
    peak_sample: int
    peak_time_s: float
    width_s: float
    quality: str

    def as_event_pick(self) -> EventPick:
        return EventPick(
            unit=self.unit,
            sample=self.sample,
            time_s=self.time_s,
            abs_time=self.abs_time,
            snr=self.snr,
            quality=self.quality,
        )


@dataclass
class BirdcallCandidate:
    unit: str
    start_s: float
    end_s: float
    center_s: float
    species_name: str
    confidence: float
    score: float
    source_count: int
    quality: str
    distance_to_hint_s: float = float("nan")


@dataclass
class BirdcallPick(EventPick):
    window_start_s: float
    window_end_s: float
    peak_time_s: float
    species_name: str
    confidence: float
    detector_score: float


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
    env_tau_s: Optional[float] = None
    tau_disagreement_s: Optional[float] = None
    quality: str = "ok"
    secondary_lags_s: Optional[np.ndarray] = None
    secondary_scores: Optional[np.ndarray] = None
    secondary_alpha_s: Optional[np.ndarray] = None
    physical_max_tau_s: Optional[float] = None
    selected_peak_rank: Optional[int] = None
    amplitude_scale: Optional[float] = None


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


def robust_envelope(x: np.ndarray, fs: int, smooth_ms: float = 0.0) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=float))
    win = max(1, int(round(smooth_ms * 1e-3 * fs)))
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def robust_mad_stats(x: np.ndarray) -> Tuple[float, float]:
    """Return robust median and Gaussian-scaled MAD for a 1-D score array."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    return med, max(mad, 1e-12)


def _first_threshold_crossing_before_peak(env: np.ndarray, peak_idx: int, fs: int, med: float, mad: float, lookback_s: float = 0.12, onset_z: float = 3.0) -> int:
    """Pick the first high-frequency envelope onset before a later peak/reverb maximum.

    The peak itself is useful for scoring, but for localization we want the
    leading broadband edge of the clap.  This finds the beginning of the final
    above-threshold run immediately preceding the peak.
    """
    if env.size == 0:
        return int(peak_idx)
    peak_idx = int(np.clip(peak_idx, 0, env.size - 1))
    start = max(0, peak_idx - int(round(float(lookback_s) * fs)))
    threshold = med + float(onset_z) * mad
    above = np.asarray(env[start:peak_idx + 1] > threshold, dtype=bool)
    if above.size == 0 or not np.any(above):
        return peak_idx

    # Work backwards from the peak to find the start of the contiguous
    # above-threshold run that contains, or immediately precedes, the peak.
    k = above.size - 1
    while k > 0 and not above[k]:
        k -= 1
    while k > 0 and above[k - 1]:
        k -= 1
    return int(start + k)


def find_impulse_candidates_in_unit(
    unit: str,
    uf: UnitFiles,
    clock: Any,
    highpass_hz: float = 500.0,
    bandpass_hz: Optional[Tuple[float, float]] = None,
    search_start_s: float = 0.0,
    search_end_s: Optional[float] = None,
    smooth_ms: float = 0.0,
    min_snr: float = 8.0,
    min_prominence: float = 6.0,
    min_separation_s: float = 0.25,
    top_k: int = 8,
    edge_guard_s: float = 0.25,
    onset_lookback_s: float = 0.12,
    onset_z: float = 3.0,
    max_width_s: float = 0.20,
) -> Sequence[ImpulseCandidate]:
    """Find candidate clap/impulse onsets anywhere in a unit clip.

    This is intentionally a robust high-frequency envelope detector, not a
    birdcall/spectrogram classifier. It high-pass/band-pass filters the audio,
    computes a short smoothed absolute-amplitude envelope, finds prominent
    robust-z envelope peaks, then moves each candidate pick back to the first
    threshold crossing before the peak so reverberant maxima are not used as
    the event time.
    """
    fs, nframes, dur_s = audio_info(uf.flac_path)
    if search_end_s is None:
        search_end_s = dur_s
    search_start_s = max(0.0, float(search_start_s))
    search_end_s = min(float(search_end_s), float(dur_s))
    if search_end_s <= search_start_s:
        return []

    seg, fs, start_sample = read_mono_segment(uf.flac_path, search_start_s, search_end_s - search_start_s)
    if seg.size == 0:
        return []
    filt = band_filter(seg, fs, highpass_hz=highpass_hz, bandpass_hz=bandpass_hz)
    env = robust_envelope(filt, fs, smooth_ms=smooth_ms)
    if env.size == 0:
        return []
    med, mad = robust_mad_stats(env)
    z = (env - med) / mad

    distance = max(1, int(round(float(min_separation_s) * fs)))
    peaks, props = signal.find_peaks(
        z,
        height=float(min_snr),
        prominence=float(min_prominence),
        distance=distance,
    )
    if peaks.size == 0:
        # Fallback: keep the best peak so diagnostics still say what was tried.
        peak = int(np.argmax(z))
        peaks = np.asarray([peak], dtype=int)
        props = {
            "peak_heights": np.asarray([float(z[peak])]),
            "prominences": np.asarray([0.0]),
        }

    try:
        widths = signal.peak_widths(z, peaks, rel_height=0.5)[0] / float(fs)
    except Exception:
        widths = np.zeros(peaks.size, dtype=float)

    candidates = []
    edge_guard = int(round(float(edge_guard_s) * fs))
    peak_heights = np.asarray(props.get("peak_heights", z[peaks]), dtype=float)
    prominences = np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=float)

    for i, peak_idx_raw in enumerate(peaks):
        peak_idx = int(peak_idx_raw)
        if edge_guard > 0 and (peak_idx < edge_guard or peak_idx > env.size - edge_guard - 1):
            continue
        onset_idx = _first_threshold_crossing_before_peak(
            env, peak_idx, fs, med, mad, lookback_s=onset_lookback_s, onset_z=onset_z
        )
        sample = int(start_sample + onset_idx)
        peak_sample = int(start_sample + peak_idx)
        snr = float(peak_heights[i])
        prominence = float(prominences[i])
        width_s = float(widths[i]) if i < len(widths) else 0.0
        width_penalty = max(0.0, width_s - float(max_width_s)) / max(float(max_width_s), 1e-6)
        score = float(snr + 0.75 * prominence - 4.0 * width_penalty)
        quality = "ok" if snr >= min_snr and prominence >= min_prominence else "low_snr"
        if width_s > max_width_s:
            quality = f"{quality}_wide"
        candidates.append(
            ImpulseCandidate(
                unit=unit,
                sample=sample,
                time_s=sample / float(fs),
                abs_time=clock.time_from_sample(sample),
                snr=snr,
                prominence=prominence,
                score=score,
                peak_sample=peak_sample,
                peak_time_s=peak_sample / float(fs),
                width_s=width_s,
                quality=quality,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    if top_k and top_k > 0:
        candidates = candidates[:int(top_k)]
    return candidates


def pick_impulse_anywhere_in_unit(
    unit: str,
    uf: UnitFiles,
    clock: Any,
    highpass_hz: float = 300.0,
    bandpass_hz: Optional[Tuple[float, float]] = None,
    min_snr: float = 8.0,
    min_prominence: float = 6.0,
    min_separation_s: float = 0.25,
    edge_guard_s: float = 0.25,
) -> EventPick:
    """Return the strongest whole-clip impulse candidate as an EventPick.

    This is a single-unit fallback. The localization script normally validates
    the top candidates across stations before choosing one.
    """
    candidates = find_impulse_candidates_in_unit(
        unit,
        uf,
        clock,
        highpass_hz=highpass_hz,
        bandpass_hz=bandpass_hz,
        min_snr=min_snr,
        min_prominence=min_prominence,
        min_separation_s=min_separation_s,
        top_k=1,
        edge_guard_s=edge_guard_s,
    )
    if candidates:
        return candidates[0].as_event_pick()
    fs, _, dur_s = audio_info(uf.flac_path)
    sample = int(round(0.5 * dur_s * fs))
    return EventPick(unit, sample, sample / fs, clock.time_from_sample(sample), 0.0, "empty")


def pick_impulse_in_unit(unit: str, uf: UnitFiles, clock: Any, guess_offset_s: Optional[float], search_half_s: float = 0.5, highpass_hz: float = 300.0, bandpass_hz: Optional[Tuple[float, float]] = None) -> EventPick:
    fs, nframes, dur_s = audio_info(uf.flac_path)
    if guess_offset_s is None:
        guess_offset_s = 0.5 * dur_s
    start_s = max(0.0, float(guess_offset_s) - float(search_half_s))
    seg, fs, start_sample = read_mono_segment(uf.flac_path, start_s, 2.0 * search_half_s)
    filt = band_filter(seg, fs, highpass_hz=highpass_hz, bandpass_hz=bandpass_hz)
    env = robust_envelope(filt, fs, smooth_ms=0.0)
    if env.size == 0:
        sample = int(round(guess_offset_s * fs))
        return EventPick(unit, sample, sample / fs, clock.time_from_sample(sample), 0.0, "empty")
    idx = int(np.argmax(env))
    med = float(np.median(env))
    mad = 1.4826 * float(np.median(np.abs(env - med))) + 1e-12
    snr = float((env[idx] - med) / mad)
    sample = int(start_sample + idx)
    return EventPick(unit, sample, sample / fs, clock.time_from_sample(sample), snr, "ok" if snr > 8 else "low_snr")


def refine_birdcall_pick(
    candidate: BirdcallCandidate,
    uf: UnitFiles,
    clock: Any,
    bandpass_hz: Tuple[float, float] = (1000.0, 9000.0),
    pad_s: float = 0.15,
    smooth_ms: float = 6.0,
    min_region_s: float = 0.04,
    hint_s: Optional[float] = None,
    hint_search_half_s: float = 0.35,
) -> BirdcallPick:
    """Refine a BirdNET 3 s candidate into a shorter reference pick.

    BirdNET provides a classification window, not a TDOA-quality event time.
    This uses band-limited energy inside that window to find the strongest
    short acoustic region while keeping the BirdNET interval for alignment.
    """
    fs, _, dur_s = audio_info(uf.flac_path)
    start_s = max(0.0, float(candidate.start_s) - float(pad_s))
    end_s = min(float(dur_s), float(candidate.end_s) + float(pad_s))
    if end_s <= start_s:
        sample = int(round(float(candidate.center_s) * fs))
        return BirdcallPick(
            unit=candidate.unit,
            sample=sample,
            time_s=sample / float(fs),
            abs_time=clock.time_from_sample(sample),
            snr=0.0,
            quality="empty",
            window_start_s=float(candidate.start_s),
            window_end_s=float(candidate.end_s),
            peak_time_s=sample / float(fs),
            species_name=candidate.species_name,
            confidence=float(candidate.confidence),
            detector_score=float(candidate.score),
        )

    seg, fs, start_sample = read_mono_segment(uf.flac_path, start_s, end_s - start_s)
    filt = band_filter(seg, fs, bandpass_hz=bandpass_hz)
    env = robust_envelope(filt, fs, smooth_ms=smooth_ms)
    if env.size == 0:
        idx = max(0, int(round((float(candidate.center_s) - start_s) * fs)))
        snr = 0.0
        quality = "empty"
    else:
        med, mad = robust_mad_stats(env)
        z = (env - med) / mad
        if hint_s is not None:
            hint_idx = int(round((float(hint_s) - start_s) * fs))
            lo = max(0, hint_idx - int(round(float(hint_search_half_s) * fs)))
            hi = min(z.size, hint_idx + int(round(float(hint_search_half_s) * fs)) + 1)
            if hi > lo:
                idx = int(lo + np.argmax(z[lo:hi]))
            else:
                idx = int(np.argmax(z))
        else:
            idx = int(np.argmax(z))
        snr = float(z[idx])
        threshold = max(float(med + 2.5 * mad), float(med + 0.20 * max(float(env[idx] - med), 0.0)))
        above = env > threshold
        if np.any(above):
            left = idx
            right = idx
            while left > 0 and above[left - 1]:
                left -= 1
            while right < env.size - 1 and above[right + 1]:
                right += 1
            if (right - left + 1) / float(fs) < float(min_region_s):
                half = int(round(0.5 * float(min_region_s) * fs))
                left = max(0, idx - half)
                right = min(env.size - 1, idx + half)
            idx = int(round(0.5 * (left + right)))
        quality = "ok" if snr >= 4.0 else "low_snr"

    sample = int(start_sample + idx)
    time_s = sample / float(fs)
    return BirdcallPick(
        unit=candidate.unit,
        sample=sample,
        time_s=time_s,
        abs_time=clock.time_from_sample(sample),
        snr=snr,
        quality=quality,
        window_start_s=float(candidate.start_s),
        window_end_s=float(candidate.end_s),
        peak_time_s=time_s,
        species_name=candidate.species_name,
        confidence=float(candidate.confidence),
        detector_score=float(candidate.score),
    )


def _normalized_xcorr_curve(sig: np.ndarray, refsig: np.ndarray, fs: int, max_tau_s: float) -> Tuple[np.ndarray, np.ndarray]:
    n = max(len(sig), len(refsig))
    if n <= 8:
        return np.zeros(1), np.zeros(1)
    x = np.pad(np.asarray(sig, dtype=float), (0, n - len(sig)))
    y = np.pad(np.asarray(refsig, dtype=float), (0, n - len(refsig)))
    x -= np.mean(x)
    y -= np.mean(y)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return np.zeros(1), np.zeros(1)
    cc = signal.correlate(x / sx, y / sy, mode="full", method="fft") / float(n)
    lag_samples = signal.correlation_lags(len(x), len(y), mode="full")
    max_shift = int(round(float(max_tau_s) * fs))
    keep = np.abs(lag_samples) <= max_shift
    return lag_samples[keep].astype(float) / float(fs), cc[keep].astype(float)


def _zscore_curve(score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, dtype=float)
    if s.size == 0:
        return s
    med, mad = robust_mad_stats(s)
    return (s - med) / mad


def gcc_phat_curve(sig: np.ndarray, refsig: np.ndarray, fs: int, max_tau_s: float, phat_exponent: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
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
    exponent = float(np.clip(phat_exponent, 0.0, 1.0))
    cc = np.fft.irfft(R / (denom ** exponent), n=nfft)
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


def _curve_peak_indices(lags: np.ndarray, score: np.ndarray, min_separation_s: float = 0.002, top_k: int = 8, lag_mask: Optional[np.ndarray] = None) -> np.ndarray:
    if score.size == 0:
        return np.zeros(0, dtype=int)
    if lag_mask is None:
        valid_indices = np.arange(score.size, dtype=int)
        search_score = score
    else:
        mask = np.asarray(lag_mask, dtype=bool)
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size == 0:
            return np.zeros(0, dtype=int)
        search_score = score[valid_indices]
    if lags.size > 1:
        step = max(float(abs(lags[1] - lags[0])), 1e-12)
        distance = max(1, int(round(float(min_separation_s) / step)))
    else:
        distance = 1
    peaks, _ = signal.find_peaks(search_score, distance=distance)
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(search_score))], dtype=int)
    window_best = int(np.argmax(search_score))
    peaks = np.unique(np.concatenate([peaks.astype(int), np.asarray([window_best], dtype=int)]))
    peaks = valid_indices[peaks]
    order = sorted(peaks, key=lambda k: float(score[int(k)]), reverse=True)
    return np.asarray(order[: int(top_k)], dtype=int)


def _tau_from_curve_at_index(lags: np.ndarray, score: np.ndarray, peak_index: int, secondary_mask: Optional[np.ndarray] = None) -> Tuple[float, float, Tuple[float, float], float, float]:
    if score.size == 0:
        return 0.0, 0.005, (-0.01, 0.01), 0.0, 1.0
    k = int(np.clip(int(peak_index), 0, score.size - 1))
    tau = float(lags[k])
    if 0 < k < score.size - 1:
        y0, y1, y2 = float(score[k - 1]), float(score[k]), float(score[k + 1])
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            delta = 0.5 * (y0 - y2) / den
            step = float(lags[1] - lags[0]) if len(lags) > 1 else 0.0
            tau += float(np.clip(delta, -1, 1)) * step

    s = np.asarray(score, dtype=float)
    if lags.size > 1:
        step_s = max(float(abs(lags[1] - lags[0])), 1e-12)
        half = max(8, int(round(0.015 / step_s)))
    else:
        half = 1
    lo_i = max(0, k - half)
    hi_i = min(s.size, k + half + 1)
    local_lags = lags[lo_i:hi_i]
    local_scores = s[lo_i:hi_i]
    z = (local_scores - float(s[k])) / max(1e-6, float(np.std(local_scores)))
    w = np.exp(np.clip(2.0 * z, -60, 0))
    w /= np.sum(w) if np.sum(w) > 0 else 1.0
    mean = float(np.sum(w * local_lags))
    sigma = float(np.sqrt(max(1e-12, np.sum(w * (local_lags - mean) ** 2))))
    cdf = np.cumsum(w)
    ci_lo = float(np.interp(0.025, cdf, local_lags))
    ci_hi = float(np.interp(0.975, cdf, local_lags))

    mask = np.ones_like(s, dtype=bool) if secondary_mask is None else np.asarray(secondary_mask, dtype=bool).copy()
    mask[max(0, k - 3):min(len(s), k + 4)] = False
    secondary = float(np.max(s[mask])) if np.any(mask) else 0.0
    ratio = secondary / max(float(s[k]), 1e-12)
    return tau, max(sigma, 1.0 / 48000.0), (ci_lo, ci_hi), float(s[k]), float(ratio)


def _signed_scale_for_lag(sig: np.ndarray, refsig: np.ndarray, fs: int, tau_s: float) -> float:
    """Fit sig ~= alpha * refsig at a candidate lag."""
    x = np.asarray(sig, dtype=float)
    y = np.asarray(refsig, dtype=float)
    n = min(x.size, y.size)
    if n <= 4:
        return float("nan")
    x = x[:n]
    y = y[:n]
    lag = int(round(float(tau_s) * fs))
    if lag >= 0:
        if lag >= n - 4:
            return float("nan")
        x_fit = x[lag:]
        y_fit = y[: n - lag]
    else:
        shift = -lag
        if shift >= n - 4:
            return float("nan")
        x_fit = x[: n - shift]
        y_fit = y[shift:]
    x_fit = x_fit - np.mean(x_fit)
    y_fit = y_fit - np.mean(y_fit)
    denom = float(np.dot(y_fit, y_fit))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(x_fit, y_fit) / denom)


def fine_align_impulse(unit: str, ref_unit: str, unit_files: Dict[str, UnitFiles], clocks: Dict[str, Any], ref_pick: EventPick, max_tau_s: float = 0.10, final_half_s: float = 0.25, highpass_hz: float = 300.0, bandpass_hz: Optional[Tuple[float, float]] = None, gcc_phat_exponent: float = 1.0) -> FineAlignment:
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
    lags, score = gcc_phat_curve(other_f, ref_f, fs, max_tau_s=max_tau_s, phat_exponent=gcc_phat_exponent)
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


def fine_align_birdcall(
    unit: str,
    ref_unit: str,
    unit_files: Dict[str, UnitFiles],
    clocks: Dict[str, Any],
    ref_pick: BirdcallPick,
    max_tau_s: float = 0.10,
    bandpass_hz: Tuple[float, float] = (1000.0, 9000.0),
    pad_before_s: float = 0.08,
    pad_after_s: float = 0.16,
    env_smooth_ms: float = 8.0,
    physical_max_tau_s: Optional[float] = None,
    physical_slack_s: float = 0.0015,
    timing_offset_shift_s: float = 0.0,
    gcc_phat_exponent: float = 1.0,
    envelope_weight: float = 0.25,
) -> FineAlignment:
    ref_uf = unit_files[ref_unit]
    fs, _, dur_s = audio_info(ref_uf.flac_path)
    ref_start_s = max(0.0, float(ref_pick.window_start_s) - float(pad_before_s))
    ref_end_s = min(float(dur_s), float(ref_pick.window_end_s) + float(pad_after_s))
    return fine_align_birdcall_interval(
        unit,
        ref_unit,
        unit_files,
        clocks,
        ref_pick,
        ref_start_s=ref_start_s,
        ref_end_s=ref_end_s,
        max_tau_s=max_tau_s,
        bandpass_hz=bandpass_hz,
        env_smooth_ms=env_smooth_ms,
        physical_max_tau_s=physical_max_tau_s,
        physical_slack_s=physical_slack_s,
        timing_offset_shift_s=timing_offset_shift_s,
        gcc_phat_exponent=gcc_phat_exponent,
        envelope_weight=envelope_weight,
    )


def fine_align_birdcall_interval(
    unit: str,
    ref_unit: str,
    unit_files: Dict[str, UnitFiles],
    clocks: Dict[str, Any],
    ref_pick: BirdcallPick,
    ref_start_s: float,
    ref_end_s: float,
    max_tau_s: float = 0.10,
    bandpass_hz: Tuple[float, float] = (1000.0, 9000.0),
    env_smooth_ms: float = 8.0,
    physical_max_tau_s: Optional[float] = None,
    physical_slack_s: float = 0.0015,
    timing_offset_shift_s: float = 0.0,
    gcc_phat_exponent: float = 1.0,
    envelope_weight: float = 0.25,
) -> FineAlignment:
    ref_uf = unit_files[ref_unit]
    uf = unit_files[unit]
    fs, _, dur_s = audio_info(ref_uf.flac_path)
    ref_start_s = max(0.0, min(float(ref_start_s), float(dur_s)))
    ref_end_s = max(0.0, min(float(ref_end_s), float(dur_s)))
    dur = max(0.05, ref_end_s - ref_start_s)
    ref_start = int(round(ref_start_s * fs))
    ref_start_abs = clocks[ref_unit].time_from_sample(ref_start)
    other_start = max(0, int(round(clocks[unit].sample_from_time(ref_start_abs))))
    other_start_abs = clocks[unit].time_from_sample(other_start)
    coarse_dt_s = (other_start_abs - ref_start_abs).total_seconds()

    ref_seg, fs_ref, _ = read_mono_segment(ref_uf.flac_path, ref_start / fs, dur)
    other_seg, fs_other, _ = read_mono_segment(uf.flac_path, other_start / fs, dur)
    if fs_ref != fs_other:
        raise RuntimeError(f"Sample-rate mismatch {unit}={fs_other} vs {ref_unit}={fs_ref}")

    n = min(len(ref_seg), len(other_seg))
    if n <= 16:
        raise RuntimeError(f"Birdcall alignment window too short for {unit}-{ref_unit}")
    ref_seg = ref_seg[:n]
    other_seg = other_seg[:n]

    ref_f = band_filter(ref_seg, fs, bandpass_hz=bandpass_hz)
    other_f = band_filter(other_seg, fs, bandpass_hz=bandpass_hz)
    taper = signal.windows.tukey(n, alpha=0.12)
    ref_t = ref_f * taper
    other_t = other_f * taper

    ref_env = robust_envelope(ref_t, fs, smooth_ms=env_smooth_ms)
    other_env = robust_envelope(other_t, fs, smooth_ms=env_smooth_ms)
    env_lags, env_score = _normalized_xcorr_curve(other_env, ref_env, fs, max_tau_s=max_tau_s)
    env_tau = float(env_lags[int(np.argmax(env_score))]) if env_score.size else 0.0

    gcc_lags, gcc_score = gcc_phat_curve(other_t, ref_t, fs, max_tau_s=max_tau_s, phat_exponent=gcc_phat_exponent)
    if gcc_lags.size and env_lags.size == gcc_lags.size and np.allclose(env_lags, gcc_lags):
        env_interp = env_score
    elif gcc_lags.size and env_lags.size:
        env_interp = np.interp(gcc_lags, env_lags, env_score)
    else:
        env_interp = np.zeros_like(gcc_score)

    env_weight = float(np.clip(float(envelope_weight), 0.0, 1.0))
    gcc_weight = max(0.0, 1.0 - env_weight)
    score = gcc_weight * _zscore_curve(gcc_score) + env_weight * _zscore_curve(env_interp)
    selection_mask = np.ones_like(score, dtype=bool)
    if physical_max_tau_s is not None and math.isfinite(float(physical_max_tau_s)):
        limit = float(physical_max_tau_s) + float(physical_slack_s)
        corrected_tdoa_for_lags = float(coarse_dt_s) + gcc_lags - float(timing_offset_shift_s)
        selection_mask = np.abs(corrected_tdoa_for_lags) <= limit
    peak_indices = _curve_peak_indices(gcc_lags, score, min_separation_s=0.002, top_k=8, lag_mask=selection_mask)
    peak_alphas = np.asarray([_signed_scale_for_lag(other_t, ref_t, fs, float(gcc_lags[int(k)])) for k in peak_indices], dtype=float)
    selected_rank = 1
    if peak_indices.size == 0:
        selected_idx = int(np.argmax(score)) if score.size else 0
    else:
        selected_idx = int(peak_indices[0])
        valid = [
            int(k)
            for i, k in enumerate(peak_indices)
            if i < peak_alphas.size
            and math.isfinite(float(peak_alphas[i]))
            and float(peak_alphas[i]) > 0.0
        ]
        if valid:
            selected_idx = valid[0]
        selected_rank = int(np.where(peak_indices == selected_idx)[0][0]) + 1
    tau, sigma, ci95, peak, sec_ratio = _tau_from_curve_at_index(gcc_lags, score, selected_idx, secondary_mask=selection_mask)
    selected_alpha = _signed_scale_for_lag(other_t, ref_t, fs, tau)
    disagreement = abs(float(tau) - float(env_tau))
    quality_flags = []
    if sec_ratio > 0.85:
        quality_flags.append("ambiguous_secondary")
    if env_weight > 0.0 and disagreement > 0.010:
        quality_flags.append("env_gcc_disagree")
    if physical_max_tau_s is not None and math.isfinite(float(physical_max_tau_s)):
        corrected_tau = float(coarse_dt_s) + float(tau) - float(timing_offset_shift_s)
        if abs(corrected_tau) > float(physical_max_tau_s) + float(physical_slack_s):
            quality_flags.append("outside_physical_bound")
        elif selected_rank > 1:
            quality_flags.append(f"selected_calibrated_physical_peak_rank_{selected_rank}")
    if not math.isfinite(float(selected_alpha)):
        quality_flags.append("amplitude_scale_nan")
    elif float(selected_alpha) <= 0.0:
        quality_flags.append("negative_amplitude_scale")
    if physical_max_tau_s is not None and math.isfinite(float(physical_max_tau_s)) and peak_indices.size == 0:
        quality_flags.append("no_peak_in_physical_window")
    if peak_indices.size and peak_alphas.size:
        has_positive_physical = any(
            math.isfinite(float(a)) and float(a) > 0.0
            for k, a in zip(peak_indices, peak_alphas)
        )
        if not has_positive_physical:
            quality_flags.append("no_positive_physical_peak")
    quality = "ok" if not quality_flags else ",".join(quality_flags)

    other_event_sample = int(round(other_start + (float(ref_pick.time_s) - ref_start_s + tau) * fs))
    return FineAlignment(
        unit=unit,
        ref_unit=ref_unit,
        tau_s=tau,
        tau_sigma_s=math.hypot(float(sigma), max(disagreement * 0.5, 0.0) if env_weight > 0.0 else 0.0),
        tau_ci95_s=ci95,
        peak_score=peak,
        secondary_ratio=sec_ratio,
        lags_s=gcc_lags,
        score=score,
        ref_start_sample=ref_start,
        other_start_sample=other_start,
        ref_start_time=ref_start_abs,
        other_start_time=other_start_abs,
        event_sample_other=other_event_sample,
        env_tau_s=env_tau,
        tau_disagreement_s=disagreement,
        quality=quality,
        secondary_lags_s=gcc_lags[peak_indices] if peak_indices.size else None,
        secondary_scores=score[peak_indices] if peak_indices.size else None,
        secondary_alpha_s=peak_alphas if peak_indices.size else None,
        physical_max_tau_s=float(physical_max_tau_s) if physical_max_tau_s is not None else None,
        selected_peak_rank=selected_rank,
        amplitude_scale=selected_alpha,
    )
