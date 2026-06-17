#!/usr/bin/env python3
"""Waveform and spectrogram diagnostics with correct absolute-time axes.

Bug fix summary:
- Unshifted waveform/spectrogram x-axes are now built from each unit's clock-map
  absolute sample times relative to the reference event time, instead of assuming
  all clipped files share the same sample-0 origin.
- Shifted spectrograms now actually shift the x-axis by the supplied TDOA rather
  than only mentioning the shift in the title.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from aru_io import UnitFiles, audio_info, read_mono_segment
from align import band_filter, EventPick


def _segment_time_axis(clock: Any, start_sample: int, n: int, fs: int, ref_abs_time) -> np.ndarray:
    """Return per-sample times in seconds relative to ref_abs_time.

    We use the absolute time of the segment start from the unit clock map, then add
    sample-relative offsets within the segment. This avoids falsely assuming that all
    clips started at the same absolute time.
    """
    start_abs = clock.time_from_sample(int(start_sample))
    t0 = (start_abs - ref_abs_time).total_seconds()
    return t0 + np.arange(n, dtype=float) / float(fs)


def _spec(ax, x, fs, title, fmin, fmax, *, t_offset_s: float = 0.0):
    if len(x) < 64:
        ax.set_title(title)
        return

    n_candidate = len(x) // 4
    if n_candidate > 64:
        nperseg = min(512, 2 ** int(np.floor(np.log2(n_candidate))))
    else:
        nperseg = min(256, len(x))
    nperseg = max(64, min(nperseg, len(x)))
    noverlap = int(0.75 * nperseg)

    f, t, S = signal.spectrogram(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    db = 20 * np.log10(S + 1e-12)
    if np.isfinite(db).any():
        db -= np.nanmax(db)
    keep = (f >= fmin) & (f <= fmax)
    if not np.any(keep):
        keep = np.ones_like(f, dtype=bool)

    # Center the spectrogram time axis at the middle of the plotted segment, then
    # place it on the common reference-event axis using t_offset_s.
    t_plot = (t - 0.5 * (t[0] + t[-1])) + float(t_offset_s)
    ax.pcolormesh(t_plot, f[keep] / 1000.0, db[keep], shading="auto", vmin=-80, vmax=0, rasterized=True)
    ax.set_title(title)
    ax.set_ylabel("kHz")


def plot_alignment_diagnostics(
    outdir: Path,
    unit_files: Dict[str, UnitFiles],
    clocks: Dict[str, Any],
    ref_pick: EventPick,
    tdoa_s: Dict[str, float],
    ref_unit: str = "five",
    half_s: float = 0.6,
    highpass_hz: float = 300.0,
    bandpass_hz: Optional[Tuple[float, float]] = None,
    fmin_hz: float = 300.0,
    fmax_hz: float = 10000.0,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    units = [ref_unit] + [u for u in sorted(unit_files) if u != ref_unit]
    segs = {}

    for u in units:
        uf = unit_files[u]
        fs, _, _ = audio_info(uf.flac_path)
        if u == ref_unit:
            center_sample = ref_pick.sample
        else:
            center_sample = clocks[u].sample_from_time(ref_pick.abs_time)
        start_s = max(0.0, center_sample / fs - half_s)
        seg, fs, start_sample = read_mono_segment(uf.flac_path, start_s, 2 * half_s)
        seg = band_filter(seg, fs, highpass_hz=highpass_hz, bandpass_hz=bandpass_hz)
        t_abs_rel = _segment_time_axis(clocks[u], start_sample, len(seg), fs, ref_pick.abs_time)
        segs[u] = {
            "t_abs_rel": t_abs_rel,
            "x": seg,
            "fs": fs,
            "start_sample": start_sample,
            "center_sample": center_sample,
        }

    # Waveforms: unshifted are on the absolute-time axis relative to the reference
    # event. Shifted subtract the measured TDOA so that the event should align to 0.
    for shifted in [False, True]:
        fig, axes = plt.subplots(len(units), 1, figsize=(10, 1.8 * len(units)), sharex=True, constrained_layout=True)
        if len(units) == 1:
            axes = [axes]
        for ax, u in zip(axes, units):
            t = segs[u]["t_abs_rel"].copy()
            x = segs[u]["x"]
            if shifted and u != ref_unit:
                t = t - float(tdoa_s.get(u, 0.0))
            scale = max(np.max(np.abs(x)), 1e-9)
            ax.plot(t, x / scale, linewidth=0.8)
            ax.axvline(0.0, color="0.2", linestyle="--", linewidth=1.0)
            ax.set_ylabel(u)
        axes[-1].set_xlabel("seconds relative to reference event time")
        fig.suptitle("Shifted waveforms" if shifted else "Unshifted waveforms on absolute-time axis")
        fig.savefig(outdir / ("waveforms_shifted.png" if shifted else "waveforms_unshifted.png"), dpi=180)
        plt.close(fig)

    # Spectrograms: same absolute-time axis logic as waveforms.
    for shifted in [False, True]:
        fig, axes = plt.subplots(len(units), 1, figsize=(10, 2.1 * len(units)), sharex=True, constrained_layout=True)
        if len(units) == 1:
            axes = [axes]
        for ax, u in zip(axes, units):
            x = segs[u]["x"]
            fs = segs[u]["fs"]
            t_abs_rel = segs[u]["t_abs_rel"]
            # Place the segment on the common axis by using the midpoint of the segment
            # in absolute time, then optionally subtract the measured TDOA.
            t_mid = 0.5 * (float(t_abs_rel[0]) + float(t_abs_rel[-1])) if len(t_abs_rel) else 0.0
            if shifted and u != ref_unit:
                t_mid = t_mid - float(tdoa_s.get(u, 0.0))
            desc = "shifted" if shifted and u != ref_unit else "unshifted"
            _spec(ax, x, fs, f"{u} ({desc})", fmin_hz, fmax_hz, t_offset_s=t_mid)
            ax.axvline(0.0, color="w", linestyle="--", linewidth=1.0)
        axes[-1].set_xlabel("seconds relative to reference event time")
        fig.suptitle("Shifted spectrograms" if shifted else "Unshifted spectrograms on absolute-time axis")
        fig.savefig(outdir / ("spectrograms_shifted.png" if shifted else "spectrograms_unshifted.png"), dpi=180)
        plt.close(fig)
