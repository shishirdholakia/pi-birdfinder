#!/usr/bin/env python3
"""Waveform and spectrogram diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from aru_io import UnitFiles, audio_info, read_mono_segment
from align import band_filter, EventPick


def _spec(ax, x, fs, title, fmin, fmax):
    if len(x) < 64:
        ax.set_title(title); return
    nperseg = min(512, max(64, 2 ** int(np.floor(np.log2(len(x)//4))) if len(x)//4 > 64 else 256))
    noverlap = int(0.75 * nperseg)
    f, t, S = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum", mode="magnitude")
    db = 20*np.log10(S + 1e-12)
    db -= np.nanmax(db) if np.isfinite(db).any() else 0
    keep = (f >= fmin) & (f <= fmax)
    ax.pcolormesh(t - t.mean(), f[keep] / 1000.0, db[keep], shading="auto", vmin=-80, vmax=0)
    ax.set_title(title)
    ax.set_ylabel("kHz")


def plot_alignment_diagnostics(outdir: Path, unit_files: Dict[str, UnitFiles], clocks: Dict[str, Any], ref_pick: EventPick, tdoa_s: Dict[str, float], ref_unit: str = "five", half_s: float = 0.6, highpass_hz: float = 300.0, bandpass_hz: Optional[Tuple[float, float]] = None, fmin_hz: float = 300.0, fmax_hz: float = 10000.0) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    units = [ref_unit] + [u for u in sorted(unit_files) if u != ref_unit]
    segs = {}
    fs0 = None
    for u in units:
        uf = unit_files[u]
        fs, _, _ = audio_info(uf.flac_path)
        fs0 = fs0 or fs
        if u == ref_unit:
            center_sample = ref_pick.sample
        else:
            center_sample = clocks[u].sample_from_time(ref_pick.abs_time)
        start_s = max(0.0, center_sample / fs - half_s)
        seg, fs, start_sample = read_mono_segment(uf.flac_path, start_s, 2*half_s)
        seg = band_filter(seg, fs, highpass_hz=highpass_hz, bandpass_hz=bandpass_hz)
        t = np.arange(len(seg))/fs + start_sample/fs - ref_pick.sample/fs
        segs[u] = (t, seg, fs)

    # Waveforms unshifted and shifted.
    for shifted in [False, True]:
        fig, axes = plt.subplots(len(units), 1, figsize=(10, 1.8*len(units)), sharex=True, constrained_layout=True)
        if len(units) == 1: axes = [axes]
        for ax, u in zip(axes, units):
            t, x, fs = segs[u]
            tshift = t - (tdoa_s.get(u, 0.0) if shifted and u != ref_unit else 0.0)
            scale = max(np.max(np.abs(x)), 1e-9)
            ax.plot(tshift, x/scale, linewidth=0.8)
            ax.axvline(0.0, color="0.2", linestyle="--", linewidth=1.0)
            ax.set_ylabel(u)
        axes[-1].set_xlabel("seconds relative to five event")
        fig.suptitle("Shifted waveforms" if shifted else "Unshifted rough-aligned waveforms")
        fig.savefig(outdir / ("waveforms_shifted.png" if shifted else "waveforms_unshifted.png"), dpi=180)
        plt.close(fig)

    # Spectrograms unshifted and shifted.
    for shifted in [False, True]:
        fig, axes = plt.subplots(len(units), 1, figsize=(10, 2.1*len(units)), sharex=True, constrained_layout=True)
        if len(units) == 1: axes = [axes]
        for ax, u in zip(axes, units):
            t, x, fs = segs[u]
            shift_s = tdoa_s.get(u, 0.0) if shifted and u != ref_unit else 0.0
            # pcolormesh x-axis is relative inside _spec; annotate shift in title.
            _spec(ax, x, fs, f"{u} ({'shifted by %.3f ms' % (-1000*shift_s) if shifted and u != ref_unit else 'unshifted'})", fmin_hz, fmax_hz)
            ax.axvline(0.0, color="w", linestyle="--", linewidth=1.0)
        axes[-1].set_xlabel("seconds around plotted window center")
        fig.suptitle("Shifted spectrograms" if shifted else "Unshifted rough-aligned spectrograms")
        fig.savefig(outdir / ("spectrograms_shifted.png" if shifted else "spectrograms_unshifted.png"), dpi=180)
        plt.close(fig)
