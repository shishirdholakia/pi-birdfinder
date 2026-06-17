#!/usr/bin/env python3
"""TDOA construction and correction helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np

from align import FineAlignment


@dataclass
class TDOAEstimate:
    unit: str
    ref_unit: str
    raw_s: float
    corrected_s: float
    sigma_s: float
    ci95_s: Tuple[float, float]
    tau_s: float
    coarse_dt_s: float
    calibration_correction_s: float
    quality_peak: float
    quality_secondary_ratio: float


def build_tdoa_from_alignment(aln: FineAlignment, timing_offsets_s: Dict[str, float] | None = None, extra_sigma_s: float = 0.0005) -> TDOAEstimate:
    coarse = (aln.other_start_time - aln.ref_start_time).total_seconds()
    raw = float(coarse + aln.tau_s)
    offsets = timing_offsets_s or {}
    corr_shift = float(offsets.get(aln.unit, 0.0) - offsets.get(aln.ref_unit, 0.0))
    corrected = raw - corr_shift
    sigma = math.hypot(float(aln.tau_sigma_s), float(extra_sigma_s))
    lo, hi = aln.tau_ci95_s
    ci = (float(coarse + lo - corr_shift), float(coarse + hi - corr_shift))
    return TDOAEstimate(
        unit=aln.unit,
        ref_unit=aln.ref_unit,
        raw_s=raw,
        corrected_s=corrected,
        sigma_s=sigma,
        ci95_s=ci,
        tau_s=float(aln.tau_s),
        coarse_dt_s=float(coarse),
        calibration_correction_s=-corr_shift,
        quality_peak=float(aln.peak_score),
        quality_secondary_ratio=float(aln.secondary_ratio),
    )


def expected_tdoa_s(pos_xy: Dict[str, np.ndarray], source_xy: np.ndarray, unit: str, ref_unit: str, c: float) -> float:
    du = float(np.linalg.norm(np.asarray(source_xy) - np.asarray(pos_xy[unit])))
    dr = float(np.linalg.norm(np.asarray(source_xy) - np.asarray(pos_xy[ref_unit])))
    return (du - dr) / float(c)


def physical_sanity(pos_xy: Dict[str, np.ndarray], tdoa_s: Dict[str, float], ref_unit: str, c: float) -> Dict[str, Dict[str, float]]:
    out = {}
    for u, tau in tdoa_s.items():
        if u == ref_unit or u not in pos_xy or ref_unit not in pos_xy:
            continue
        baseline = float(np.linalg.norm(pos_xy[u] - pos_xy[ref_unit]))
        dd = abs(float(c) * float(tau))
        out[u] = {"baseline_m": baseline, "abs_range_difference_m": dd, "ratio": dd / baseline if baseline > 0 else float("inf")}
    return out
