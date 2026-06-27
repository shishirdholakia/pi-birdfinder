#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple
import math
import csv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from aru_io import (
    find_unit_files, load_station_locations, stations_to_xy, xy_to_latlon, parse_offset_seconds,
    fit_clock_maps, audio_info, read_mono_segment, read_sectioned_tables
)
from align import (
    BirdcallPick,
    EventPick,
    band_filter,
    find_impulse_candidates_in_unit,
    fine_align_birdcall,
    fine_align_impulse,
    pick_impulse_in_unit,
    refine_birdcall_pick,
    robust_envelope,
)
from tdoa import build_tdoa_from_alignment, physical_sanity
from calibration import extract_timing_offsets_from_tables
from localize import localize_least_squares, leave_one_out_localizations, mc_location_samples, covariance_ellipse
from maps import write_event_map
from viz import plot_alignment_diagnostics


def load_calibration(path: Path, ref_unit: str):
    if path is None or not Path(path).exists():
        return {}, None
    kv, tables = read_sectioned_tables(Path(path))
    offsets = extract_timing_offsets_from_tables(tables)
    station_latlon = {}
    for row in tables.get("stations", []):
        u = row.get("unit", "")
        if not u:
            continue
        lat = row.get("fit_lat", "") or row.get("lat", "")
        lon = row.get("fit_lon", "") or row.get("lon", "")
        try:
            station_latlon[u] = (float(lat), float(lon))
        except Exception:
            pass
    return offsets, station_latlon if station_latlon else None


def write_tdoa_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def _align_all_units_for_pick(
    ref_pick: EventPick,
    units,
    ref_unit: str,
    unit_files: Dict[str, Any],
    clocks: Dict[str, Any],
    timing_offsets: Dict[str, float],
    max_tau_s: float,
    highpass_hz: float,
    bandpass_hz,
    gcc_phat_exponent: float = 1.0,
):
    """Run existing GCC/TDOA alignment for one candidate reference pick."""
    tdoa_rows = []
    tdoa_corr = {ref_unit: 0.0}
    sigma = {}
    raw_plot = {ref_unit: 0.0}
    alignments = {}
    for u in units:
        if u == ref_unit or u not in unit_files:
            continue
        aln = fine_align_impulse(
            u,
            ref_unit,
            unit_files,
            clocks,
            ref_pick,
            max_tau_s=max_tau_s,
            highpass_hz=highpass_hz,
            bandpass_hz=bandpass_hz,
            gcc_phat_exponent=gcc_phat_exponent,
        )
        est = build_tdoa_from_alignment(aln, timing_offsets_s=timing_offsets)
        alignments[u] = aln
        tdoa_corr[u] = est.corrected_s
        raw_plot[u] = est.raw_s
        sigma[u] = est.sigma_s
        tdoa_rows.append({
            "unit": u,
            "ref_unit": ref_unit,
            "raw_tdoa_s": f"{est.raw_s:+.9f}",
            "corrected_tdoa_s": f"{est.corrected_s:+.9f}",
            "sigma_s": f"{est.sigma_s:.9f}",
            "ci95_low_s": f"{est.ci95_s[0]:+.9f}",
            "ci95_high_s": f"{est.ci95_s[1]:+.9f}",
            "gcc_tau_s": f"{est.tau_s:+.9f}",
            "coarse_dt_s": f"{est.coarse_dt_s:+.9f}",
            "calibration_correction_s": f"{est.calibration_correction_s:+.9f}",
            "quality_peak": f"{est.quality_peak:.6f}",
            "quality_secondary_ratio": f"{est.quality_secondary_ratio:.6f}",
        })
    return tdoa_corr, sigma, raw_plot, tdoa_rows, alignments


def _align_all_units_for_birdcall(
    ref_pick: BirdcallPick,
    units,
    ref_unit: str,
    unit_files: Dict[str, Any],
    clocks: Dict[str, Any],
    timing_offsets: Dict[str, float],
    max_tau_s: float,
    bandpass_hz,
    pad_before_s: float,
    pad_after_s: float,
    pos_xy,
    sound_speed: float,
    gcc_phat_exponent: float,
    physical_slack_s: float,
    envelope_weight: float,
):
    tdoa_rows = []
    tdoa_corr = {ref_unit: 0.0}
    sigma = {}
    raw_plot = {ref_unit: 0.0}
    alignments = {}
    for u in units:
        if u == ref_unit or u not in unit_files:
            continue
        try:
            physical_max_tau_s = None
            if u in pos_xy and ref_unit in pos_xy:
                physical_max_tau_s = float(np.linalg.norm(pos_xy[u] - pos_xy[ref_unit])) / max(float(sound_speed), 1e-9)
            aln = fine_align_birdcall(
                u,
                ref_unit,
                unit_files,
                clocks,
                ref_pick,
                max_tau_s=max_tau_s,
                bandpass_hz=bandpass_hz,
                pad_before_s=pad_before_s,
                pad_after_s=pad_after_s,
                physical_max_tau_s=physical_max_tau_s,
                physical_slack_s=physical_slack_s,
                timing_offset_shift_s=float(timing_offsets.get(u, 0.0) - timing_offsets.get(ref_unit, 0.0)),
                gcc_phat_exponent=gcc_phat_exponent,
                envelope_weight=envelope_weight,
            )
            est = build_tdoa_from_alignment(aln, timing_offsets_s=timing_offsets)
            alignments[u] = aln
            tdoa_corr[u] = est.corrected_s
            raw_plot[u] = est.raw_s
            sigma[u] = est.sigma_s
            tdoa_rows.append({
                "unit": u,
                "ref_unit": ref_unit,
                "raw_tdoa_s": f"{est.raw_s:+.9f}",
                "corrected_tdoa_s": f"{est.corrected_s:+.9f}",
                "sigma_s": f"{est.sigma_s:.9f}",
                "ci95_low_s": f"{est.ci95_s[0]:+.9f}",
                "ci95_high_s": f"{est.ci95_s[1]:+.9f}",
                "gcc_tau_s": f"{est.tau_s:+.9f}",
                "coarse_dt_s": f"{est.coarse_dt_s:+.9f}",
                "calibration_correction_s": f"{est.calibration_correction_s:+.9f}",
                "quality_peak": f"{est.quality_peak:.6f}",
                "quality_secondary_ratio": f"{est.quality_secondary_ratio:.6f}",
                "env_tau_s": f"{aln.env_tau_s:+.9f}" if aln.env_tau_s is not None else "",
                "env_gcc_disagreement_s": f"{aln.tau_disagreement_s:.9f}" if aln.tau_disagreement_s is not None else "",
                "physical_max_tau_s": f"{aln.physical_max_tau_s:.9f}" if aln.physical_max_tau_s is not None else "",
                "selected_peak_rank": aln.selected_peak_rank if aln.selected_peak_rank is not None else "",
                "selected_amplitude_scale": f"{aln.amplitude_scale:.9f}" if aln.amplitude_scale is not None else "",
                "candidate_peak_lags_s": ";".join(f"{x:+.9f}" for x in (aln.secondary_lags_s if aln.secondary_lags_s is not None else [])),
                "candidate_peak_scores": ";".join(f"{x:.6f}" for x in (aln.secondary_scores if aln.secondary_scores is not None else [])),
                "candidate_peak_amplitude_scales": ";".join(f"{x:.9f}" for x in (aln.secondary_alpha_s if aln.secondary_alpha_s is not None else [])),
                "alignment_quality": aln.quality,
                "error": "",
            })
        except Exception as e:
            tdoa_rows.append({
                "unit": u,
                "ref_unit": ref_unit,
                "raw_tdoa_s": "",
                "corrected_tdoa_s": "",
                "sigma_s": "",
                "ci95_low_s": "",
                "ci95_high_s": "",
                "gcc_tau_s": "",
                "coarse_dt_s": "",
                "calibration_correction_s": "",
                "quality_peak": "",
                "quality_secondary_ratio": "",
                "env_tau_s": "",
                "env_gcc_disagreement_s": "",
                "physical_max_tau_s": "",
                "selected_peak_rank": "",
                "selected_amplitude_scale": "",
                "candidate_peak_lags_s": "",
                "candidate_peak_scores": "",
                "candidate_peak_amplitude_scales": "",
                "alignment_quality": "failed",
                "error": str(e),
            })
    return tdoa_corr, sigma, raw_plot, tdoa_rows, alignments


def _birdcall_candidate_options(aln, timing_offsets: Dict[str, float], max_peak_rank: int = 3, physical_slack_s: float = 0.0015):
    lags = np.asarray(aln.secondary_lags_s if aln.secondary_lags_s is not None else [], dtype=float)
    scores = np.asarray(aln.secondary_scores if aln.secondary_scores is not None else [], dtype=float)
    alphas = np.asarray(aln.secondary_alpha_s if aln.secondary_alpha_s is not None else [], dtype=float)
    if lags.size == 0:
        lags = np.asarray([float(aln.tau_s)], dtype=float)
        scores = np.asarray([float(aln.peak_score)], dtype=float)
        alphas = np.asarray([float(aln.amplitude_scale) if aln.amplitude_scale is not None else float("nan")], dtype=float)

    coarse = (aln.other_start_time - aln.ref_start_time).total_seconds()
    corr_shift = float(timing_offsets.get(aln.unit, 0.0) - timing_offsets.get(aln.ref_unit, 0.0))
    out = []
    for idx, lag in enumerate(lags[:max(1, int(max_peak_rank))]):
        score = float(scores[idx]) if idx < scores.size else float("nan")
        alpha = float(alphas[idx]) if idx < alphas.size else float("nan")
        lag = float(lag)
        if not math.isfinite(lag):
            continue
        if not math.isfinite(alpha) or alpha <= 0.0:
            continue
        raw = float(coarse + lag)
        corrected = raw - corr_shift
        if aln.physical_max_tau_s is not None and math.isfinite(float(aln.physical_max_tau_s)):
            if abs(corrected) > float(aln.physical_max_tau_s) + float(physical_slack_s):
                continue
        out.append({
            "unit": aln.unit,
            "rank": idx + 1,
            "lag_s": lag,
            "score": score,
            "alpha": alpha,
            "raw_s": raw,
            "corrected_s": corrected,
        })
    return out


def _birdcall_tdoa_rows_from_alignments(units, ref_unit: str, alignments: Dict[str, Any], timing_offsets: Dict[str, float]):
    tdoa_rows = []
    tdoa_corr = {ref_unit: 0.0}
    sigma = {}
    raw_plot = {ref_unit: 0.0}
    for u in units:
        if u == ref_unit:
            continue
        aln = alignments.get(u)
        if aln is None:
            continue
        est = build_tdoa_from_alignment(aln, timing_offsets_s=timing_offsets)
        tdoa_corr[u] = est.corrected_s
        raw_plot[u] = est.raw_s
        sigma[u] = est.sigma_s
        tdoa_rows.append({
            "unit": u,
            "ref_unit": ref_unit,
            "raw_tdoa_s": f"{est.raw_s:+.9f}",
            "corrected_tdoa_s": f"{est.corrected_s:+.9f}",
            "sigma_s": f"{est.sigma_s:.9f}",
            "ci95_low_s": f"{est.ci95_s[0]:+.9f}",
            "ci95_high_s": f"{est.ci95_s[1]:+.9f}",
            "gcc_tau_s": f"{est.tau_s:+.9f}",
            "coarse_dt_s": f"{est.coarse_dt_s:+.9f}",
            "calibration_correction_s": f"{est.calibration_correction_s:+.9f}",
            "quality_peak": f"{est.quality_peak:.6f}",
            "quality_secondary_ratio": f"{est.quality_secondary_ratio:.6f}",
            "env_tau_s": f"{aln.env_tau_s:+.9f}" if aln.env_tau_s is not None else "",
            "env_gcc_disagreement_s": f"{aln.tau_disagreement_s:.9f}" if aln.tau_disagreement_s is not None else "",
            "physical_max_tau_s": f"{aln.physical_max_tau_s:.9f}" if aln.physical_max_tau_s is not None else "",
            "selected_peak_rank": aln.selected_peak_rank if aln.selected_peak_rank is not None else "",
            "selected_amplitude_scale": f"{aln.amplitude_scale:.9f}" if aln.amplitude_scale is not None else "",
            "candidate_peak_lags_s": ";".join(f"{x:+.9f}" for x in (aln.secondary_lags_s if aln.secondary_lags_s is not None else [])),
            "candidate_peak_scores": ";".join(f"{x:.6f}" for x in (aln.secondary_scores if aln.secondary_scores is not None else [])),
            "candidate_peak_amplitude_scales": ";".join(f"{x:.9f}" for x in (aln.secondary_alpha_s if aln.secondary_alpha_s is not None else [])),
            "alignment_quality": aln.quality,
            "error": "",
        })
    return tdoa_corr, sigma, raw_plot, tdoa_rows


def _select_birdcall_geometric_peak_combo(
    outdir: Path,
    ref_pick: BirdcallPick,
    units,
    ref_unit: str,
    alignments: Dict[str, Any],
    timing_offsets: Dict[str, float],
    pos_xy,
    ref_latlon,
    sound_speed: float,
    unit_files: Dict[str, Any],
    physical_slack_s: float,
    max_peak_rank: int = 3,
):
    """Select one candidate GCC peak per station by minimizing the global TDOA fit residual."""
    candidate_options = {}
    rejected_rows = []
    for u in units:
        if u == ref_unit:
            continue
        aln = alignments.get(u)
        if aln is None:
            continue
        options = _birdcall_candidate_options(
            aln,
            timing_offsets,
            max_peak_rank=max_peak_rank,
            physical_slack_s=physical_slack_s,
        )
        if options:
            candidate_options[u] = options
        else:
            rejected_rows.append({
                "unit": u,
                "reason": f"no_positive_candidate_peak_inside_physical_window_top_{max_peak_rank}",
            })

    available = [u for u in units if u in candidate_options and u in pos_xy]
    combo_rows = []
    best = None
    if len(available) >= 2:
        combo_sizes = [len(available)] + list(range(len(available) - 1, 1, -1))
        combo_rank = 0
        for size in combo_sizes:
            any_valid_for_size = False
            for unit_subset in itertools.combinations(available, size):
                for selected in itertools.product(*(candidate_options[u] for u in unit_subset)):
                    measured = {item["unit"]: float(item["corrected_s"]) for item in selected}
                    sigma = {
                        item["unit"]: math.hypot(max(float(alignments[item["unit"]].tau_sigma_s), 1e-6), 0.0005)
                        for item in selected
                    }
                    try:
                        source_xy, residuals, cost = localize_least_squares(
                            pos_xy,
                            measured,
                            sigma_s=sigma,
                            ref_unit=ref_unit,
                            c=sound_speed,
                        )
                    except Exception:
                        continue
                    if not math.isfinite(float(cost)) or not np.all(np.isfinite(source_xy)):
                        continue
                    any_valid_for_size = True
                    combo_rank += 1
                    ll = xy_to_latlon(float(source_xy[0]), float(source_xy[1]), ref_latlon)
                    row = {
                        "combo_rank": combo_rank,
                        "selected": "",
                        "n_units": len(unit_subset),
                        "units": ",".join(unit_subset),
                        "weighted_cost": f"{float(cost):.9f}",
                        "source_x_m": f"{float(source_xy[0]):.6f}",
                        "source_y_m": f"{float(source_xy[1]):.6f}",
                        "source_lat": f"{ll[0]:.9f}",
                        "source_lon": f"{ll[1]:.9f}",
                    }
                    for item in selected:
                        unit = item["unit"]
                        row[f"{unit}_peak_rank"] = item["rank"]
                        row[f"{unit}_gcc_tau_s"] = f"{item['lag_s']:+.9f}"
                        row[f"{unit}_corrected_tdoa_s"] = f"{item['corrected_s']:+.9f}"
                        row[f"{unit}_peak_score"] = f"{item['score']:.6f}"
                        row[f"{unit}_amplitude_scale"] = f"{item['alpha']:.9f}"
                        row[f"{unit}_residual_s"] = f"{float(residuals.get(unit, float('nan'))):+.9f}"
                    combo_rows.append(row)
                    key = (
                        float(cost),
                        -len(unit_subset),
                        -float(np.mean([item["score"] for item in selected if math.isfinite(float(item["score"]))]) if selected else 0.0),
                    )
                    if best is None or key < best[0]:
                        best = (key, selected, source_xy, residuals, cost)
            if best is not None and any_valid_for_size:
                break

    if best is None:
        for row in rejected_rows:
            combo_rows.append({
                "combo_rank": "",
                "selected": "",
                "n_units": "",
                "units": row["unit"],
                "weighted_cost": "",
                "source_x_m": "",
                "source_y_m": "",
                "source_lat": "",
                "source_lon": "",
                "rejection_reason": row["reason"],
            })
        write_tdoa_csv(outdir / "geometric_peak_combos.csv", combo_rows)
        return _birdcall_tdoa_rows_from_alignments(units, ref_unit, alignments, timing_offsets)

    _, selected, _, _, cost = best
    selected_by_unit = {item["unit"]: item for item in selected}
    for row in combo_rows:
        if row.get("n_units") != len(selected):
            continue
        if row.get("units") != ",".join(item["unit"] for item in selected):
            continue
        matches = True
        for item in selected:
            if str(row.get(f"{item['unit']}_peak_rank", "")) != str(item["rank"]):
                matches = False
                break
        if matches:
            row["selected"] = "yes"
            break

    for unit, item in selected_by_unit.items():
        aln = alignments[unit]
        old_quality = str(aln.quality or "ok")
        flags = [] if old_quality == "ok" else [old_quality]
        flags.append(f"geometric_combo_peak_rank_{item['rank']}")
        aln.tau_s = float(item["lag_s"])
        aln.peak_score = float(item["score"])
        aln.selected_peak_rank = int(item["rank"])
        aln.amplitude_scale = float(item["alpha"])
        aln.tau_disagreement_s = abs(float(aln.tau_s) - float(aln.env_tau_s)) if aln.env_tau_s is not None else aln.tau_disagreement_s
        aln.quality = ",".join(flags)
        try:
            fs_u, _, _ = audio_info(unit_files[unit].flac_path)
            ref_pick_offset_s = (ref_pick.abs_time - aln.ref_start_time).total_seconds()
            aln.event_sample_other = int(round(
                aln.other_start_sample + (ref_pick_offset_s + aln.tau_s) * float(fs_u)
            ))
        except Exception:
            pass

    write_tdoa_csv(outdir / "geometric_peak_combos.csv", combo_rows)
    print(
        f"Geometric peak combo: selected {len(selected)} stations "
        f"with weighted_cost={float(cost):.6f}; wrote {outdir / 'geometric_peak_combos.csv'}"
    )
    return _birdcall_tdoa_rows_from_alignments(units, ref_unit, alignments, timing_offsets)


def _plot_birdcall_correlation_diagnostics(outdir: Path, alignments: Dict[str, Any], pos_xy, ref_unit: str, sound_speed: float, timing_offsets: Dict[str, float] | None = None, physical_slack_s: float = 0.0015) -> None:
    corr_dir = outdir / "diagnostics" / "correlation"
    corr_dir.mkdir(parents=True, exist_ok=True)
    for unit, aln in alignments.items():
        lags = np.asarray(aln.lags_s, dtype=float)
        score = np.asarray(aln.score, dtype=float)
        if lags.size == 0 or score.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(lags, score, linewidth=1.0, color="0.2")
        ax.axvline(float(aln.tau_s), color="tab:red", linewidth=1.6, label=f"selected {aln.tau_s:+.6f}s")
        if aln.env_tau_s is not None:
            ax.axvline(float(aln.env_tau_s), color="tab:green", linestyle="--", linewidth=1.2, label=f"envelope {aln.env_tau_s:+.6f}s")
        if unit in pos_xy and ref_unit in pos_xy:
            physical = float(np.linalg.norm(pos_xy[unit] - pos_xy[ref_unit])) / max(float(sound_speed), 1e-9)
            offsets = timing_offsets or {}
            coarse = (aln.other_start_time - aln.ref_start_time).total_seconds()
            corr_shift = float(offsets.get(unit, 0.0) - offsets.get(ref_unit, 0.0))
            lag_center = corr_shift - coarse
            ax.axvline(lag_center - physical, color="tab:blue", linestyle=":", linewidth=1.0, label="calibrated physical bounds")
            ax.axvline(lag_center + physical, color="tab:blue", linestyle=":", linewidth=1.0)
            ax.axvspan(
                lag_center - physical - float(physical_slack_s),
                lag_center - physical,
                color="tab:blue",
                alpha=0.08,
                linewidth=0,
                label="physical padding",
            )
            ax.axvspan(
                lag_center + physical,
                lag_center + physical + float(physical_slack_s),
                color="tab:blue",
                alpha=0.08,
                linewidth=0,
            )

        candidate_lags = np.asarray(aln.secondary_lags_s if aln.secondary_lags_s is not None else [], dtype=float)
        candidate_scores = np.asarray(aln.secondary_scores if aln.secondary_scores is not None else [], dtype=float)
        candidate_alphas = np.asarray(aln.secondary_alpha_s if aln.secondary_alpha_s is not None else [], dtype=float)
        if candidate_lags.size == 0:
            candidate_lags = np.asarray([float(aln.tau_s)])
            candidate_scores = np.asarray([float(aln.peak_score)])
            candidate_alphas = np.asarray([float(aln.amplitude_scale) if aln.amplitude_scale is not None else float("nan")])
        for rank, (lag, peak_score, alpha) in enumerate(zip(candidate_lags[:8], candidate_scores[:8], candidate_alphas[:8]), start=1):
            inside = True
            if aln.physical_max_tau_s is not None:
                offsets = timing_offsets or {}
                coarse = (aln.other_start_time - aln.ref_start_time).total_seconds()
                corr_shift = float(offsets.get(unit, 0.0) - offsets.get(ref_unit, 0.0))
                corrected = float(coarse) + float(lag) - corr_shift
                inside = abs(corrected) <= float(aln.physical_max_tau_s) + float(physical_slack_s)
            positive = math.isfinite(float(alpha)) and float(alpha) > 0.0
            selected = rank == int(aln.selected_peak_rank or 1)
            color = "tab:red" if selected else ("tab:orange" if inside and positive else "0.55")
            marker = "o" if inside and positive else "x"
            ax.plot(lag, peak_score, marker=marker, color=color)
            alpha_label = f"a={alpha:+.2g}" if math.isfinite(float(alpha)) else "a=nan"
            ax.annotate(
                f"p{rank}\n{lag:+.5f}s\n{alpha_label}",
                xy=(lag, peak_score),
                xytext=(4, 8),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_title(f"Birdcall GCC/correlation: {unit} vs {ref_unit} ({aln.quality})")
        ax.set_xlabel("lag seconds")
        ax.set_ylabel("combined GCC/envelope score")
        ax.legend(loc="best", fontsize=8)
        fig.savefig(corr_dir / f"gcc_{unit}_vs_{ref_unit}.png", dpi=180)
        plt.close(fig)


def _plot_birdcall_envelope_diagnostics(
    outdir: Path,
    unit_files: Dict[str, Any],
    clocks: Dict[str, Any],
    ref_pick: BirdcallPick,
    alignments: Dict[str, Any],
    ref_unit: str,
    bandpass_hz,
    pad_before_s: float,
    pad_after_s: float,
) -> None:
    env_dir = outdir / "diagnostics" / "envelope"
    env_dir.mkdir(parents=True, exist_ok=True)
    first = next(iter(alignments.values()), None)
    if first is None:
        return
    fs, _, dur_s = audio_info(unit_files[ref_unit].flac_path)
    ref_start_s = max(0.0, float(ref_pick.window_start_s) - float(pad_before_s))
    ref_end_s = min(float(dur_s), float(ref_pick.window_end_s) + float(pad_after_s))
    read_dur_s = max(0.05, ref_end_s - ref_start_s)

    for unit in [ref_unit] + [u for u in sorted(alignments)]:
        if unit == ref_unit:
            start_sample = int(first.ref_start_sample)
            selected_time_s = 0.0
        else:
            aln = alignments[unit]
            start_sample = int(aln.other_start_sample)
            selected_time_s = (
                (aln.other_start_time - ref_pick.abs_time).total_seconds()
                + (float(aln.event_sample_other) - float(aln.other_start_sample)) / float(fs)
            )
        seg, fs_u, actual_start = read_mono_segment(unit_files[unit].flac_path, start_sample / float(fs), read_dur_s)
        filt = band_filter(seg, fs_u, bandpass_hz=bandpass_hz)
        env = robust_envelope(filt, fs_u, smooth_ms=8.0)
        start_abs = clocks[unit].time_from_sample(actual_start)
        t = (start_abs - ref_pick.abs_time).total_seconds() + np.arange(len(filt), dtype=float) / float(fs_u)

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        if len(filt) >= 64:
            nperseg = min(1024, max(128, 2 ** int(np.floor(np.log2(max(64, len(filt) // 8))))))
            noverlap = int(0.75 * nperseg)
            f, tt, S = signal.spectrogram(
                filt,
                fs=fs_u,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="spectrum",
                mode="magnitude",
            )
            db = 20 * np.log10(S + 1e-12)
            db -= np.nanmax(db) if np.isfinite(db).any() else 0.0
            keep = (f >= float(bandpass_hz[0])) & (f <= float(bandpass_hz[1]))
            t_spec = (start_abs - ref_pick.abs_time).total_seconds() + tt
            ax.pcolormesh(t_spec, f[keep] / 1000.0, db[keep], shading="auto", vmin=-80, vmax=0, rasterized=True)
        ax.axvline(0.0, color="w", linestyle="--", linewidth=1.0, label="ref pick")
        ax.axvline(selected_time_s, color="tab:red", linestyle="-", linewidth=1.2, label="selected event")
        ax.set_title(f"Birdcall alignment envelope: {unit}")
        ax.set_xlabel("seconds relative to reference pick")
        ax.set_ylabel("kHz")
        ax2 = ax.twinx()
        scale = max(float(np.max(env)) if env.size else 0.0, 1e-12)
        ax2.plot(t[: len(env)], env / scale, color="tab:cyan", linewidth=1.2, label="band-limited envelope")
        ax2.set_ylabel("normalized envelope")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
        fig.savefig(env_dir / f"envelope_{unit}.png", dpi=180)
        plt.close(fig)


def _matched_spectrogram_db(x: np.ndarray, fs: int, nperseg: int = 1024, noverlap: int = 768, normalize_db: bool = True):
    if len(x) < 64:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))
    nperseg = min(int(nperseg), len(x))
    if nperseg < 64:
        nperseg = min(64, len(x))
    noverlap = min(int(noverlap), max(0, nperseg - 1))
    f, t, S = signal.spectrogram(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    db = 20 * np.log10(S + 1e-12)
    if normalize_db and np.isfinite(db).any():
        db -= np.nanmax(db)
    return f, t, db


def _plot_birdcall_residual_spectrograms(
    outdir: Path,
    unit_files: Dict[str, Any],
    clocks: Dict[str, Any],
    ref_pick: BirdcallPick,
    tdoa_s: Dict[str, float],
    ref_unit: str,
    bandpass_hz,
    half_s: float = 0.65,
) -> None:
    """Plot aligned spectrogram residuals, non-reference minus reference."""
    res_dir = outdir / "diagnostics" / "spectrogram_residuals"
    res_dir.mkdir(parents=True, exist_ok=True)
    fs, _, _ = audio_info(unit_files[ref_unit].flac_path)
    ref_center_sample = int(ref_pick.sample)
    ref_start_s = max(0.0, ref_center_sample / float(fs) - float(half_s))
    ref_seg, fs_ref, _ = read_mono_segment(unit_files[ref_unit].flac_path, ref_start_s, 2.0 * float(half_s))
    ref_filt = band_filter(ref_seg, fs_ref, bandpass_hz=bandpass_hz)
    nperseg = min(1024, max(128, 2 ** int(np.floor(np.log2(max(64, len(ref_filt) // 6))))))
    noverlap = int(0.75 * nperseg)
    f_ref_base, t_ref_base, db_ref_base = _matched_spectrogram_db(ref_filt, fs_ref, nperseg=nperseg, noverlap=noverlap, normalize_db=False)
    if db_ref_base.size == 0:
        return
    keep = (f_ref_base >= float(bandpass_hz[0])) & (f_ref_base <= float(bandpass_hz[1]))
    if not np.any(keep):
        keep = np.ones_like(f_ref_base, dtype=bool)
    t_plot = t_ref_base - 0.5 * (t_ref_base[0] + t_ref_base[-1])
    scale_rows = []

    for unit in sorted(unit_files):
        if unit == ref_unit:
            continue
        arrival_abs = ref_pick.abs_time + timedelta(seconds=float(tdoa_s.get(unit, 0.0)))
        fs_unit, _, _ = audio_info(unit_files[unit].flac_path)
        center_sample = clocks[unit].sample_from_time(arrival_abs)
        start_s = max(0.0, center_sample / float(fs_unit) - float(half_s))
        seg, fs_u, _ = read_mono_segment(unit_files[unit].flac_path, start_s, 2.0 * float(half_s))
        filt = band_filter(seg, fs_u, bandpass_hz=bandpass_hz)
        alpha = 1.0
        scaled_ref = alpha * ref_filt
        _, _, db_ref = _matched_spectrogram_db(scaled_ref, fs_ref, nperseg=nperseg, noverlap=noverlap, normalize_db=False)
        f_u, t_u, db_u = _matched_spectrogram_db(filt, fs_u, nperseg=nperseg, noverlap=noverlap, normalize_db=False)
        if db_u.size == 0 or db_ref.size == 0:
            continue
        min_t = min(db_ref.shape[1], db_u.shape[1])
        min_f = min(db_ref.shape[0], db_u.shape[0])
        residual = db_u[:min_f, :min_t] - db_ref[:min_f, :min_t]
        f_plot = f_ref_base[:min_f]
        keep_plot = keep[:min_f]
        t_use = t_plot[:min_t]
        rms_db = float(np.sqrt(np.nanmean(residual[keep_plot] ** 2))) if np.any(keep_plot) else float("nan")
        scale_rows.append({
            "unit": unit,
            "ref_unit": ref_unit,
            "ref_amplitude_scale": f"{alpha:.9f}",
            "ref_signed_least_squares_scale": "",
            "scale_quality": "unscaled",
            "residual_rms_db": f"{rms_db:.6f}",
        })

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        img = ax.pcolormesh(
            t_use,
            f_plot[keep_plot] / 1000.0,
            residual[keep_plot],
            shading="auto",
            cmap="coolwarm",
            vmin=-24,
            vmax=24,
            rasterized=True,
        )
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(f"Aligned spectrogram residual: {unit} - {alpha:.3g} * {ref_unit}")
        ax.set_xlabel("seconds relative to aligned event")
        ax.set_ylabel("kHz")
        cbar = fig.colorbar(img, ax=ax, pad=0.02)
        cbar.set_label("dB residual")
        fig.savefig(res_dir / f"residual_{unit}_minus_{ref_unit}.png", dpi=180)
        plt.close(fig)

    if scale_rows:
        write_tdoa_csv(res_dir / "residual_amplitude_scales.csv", scale_rows)


def _birdnet_pick_and_align(
    args,
    units,
    unit_files,
    clocks,
    timing_offsets,
    bandpass_hz,
    pos_xy,
    ref_latlon,
    outdir: Path,
):
    from aru_birdnet_clip_detect import run_clip_detection, select_reference_candidate
    from birdnet_detect import write_birdnet_candidates_csv

    hint_s = parse_offset_seconds(args.event_ref_offset) if args.event_ref_offset else None
    result = run_clip_detection(
        Path(args.event_dir),
        units=units,
        out_dir=outdir,
        write_csv=True,
        min_units=args.birdnet_min_units,
        window_primary_confidence=args.birdnet_window_primary_confidence,
        window_other_confidence=args.birdnet_window_other_confidence,
        locations=Path(args.locations) if args.locations else None,
        birdnet_species_list=Path(args.birdnet_species_list) if args.birdnet_species_list else None,
        birdnet_species_unit=args.birdnet_species_unit,
        birdnet_geo_week=args.birdnet_geo_week,
        birdnet_geo_min_confidence=args.birdnet_geo_min_confidence,
        no_birdnet_location_species_list=args.no_birdnet_location_species_list,
        threshold=args.birdnet_threshold,
        top_k=args.birdnet_top_k,
        overlap_s=args.birdnet_overlap_s,
        bandpass_fmin=args.birdnet_bandpass_fmin,
        bandpass_fmax=args.birdnet_bandpass_fmax,
        species_query=args.birdnet_species_query,
        merge_gap_s=args.birdnet_merge_gap_s,
        max_calls=0,
        max_candidates=args.birdnet_max_candidates,
        hint_s=hint_s,
        batch_size=args.birdnet_batch_size,
        workers=args.birdnet_workers,
        device=args.birdnet_device,
        write_outputs=True,
    )
    if args.ref not in result.predictions_by_unit:
        raise SystemExit(f"BirdNET detection did not produce predictions for reference unit {args.ref!r}")
    candidate, candidates, selected_call = select_reference_candidate(
        result,
        args.ref,
        hint_s=hint_s,
        hint_fallback_half_s=args.bird_refine_hint_half_s,
    )
    pred_path = outdir / "birdnet_predictions.csv"
    result.predictions_by_unit[args.ref].to_csv(pred_path, index=False)
    write_birdnet_candidates_csv(outdir / "birdnet_candidates.csv", candidates, selected=candidate)
    ref_pick = refine_birdcall_pick(
        candidate,
        unit_files[args.ref],
        clocks[args.ref],
        bandpass_hz=bandpass_hz,
        hint_s=hint_s,
        hint_search_half_s=args.bird_refine_hint_half_s,
    )
    _fs_ref, _n_ref, ref_dur_s = audio_info(unit_files[args.ref].flac_path)
    effective_align_start_s = max(0.0, float(candidate.start_s) - float(args.bird_align_pad_before_s))
    effective_align_end_s = min(float(ref_dur_s), float(candidate.end_s) + float(args.bird_align_pad_after_s))
    effective_align_duration_s = max(0.0, effective_align_end_s - effective_align_start_s)
    tdoa_corr, sigma, raw_plot, tdoa_rows, alignments = _align_all_units_for_birdcall(
        ref_pick,
        units,
        args.ref,
        unit_files,
        clocks,
        timing_offsets,
        args.max_tau_s,
        bandpass_hz,
        args.bird_align_pad_before_s,
        args.bird_align_pad_after_s,
        pos_xy,
        args.sound_speed,
        args.gcc_phat_exponent,
        float(args.bird_physical_slack_m) / max(float(args.sound_speed), 1e-9),
        args.bird_envelope_weight,
    )
    if args.no_geometric_peak_selection:
        write_tdoa_csv(outdir / "geometric_peak_combos.csv", [{
            "selected": "",
            "status": "disabled",
            "reason": "--no-geometric-peak-selection",
        }])
    else:
        tdoa_corr, sigma, raw_plot, tdoa_rows = _select_birdcall_geometric_peak_combo(
            outdir,
            ref_pick,
            units,
            args.ref,
            alignments,
            timing_offsets,
            pos_xy,
            ref_latlon,
            args.sound_speed,
            unit_files,
            float(args.bird_physical_slack_m) / max(float(args.sound_speed), 1e-9),
            max_peak_rank=args.bird_geometric_max_peak_rank,
        )
    print(
        f"BirdNET pick: ref={args.ref} species={candidate.species_name!r} "
        f"window={candidate.start_s:.3f}-{candidate.end_s:.3f}s "
        f"conf={candidate.confidence:.3f} refined_time={ref_pick.time_s:.6f}s"
    )
    print(f"Wrote BirdNET predictions: {pred_path}")
    print(f"Wrote BirdNET candidates: {outdir / 'birdnet_candidates.csv'}")
    return ref_pick, tdoa_corr, sigma, raw_plot, tdoa_rows, alignments, {
        "birdnet_pick": True,
        "candidate": candidate,
        "predictions_csv": pred_path,
        "candidates_csv": outdir / "birdnet_candidates.csv",
        "calls_txt": result.out_txt,
        "calls_csv": outdir / "birdnet_calls.csv",
        "selected_call": selected_call,
        "effective_align_start_s": effective_align_start_s,
        "effective_align_end_s": effective_align_end_s,
        "effective_align_duration_s": effective_align_duration_s,
        "geometric_peak_combos_csv": outdir / "geometric_peak_combos.csv",
        "species_list": result.species_list_path,
    }


def _candidate_validation_score(candidate, tdoa_corr, sigma, tdoa_rows, pos_xy, ref_unit: str, sound_speed: float, physical_ratio_slack: float):
    """Score an auto-picked clap candidate using multi-unit physical consistency."""
    measured = {u: t for u, t in tdoa_corr.items() if u != ref_unit}
    sanity = physical_sanity(pos_xy, measured, ref_unit, sound_speed) if measured else {}
    ratios = []
    for row in sanity.values():
        try:
            ratios.append(float(row.get("ratio", float("nan"))))
        except Exception:
            pass
    finite_ratios = [r for r in ratios if math.isfinite(r)]
    physical_ok = sum(1 for r in finite_ratios if r <= physical_ratio_slack)
    physical_bad = max(0, len(finite_ratios) - physical_ok)

    peaks = []
    secondary = []
    for row in tdoa_rows:
        try:
            peaks.append(float(row.get("quality_peak", 0.0)))
            secondary.append(float(row.get("quality_secondary_ratio", 1.0)))
        except Exception:
            pass
    mean_peak = float(np.mean(peaks)) if peaks else 0.0
    median_secondary = float(np.median(secondary)) if secondary else 1.0

    loc_cost = float("nan")
    if len(measured) >= 2:
        try:
            _, _, loc_cost = localize_least_squares(pos_xy, measured, sigma_s=sigma, ref_unit=ref_unit, c=sound_speed)
            loc_cost = float(loc_cost)
        except Exception:
            loc_cost = float("nan")

    # Primary term: how many stations produce physically plausible TDOAs.
    # Secondary terms: strong local clap score, stronger GCC peak, lower ambiguous secondary peak, lower localization cost.
    cost_penalty = 0.25 * loc_cost if math.isfinite(loc_cost) else 0.0
    score = (
        1000.0 * physical_ok
        - 500.0 * physical_bad
        + 5.0 * float(getattr(candidate, "score", 0.0))
        + 100.0 * mean_peak
        - 50.0 * median_secondary
        - cost_penalty
    )
    return score, {
        "n_physical_ok": physical_ok,
        "n_physical_bad": physical_bad,
        "mean_gcc_peak": mean_peak,
        "median_secondary_ratio": median_secondary,
        "localization_cost": loc_cost,
        "max_physical_ratio": max(finite_ratios) if finite_ratios else float("nan"),
    }


def _auto_pick_impulse_with_validation(
    args,
    units,
    unit_files,
    clocks,
    timing_offsets,
    pos_xy,
    highpass_hz: float,
    bandpass_hz,
    outdir: Path,
):
    """Detect clap candidates anywhere in the reference clip and choose the best cross-unit candidate."""
    candidates = list(find_impulse_candidates_in_unit(
        args.ref,
        unit_files[args.ref],
        clocks[args.ref],
        highpass_hz=highpass_hz,
        bandpass_hz=bandpass_hz,
        min_snr=args.auto_clap_min_snr,
        min_prominence=args.auto_clap_min_prominence,
        min_separation_s=args.auto_clap_min_separation_s,
        top_k=args.auto_clap_top_k,
        edge_guard_s=args.auto_clap_edge_guard_s,
        max_width_s=args.auto_clap_max_width_s,
    ))
    if not candidates:
        # Last-resort auto fallback: do not revert to midpoint; allow edge candidates.
        candidates = list(find_impulse_candidates_in_unit(
            args.ref,
            unit_files[args.ref],
            clocks[args.ref],
            highpass_hz=highpass_hz,
            bandpass_hz=bandpass_hz,
            min_snr=0.0,
            min_prominence=0.0,
            min_separation_s=args.auto_clap_min_separation_s,
            top_k=1,
            edge_guard_s=0.0,
            max_width_s=args.auto_clap_max_width_s,
        ))
    if not candidates:
        raise SystemExit("Auto clap detection found no usable candidates in the reference clip")

    best = None
    validation_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        ref_pick = candidate.as_event_pick()
        try:
            tdoa_corr, sigma, raw_plot, tdoa_rows, alignments = _align_all_units_for_pick(
                ref_pick,
                units,
                args.ref,
                unit_files,
                clocks,
                timing_offsets,
                args.max_tau_s,
                highpass_hz,
                bandpass_hz,
            )
            validation_score, detail = _candidate_validation_score(
                candidate,
                tdoa_corr,
                sigma,
                tdoa_rows,
                pos_xy,
                args.ref,
                args.sound_speed,
                args.auto_clap_physical_ratio_slack,
            )
            row = {
                "rank": rank,
                "selected": "",
                "time_s": f"{candidate.time_s:.6f}",
                "sample": candidate.sample,
                "peak_time_s": f"{candidate.peak_time_s:.6f}",
                "snr": f"{candidate.snr:.3f}",
                "prominence": f"{candidate.prominence:.3f}",
                "width_s": f"{candidate.width_s:.6f}",
                "detector_score": f"{candidate.score:.3f}",
                "validation_score": f"{validation_score:.3f}",
                "quality": candidate.quality,
                "error": "",
                **{k: (f"{v:.6f}" if isinstance(v, float) and math.isfinite(v) else v) for k, v in detail.items()},
            }
            pack = (validation_score, candidate, ref_pick, tdoa_corr, sigma, raw_plot, tdoa_rows, alignments, row)
            if best is None or validation_score > best[0]:
                best = pack
            validation_rows.append(row)
        except Exception as e:
            validation_rows.append({
                "rank": rank,
                "selected": "",
                "time_s": f"{candidate.time_s:.6f}",
                "sample": candidate.sample,
                "peak_time_s": f"{candidate.peak_time_s:.6f}",
                "snr": f"{candidate.snr:.3f}",
                "prominence": f"{candidate.prominence:.3f}",
                "width_s": f"{candidate.width_s:.6f}",
                "detector_score": f"{candidate.score:.3f}",
                "validation_score": "",
                "quality": candidate.quality,
                "error": str(e),
            })

    if best is None:
        write_tdoa_csv(outdir / "auto_clap_candidates.csv", validation_rows)
        raise SystemExit("Auto clap detection found candidates, but none could be validated across units")

    _, candidate, ref_pick, tdoa_corr, sigma, raw_plot, tdoa_rows, alignments, selected_row = best
    for row in validation_rows:
        if row.get("sample") == candidate.sample and row.get("time_s") == f"{candidate.time_s:.6f}":
            row["selected"] = "yes"
            break
    write_tdoa_csv(outdir / "auto_clap_candidates.csv", validation_rows)
    print(
        f"Auto clap pick: ref={args.ref} time={candidate.time_s:.6f}s "
        f"sample={candidate.sample} snr={candidate.snr:.2f} "
        f"prominence={candidate.prominence:.2f} validation_score={best[0]:.2f}"
    )
    print(f"Wrote auto-pick candidates: {outdir / 'auto_clap_candidates.csv'}")
    return ref_pick, tdoa_corr, sigma, raw_plot, tdoa_rows, {"auto_pick": True, "candidate": candidate, "validation": selected_row}


def main() -> int:
    p = argparse.ArgumentParser(description="Localize an ARU event from fetched clips, optional TXT calibration, and polished Folium map.")
    p.add_argument("event_dir")
    p.add_argument("--ref", default="five")
    p.add_argument("--mode", choices=["impulse", "birdcall"], default="impulse")
    p.add_argument("--event-ref-offset", default=None, help="Manual offset into ref-unit clip. For impulse mode, omit this to auto-detect a clap anywhere in the clip.")
    p.add_argument("--units", default="zero,one,four,five")
    p.add_argument("--locations", default=None, help="Optional station override TXT with [stations].")
    p.add_argument("--calibration-txt", default=None, help="TXT calibration from aru_calibrate_session.py")
    p.add_argument("--sound-speed", type=float, default=343.0)
    p.add_argument("--max-tau-s", type=float, default=0.10)
    p.add_argument("--gcc-phat-exponent", type=float, default=1.0, help="0.0 is GCC, 1.0 is GCC-PHAT, 0.25 is lightly whitened GCC.")
    p.add_argument("--highpass-hz", type=float, default=300.0)
    p.add_argument("--bird-bandpass", default="1000,9000")
    p.add_argument("--birdnet-detect", action="store_true", help="Use birdnet-team/birdnet on event clips to find birdcall candidate windows.")
    p.add_argument("--birdnet-threshold", type=float, default=0.25, help="Merged-candidate threshold. Prediction-window filtering uses --birdnet-window-other-confidence.")
    p.add_argument("--birdnet-min-units", type=int, default=0, help="Minimum units required for BirdNET call-list output. 0 means all operating units.")
    p.add_argument("--birdnet-window-primary-confidence", type=float, default=0.70, help="BirdNET call-list output requires at least one unit at or above this confidence.")
    p.add_argument("--birdnet-window-other-confidence", type=float, default=0.01, help="BirdNET call-list output requires other identifying units at or above this confidence.")
    p.add_argument("--birdnet-top-k", type=int, default=5, help="Top BirdNET labels retained per analysis window.")
    p.add_argument("--birdnet-overlap-s", type=float, default=2.8, help="BirdNET overlap duration. 2.8 gives 0.2 s hop for 3 s windows.")
    p.add_argument("--birdnet-bandpass-fmin", type=int, default=0, help="BirdNET preprocessing high-pass frequency. Default 0 disables high-pass filtering.")
    p.add_argument("--birdnet-bandpass-fmax", type=int, default=9000, help="BirdNET preprocessing bandpass high edge.")
    p.add_argument("--birdnet-species-query", default=None, help="Optional case-insensitive substring filter for BirdNET species_name.")
    p.add_argument("--birdnet-merge-gap-s", type=float, default=0.4, help="Merge BirdNET windows of the same species within this gap.")
    p.add_argument("--birdnet-max-candidates", type=int, default=8, help="Maximum merged BirdNET candidates to keep.")
    p.add_argument("--birdnet-batch-size", type=int, default=8)
    p.add_argument("--birdnet-workers", type=int, default=1)
    p.add_argument("--birdnet-device", default="CPU")
    p.add_argument("--birdnet-species-list", default=None, help="Optional custom BirdNET species list. If omitted, birdcall runs generate out-dir/species.txt from unit five location.")
    p.add_argument("--birdnet-species-unit", default="five", help="Unit whose location is used for generated BirdNET species.txt.")
    p.add_argument("--birdnet-geo-week", type=int, default=None, help="BirdNET 1..48 week override for generated species.txt. Default: infer from event clip date.")
    p.add_argument("--birdnet-geo-min-confidence", type=float, default=0.03, help="Minimum BirdNET geo prior confidence for generated species.txt.")
    p.add_argument("--no-birdnet-location-species-list", action="store_true", help="Disable automatic generated species.txt for BirdNET runs.")
    p.add_argument("--bird-align-pad-before-s", type=float, default=0.08)
    p.add_argument("--bird-align-pad-after-s", type=float, default=0.16)
    p.add_argument("--bird-refine-hint-half-s", type=float, default=0.35, help="When --event-ref-offset is supplied, refine BirdNET timing only within this half-window.")
    p.add_argument("--bird-physical-slack-m", type=float, default=0.5145, help="Extra one-sided padding around each calibrated physical TDOA bound, in meters.")
    p.add_argument("--bird-envelope-weight", type=float, default=0.25, help="Weight of envelope correlation in birdcall alignment score. Use 0 to disable envelope correlation.")
    p.add_argument("--no-geometric-peak-selection", action="store_true", help="Disable second-stage birdcall peak combo selection by localization residual.")
    p.add_argument("--bird-geometric-max-peak-rank", type=int, default=3, help="Maximum per-station GCC peak rank allowed in birdcall geometric combo selection.")
    p.add_argument("--impulse-search-half-s", type=float, default=0.5, help="Manual-pick search half-window when --event-ref-offset or --no-auto-clap is used.")
    p.add_argument("--no-auto-clap", action="store_true", help="Disable whole-clip clap auto-detection and use the old midpoint/manual-offset impulse picker.")
    p.add_argument("--auto-clap-top-k", type=int, default=8, help="Number of reference-unit clap candidates to validate across units.")
    p.add_argument("--auto-clap-min-snr", type=float, default=8.0, help="Minimum robust-z high-frequency envelope SNR for clap candidates.")
    p.add_argument("--auto-clap-min-prominence", type=float, default=6.0, help="Minimum robust-z peak prominence for clap candidates.")
    p.add_argument("--auto-clap-min-separation-s", type=float, default=0.25, help="Minimum spacing between reference clap candidates.")
    p.add_argument("--auto-clap-edge-guard-s", type=float, default=0.25, help="Ignore auto-pick candidates this close to clip edges.")
    p.add_argument("--auto-clap-max-width-s", type=float, default=0.20, help="Envelope peak width above which a candidate is penalized as too broad.")
    p.add_argument("--auto-clap-physical-ratio-slack", type=float, default=1.05, help="Physical sanity ratio allowed during auto-pick validation.")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--mc", type=int, default=1000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    event_dir = Path(args.event_dir)
    if args.out_dir:
        outdir = Path(args.out_dir)
    elif args.mode == "birdcall" and args.birdnet_detect:
        outdir = event_dir / "localization_output_codex"
    else:
        outdir = event_dir / "localization_output"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.bird_geometric_max_peak_rank < 1:
        raise SystemExit("--bird-geometric-max-peak-rank must be >= 1")
    units = [u.strip() for u in args.units.split(",") if u.strip()]
    unit_files = find_unit_files(event_dir, units)
    if args.ref not in unit_files:
        raise SystemExit(f"Reference unit {args.ref!r} missing from event dir")

    # Locations: calibration fit wins if present; otherwise event_dir location summaries / override file.
    timing_offsets, cal_latlon = load_calibration(Path(args.calibration_txt), args.ref) if args.calibration_txt else ({}, None)
    if cal_latlon:
        stations = {u: {"lat": ll[0], "lon": ll[1], "sigma_m": 1.0, "source": args.calibration_txt} for u, ll in cal_latlon.items() if u in unit_files}
    else:
        stations = load_station_locations(event_dir, units=units, override_txt=Path(args.locations) if args.locations else None)
    missing = [u for u in unit_files if u not in stations]
    if missing:
        raise SystemExit(f"Missing station locations for: {missing}")
    station_latlon = {u: (float(stations[u]["lat"]), float(stations[u]["lon"])) for u in stations if u in unit_files}
    pos_xy = stations_to_xy(stations, args.ref)
    ref_latlon = station_latlon[args.ref]

    clocks = fit_clock_maps(unit_files, units=units, draws=args.draws)
    fs, _, dur_s = audio_info(unit_files[args.ref].flac_path)
    if args.mode == "birdcall":
        lo, hi = [float(x) for x in args.bird_bandpass.split(",")]
        bandpass = (lo, hi)
        highpass = None
    else:
        bandpass = None
        highpass = args.highpass_hz

    pick_meta = {"auto_pick": False}
    use_auto_clap = args.mode == "impulse" and args.event_ref_offset is None and not args.no_auto_clap
    if use_auto_clap:
        ref_pick, tdoa_corr, sigma, raw_plot, tdoa_rows, pick_meta = _auto_pick_impulse_with_validation(
            args,
            units,
            unit_files,
            clocks,
            timing_offsets,
            pos_xy,
            highpass_hz=highpass or 0.0,
            bandpass_hz=bandpass,
            outdir=outdir,
        )
        alignments = {}
    elif args.mode == "birdcall" and args.birdnet_detect:
        ref_pick, tdoa_corr, sigma, raw_plot, tdoa_rows, alignments, pick_meta = _birdnet_pick_and_align(
            args,
            units,
            unit_files,
            clocks,
            timing_offsets,
            bandpass,
            pos_xy,
            ref_latlon,
            outdir,
        )
    else:
        offset_s = parse_offset_seconds(args.event_ref_offset) if args.event_ref_offset else 0.5 * dur_s
        ref_pick = pick_impulse_in_unit(
            args.ref,
            unit_files[args.ref],
            clocks[args.ref],
            offset_s,
            search_half_s=args.impulse_search_half_s,
            highpass_hz=highpass or 0.0,
            bandpass_hz=bandpass,
        )
        tdoa_corr, sigma, raw_plot, tdoa_rows, _ = _align_all_units_for_pick(
            ref_pick,
            units,
            args.ref,
            unit_files,
            clocks,
            timing_offsets,
            args.max_tau_s,
            highpass or 0.0,
            bandpass,
            args.gcc_phat_exponent,
        )
        alignments = {}
    write_tdoa_csv(outdir / "event_tdoa.csv", tdoa_rows)

    # Audio diagnostics should show alignment in the recorded file timeline.
    # Calibration offsets model upstream buffer delays and are used for localization,
    # but should not be applied when visually shifting waveforms/spectrograms.
    plot_alignment_diagnostics(outdir / "diagnostics", unit_files, clocks, ref_pick, raw_plot, ref_unit=args.ref, highpass_hz=highpass or 0.0, bandpass_hz=bandpass)
    if args.mode == "birdcall" and args.birdnet_detect and alignments:
        _plot_birdcall_correlation_diagnostics(
            outdir,
            alignments,
            pos_xy,
            args.ref,
            args.sound_speed,
            timing_offsets,
            physical_slack_s=float(args.bird_physical_slack_m) / max(float(args.sound_speed), 1e-9),
        )
        _plot_birdcall_envelope_diagnostics(
            outdir,
            unit_files,
            clocks,
            ref_pick,
            alignments,
            args.ref,
            bandpass,
            args.bird_align_pad_before_s,
            args.bird_align_pad_after_s,
        )
        _plot_birdcall_residual_spectrograms(
            outdir,
            unit_files,
            clocks,
            ref_pick,
            raw_plot,
            args.ref,
            bandpass,
        )

    measured = {u: t for u, t in tdoa_corr.items() if u != args.ref}
    if len(measured) < 2:
        raise SystemExit("Need at least two non-reference TDOAs for 2D localization")
    source_xy, residuals, cost = localize_least_squares(pos_xy, measured, sigma_s=sigma, ref_unit=args.ref, c=args.sound_speed)
    loo = leave_one_out_localizations(pos_xy, measured, sigma, args.ref, args.sound_speed)
    samples = mc_location_samples(pos_xy, measured, sigma, args.ref, args.sound_speed, n=args.mc, seed=0) if args.mc > 0 else np.zeros((0,2))
    ell95 = covariance_ellipse(samples, chi2_level=5.991) if samples.size else None
    map_path = outdir / "localization_map.html"
    write_event_map(str(map_path), station_latlon, pos_xy, args.ref, source_xy, measured, args.sound_speed, samples_xy=samples, ellipse95_xy=ell95, loo_xy=loo)
    source_ll = xy_to_latlon(float(source_xy[0]), float(source_xy[1]), ref_latlon)

    # Text summary, not JSON.
    with (outdir / "localization_summary.txt").open("w") as f:
        f.write("# ARU localization summary TXT v1\n")
        f.write(f"event_dir = {event_dir}\n")
        f.write(f"ref_unit = {args.ref}\n")
        f.write(f"mode = {args.mode}\n")
        f.write(f"calibration_txt = {args.calibration_txt or ''}\n")
        f.write(f"gcc_phat_exponent = {args.gcc_phat_exponent:.6f}\n")
        f.write(f"event_pick_ref_unit = {args.ref}\n")
        f.write(f"event_pick_sample = {ref_pick.sample}\n")
        f.write(f"event_pick_time_s = {ref_pick.time_s:.9f}\n")
        f.write(f"event_pick_abs_time = {ref_pick.abs_time.isoformat()}\n")
        f.write(f"event_pick_snr = {ref_pick.snr:.6f}\n")
        f.write(f"event_pick_quality = {ref_pick.quality}\n")
        if pick_meta.get("auto_pick"):
            method = "auto_clap_validated"
        elif pick_meta.get("birdnet_pick"):
            method = "birdnet_refined"
        else:
            method = "manual_or_midpoint_window"
        f.write(f"event_pick_method = {method}\n")
        if pick_meta.get("auto_pick"):
            cand = pick_meta.get("candidate")
            f.write(f"event_pick_peak_time_s = {cand.peak_time_s:.9f}\n")
            f.write(f"event_pick_prominence = {cand.prominence:.6f}\n")
            f.write(f"event_pick_width_s = {cand.width_s:.9f}\n")
            f.write(f"event_pick_detector_score = {cand.score:.6f}\n")
            f.write(f"auto_clap_candidates_csv = {outdir / 'auto_clap_candidates.csv'}\n")
        if pick_meta.get("birdnet_pick"):
            cand = pick_meta.get("candidate")
            f.write(f"birdnet_species_name = {cand.species_name}\n")
            f.write(f"birdnet_confidence = {cand.confidence:.9f}\n")
            f.write(f"birdnet_window_start_s = {cand.start_s:.9f}\n")
            f.write(f"birdnet_window_end_s = {cand.end_s:.9f}\n")
            f.write(f"birdnet_window_duration_s = {float(cand.end_s) - float(cand.start_s):.9f}\n")
            f.write(f"birdnet_detector_score = {cand.score:.9f}\n")
            selected_call = pick_meta.get("selected_call") or {}
            f.write(f"birdnet_selected_call_id = {selected_call.get('call_id', '')}\n")
            f.write(f"birdnet_selected_units = {selected_call.get('units', '')}\n")
            f.write(f"birdnet_selected_per_unit_confidence = {selected_call.get('per_unit_confidence', '')}\n")
            dist = selected_call.get("distance_to_hint_s", "")
            if isinstance(dist, float) and math.isfinite(dist):
                f.write(f"birdnet_selected_distance_to_hint_s = {dist:.9f}\n")
            else:
                f.write("birdnet_selected_distance_to_hint_s = \n")
            f.write(f"birdcall_alignment_window_start_s = {float(pick_meta.get('effective_align_start_s', float('nan'))):.9f}\n")
            f.write(f"birdcall_alignment_window_end_s = {float(pick_meta.get('effective_align_end_s', float('nan'))):.9f}\n")
            f.write(f"birdcall_alignment_window_duration_s = {float(pick_meta.get('effective_align_duration_s', float('nan'))):.9f}\n")
            f.write(f"birdnet_species_list = {pick_meta.get('species_list')}\n")
            f.write(f"birdnet_candidates_csv = {pick_meta.get('candidates_csv')}\n")
            f.write(f"birdnet_predictions_csv = {pick_meta.get('predictions_csv')}\n")
            f.write(f"birdnet_calls_txt = {pick_meta.get('calls_txt')}\n")
            f.write(f"birdnet_calls_csv = {pick_meta.get('calls_csv')}\n")
            f.write(f"geometric_peak_combos_csv = {pick_meta.get('geometric_peak_combos_csv')}\n")
        f.write(f"source_lat = {source_ll[0]:.9f}\n")
        f.write(f"source_lon = {source_ll[1]:.9f}\n")
        f.write(f"source_x_m = {source_xy[0]:.4f}\n")
        f.write(f"source_y_m = {source_xy[1]:.4f}\n")
        f.write(f"weighted_cost = {cost:.6f}\n")
        f.write("\n[tdoa_residuals]\nunit\tresidual_s\tsigma_s\n")
        for u, r in residuals.items():
            f.write(f"{u}\t{r:+.9f}\t{sigma.get(u, float('nan')):.9f}\n")
        f.write("\n[leave_one_out]\ndropped_unit\tlat\tlon\n")
        for drop, xy in loo.items():
            ll = xy_to_latlon(float(xy[0]), float(xy[1]), ref_latlon)
            f.write(f"{drop}\t{ll[0]:.9f}\t{ll[1]:.9f}\n")
        f.write("\n[physical_sanity]\nunit\tbaseline_m\tabs_range_difference_m\tratio\n")
        for u, row in physical_sanity(pos_xy, measured, args.ref, args.sound_speed).items():
            f.write(f"{u}\t{row['baseline_m']:.4f}\t{row['abs_range_difference_m']:.4f}\t{row['ratio']:.4f}\n")

    print(f"Source lat/lon: {source_ll[0]:.9f}, {source_ll[1]:.9f}")
    print(f"Wrote map: {map_path}")
    print(f"Wrote TDOAs: {outdir / 'event_tdoa.csv'}")
    print(f"Wrote summary: {outdir / 'localization_summary.txt'}")
    if loo:
        print("Leave-one-out localizations written to map and summary.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
