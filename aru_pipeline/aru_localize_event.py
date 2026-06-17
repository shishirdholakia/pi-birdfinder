#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple
import math
import csv
import numpy as np

from aru_io import (
    find_unit_files, load_station_locations, stations_to_xy, xy_to_latlon, parse_offset_seconds,
    fit_clock_maps, audio_info, read_sectioned_tables
)
from align import EventPick, find_impulse_candidates_in_unit, pick_impulse_in_unit, fine_align_impulse
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
    p.add_argument("--highpass-hz", type=float, default=300.0)
    p.add_argument("--bird-bandpass", default="1000,9000")
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
    outdir = Path(args.out_dir) if args.out_dir else event_dir / "localization_output"
    outdir.mkdir(parents=True, exist_ok=True)
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
        )
    write_tdoa_csv(outdir / "event_tdoa.csv", tdoa_rows)

    plot_alignment_diagnostics(outdir / "diagnostics", unit_files, clocks, ref_pick, tdoa_corr, ref_unit=args.ref, highpass_hz=highpass or 0.0, bandpass_hz=bandpass)

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
        f.write(f"event_pick_ref_unit = {args.ref}\n")
        f.write(f"event_pick_sample = {ref_pick.sample}\n")
        f.write(f"event_pick_time_s = {ref_pick.time_s:.9f}\n")
        f.write(f"event_pick_abs_time = {ref_pick.abs_time.isoformat()}\n")
        f.write(f"event_pick_snr = {ref_pick.snr:.6f}\n")
        f.write(f"event_pick_quality = {ref_pick.quality}\n")
        f.write(f"event_pick_method = {'auto_clap_validated' if pick_meta.get('auto_pick') else 'manual_or_midpoint_window'}\n")
        if pick_meta.get("auto_pick"):
            cand = pick_meta.get("candidate")
            f.write(f"event_pick_peak_time_s = {cand.peak_time_s:.9f}\n")
            f.write(f"event_pick_prominence = {cand.prominence:.6f}\n")
            f.write(f"event_pick_width_s = {cand.width_s:.9f}\n")
            f.write(f"event_pick_detector_score = {cand.score:.6f}\n")
            f.write(f"auto_clap_candidates_csv = {outdir / 'auto_clap_candidates.csv'}\n")
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
