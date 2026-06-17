#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import math
import csv
import numpy as np

from aru_io import (
    find_unit_files, load_station_locations, stations_to_xy, xy_to_latlon, parse_offset_seconds,
    fit_clock_maps, audio_info, read_sectioned_tables
)
from align import pick_impulse_in_unit, fine_align_impulse
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


def main() -> int:
    p = argparse.ArgumentParser(description="Localize an ARU event from fetched clips, optional TXT calibration, and polished Folium map.")
    p.add_argument("event_dir")
    p.add_argument("--ref", default="five")
    p.add_argument("--mode", choices=["impulse", "birdcall"], default="impulse")
    p.add_argument("--event-ref-offset", default=None, help="Offset into ref-unit clip; default middle of clip.")
    p.add_argument("--units", default="zero,one,four,five")
    p.add_argument("--locations", default=None, help="Optional station override TXT with [stations].")
    p.add_argument("--calibration-txt", default=None, help="TXT calibration from aru_calibrate_session.py")
    p.add_argument("--sound-speed", type=float, default=343.0)
    p.add_argument("--max-tau-s", type=float, default=0.10)
    p.add_argument("--highpass-hz", type=float, default=300.0)
    p.add_argument("--bird-bandpass", default="1000,9000")
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
    offset_s = parse_offset_seconds(args.event_ref_offset) if args.event_ref_offset else 0.5 * dur_s
    if args.mode == "birdcall":
        lo, hi = [float(x) for x in args.bird_bandpass.split(",")]
        bandpass = (lo, hi)
        highpass = None
    else:
        bandpass = None
        highpass = args.highpass_hz
    ref_pick = pick_impulse_in_unit(args.ref, unit_files[args.ref], clocks[args.ref], offset_s, highpass_hz=highpass or 0.0, bandpass_hz=bandpass)

    tdoa_rows = []
    tdoa_corr = {args.ref: 0.0}
    sigma = {}
    raw_plot = {args.ref: 0.0}
    for u in units:
        if u == args.ref or u not in unit_files:
            continue
        aln = fine_align_impulse(u, args.ref, unit_files, clocks, ref_pick, max_tau_s=args.max_tau_s, highpass_hz=highpass or 0.0, bandpass_hz=bandpass)
        est = build_tdoa_from_alignment(aln, timing_offsets_s=timing_offsets)
        tdoa_corr[u] = est.corrected_s
        raw_plot[u] = est.raw_s
        sigma[u] = est.sigma_s
        tdoa_rows.append({
            "unit": u,
            "ref_unit": args.ref,
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
