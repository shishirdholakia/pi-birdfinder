#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from align import find_impulse_candidates_in_unit, fine_align_impulse, pick_impulse_in_unit
from aru_io import (
    audio_info,
    find_unit_files,
    fit_clock_maps,
    latlon_to_xy_m,
    load_station_locations,
    parse_offset_seconds,
    stations_to_xy,
)
from tdoa import build_tdoa_from_alignment
from viz import plot_alignment_diagnostics


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _expected_tdoa_s(pos_xy, source_xy: np.ndarray, unit: str, ref_unit: str, sound_speed: float) -> float:
    d_unit = float(np.linalg.norm(np.asarray(source_xy, dtype=float) - np.asarray(pos_xy[unit], dtype=float)))
    d_ref = float(np.linalg.norm(np.asarray(source_xy, dtype=float) - np.asarray(pos_xy[ref_unit], dtype=float)))
    return (d_unit - d_ref) / float(sound_speed)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fit per-station buffer lag offsets from one ground-truth impulse event."
    )
    p.add_argument("event_dir")
    p.add_argument("--ref", default="five")
    p.add_argument("--event-ref-offset", default=None, help="Optional approximate impulse offset in the ref-unit clip. If omitted, auto-detect in the ref unit.")
    p.add_argument("--source-lat", type=float, required=True)
    p.add_argument("--source-lon", type=float, required=True)
    p.add_argument("--units", default="zero,one,four,five")
    p.add_argument("--locations", default=None)
    p.add_argument("--sound-speed", type=float, default=343.0)
    p.add_argument("--max-tau-s", type=float, default=0.10)
    p.add_argument("--highpass-hz", type=float, default=300.0)
    p.add_argument("--bandpass", default=None, help="Optional impulse bandpass as low,high Hz.")
    p.add_argument("--gcc-phat-exponent", type=float, default=1.0)
    p.add_argument("--impulse-search-half-s", type=float, default=0.5)
    p.add_argument("--auto-impulse-top-k", type=int, default=8)
    p.add_argument("--auto-impulse-min-snr", type=float, default=8.0)
    p.add_argument("--auto-impulse-min-prominence", type=float, default=6.0)
    p.add_argument("--auto-impulse-min-separation-s", type=float, default=0.25)
    p.add_argument("--auto-impulse-edge-guard-s", type=float, default=0.25)
    p.add_argument("--auto-impulse-max-width-s", type=float, default=0.20)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    event_dir = Path(args.event_dir)
    outdir = Path(args.out_dir) if args.out_dir else event_dir / "impulse_offset_fit"
    outdir.mkdir(parents=True, exist_ok=True)

    units = [u.strip() for u in args.units.split(",") if u.strip()]
    unit_files = find_unit_files(event_dir, units)
    if args.ref not in unit_files:
        raise SystemExit(f"Reference unit {args.ref!r} missing from event dir")

    stations = load_station_locations(event_dir, units=units, override_txt=Path(args.locations) if args.locations else None)
    missing = [u for u in unit_files if u not in stations]
    if missing:
        raise SystemExit(f"Missing station locations for: {missing}")
    pos_xy = stations_to_xy(stations, args.ref)
    ref_latlon = (float(stations[args.ref]["lat"]), float(stations[args.ref]["lon"]))
    source_xy = np.asarray(latlon_to_xy_m(float(args.source_lat), float(args.source_lon), ref_latlon), dtype=float)

    bandpass = None
    if args.bandpass:
        lo, hi = [float(x) for x in str(args.bandpass).split(",")]
        bandpass = (lo, hi)

    clocks = fit_clock_maps(unit_files, units=units, draws=args.draws)
    fs, _, dur_s = audio_info(unit_files[args.ref].flac_path)
    if args.event_ref_offset:
        offset_s = parse_offset_seconds(args.event_ref_offset)
        if not (0.0 <= offset_s <= dur_s):
            raise SystemExit(f"--event-ref-offset {offset_s} is outside the ref clip duration {dur_s:.3f}s")
        ref_pick = pick_impulse_in_unit(
            args.ref,
            unit_files[args.ref],
            clocks[args.ref],
            offset_s,
            search_half_s=args.impulse_search_half_s,
            highpass_hz=args.highpass_hz,
            bandpass_hz=bandpass,
        )
        pick_method = "manual_window"
    else:
        candidates = list(find_impulse_candidates_in_unit(
            args.ref,
            unit_files[args.ref],
            clocks[args.ref],
            highpass_hz=args.highpass_hz,
            bandpass_hz=bandpass,
            min_snr=args.auto_impulse_min_snr,
            min_prominence=args.auto_impulse_min_prominence,
            min_separation_s=args.auto_impulse_min_separation_s,
            top_k=args.auto_impulse_top_k,
            edge_guard_s=args.auto_impulse_edge_guard_s,
            max_width_s=args.auto_impulse_max_width_s,
        ))
        if not candidates:
            candidates = list(find_impulse_candidates_in_unit(
                args.ref,
                unit_files[args.ref],
                clocks[args.ref],
                highpass_hz=args.highpass_hz,
                bandpass_hz=bandpass,
                min_snr=0.0,
                min_prominence=0.0,
                min_separation_s=args.auto_impulse_min_separation_s,
                top_k=1,
                edge_guard_s=0.0,
                max_width_s=args.auto_impulse_max_width_s,
            ))
        if not candidates:
            raise SystemExit("Auto impulse detection found no candidates in the reference unit")
        ref_candidate = candidates[0]
        ref_pick = ref_candidate.as_event_pick()
        pick_method = "auto_ref_unit_strongest"
        _write_csv(outdir / "auto_impulse_candidates.csv", [
            {
                "rank": rank,
                "selected": "yes" if cand == ref_candidate else "",
                "unit": cand.unit,
                "time_s": f"{cand.time_s:.9f}",
                "sample": cand.sample,
                "peak_time_s": f"{cand.peak_time_s:.9f}",
                "peak_sample": cand.peak_sample,
                "snr": f"{cand.snr:.6f}",
                "prominence": f"{cand.prominence:.6f}",
                "score": f"{cand.score:.6f}",
                "width_s": f"{cand.width_s:.9f}",
                "quality": cand.quality,
            }
            for rank, cand in enumerate(candidates, start=1)
        ])

    rows = []
    raw_plot = {args.ref: 0.0}
    expected_plot = {args.ref: 0.0}
    offset_plot = {args.ref: 0.0}
    rows.append({
        "unit": args.ref,
        "ref_unit": args.ref,
        "expected_tdoa_s": "+0.000000000",
        "raw_tdoa_s": "+0.000000000",
        "fitted_offset_s": "+0.000000000",
        "fitted_offset_ms": "+0.000000",
        "corrected_tdoa_s": "+0.000000000",
        "postfit_residual_s": "+0.000000000",
        "gcc_tau_s": "+0.000000000",
        "coarse_dt_s": "+0.000000000",
        "sigma_s": "",
        "quality_peak": "",
        "quality_secondary_ratio": "",
        "quality": "reference",
    })

    for unit in units:
        if unit == args.ref or unit not in unit_files:
            continue
        if unit not in pos_xy:
            continue
        aln = fine_align_impulse(
            unit,
            args.ref,
            unit_files,
            clocks,
            ref_pick,
            max_tau_s=args.max_tau_s,
            highpass_hz=args.highpass_hz,
            bandpass_hz=bandpass,
            gcc_phat_exponent=args.gcc_phat_exponent,
        )
        est = build_tdoa_from_alignment(aln, timing_offsets_s={})
        expected = _expected_tdoa_s(pos_xy, source_xy, unit, args.ref, args.sound_speed)
        fitted_offset = float(est.raw_s) - expected
        corrected = float(est.raw_s) - fitted_offset
        residual = corrected - expected
        raw_plot[unit] = float(est.raw_s)
        expected_plot[unit] = expected
        offset_plot[unit] = fitted_offset
        rows.append({
            "unit": unit,
            "ref_unit": args.ref,
            "expected_tdoa_s": f"{expected:+.9f}",
            "raw_tdoa_s": f"{est.raw_s:+.9f}",
            "fitted_offset_s": f"{fitted_offset:+.9f}",
            "fitted_offset_ms": f"{1000.0 * fitted_offset:+.6f}",
            "corrected_tdoa_s": f"{corrected:+.9f}",
            "postfit_residual_s": f"{residual:+.9f}",
            "gcc_tau_s": f"{est.tau_s:+.9f}",
            "coarse_dt_s": f"{est.coarse_dt_s:+.9f}",
            "sigma_s": f"{est.sigma_s:.9f}",
            "quality_peak": f"{est.quality_peak:.6f}",
            "quality_secondary_ratio": f"{est.quality_secondary_ratio:.6f}",
            "quality": "ok" if math.isfinite(fitted_offset) else "failed",
        })

    _write_csv(outdir / "impulse_buffer_offsets.csv", rows)
    plot_alignment_diagnostics(
        outdir / "diagnostics_raw",
        unit_files,
        clocks,
        ref_pick,
        raw_plot,
        ref_unit=args.ref,
        highpass_hz=args.highpass_hz,
        bandpass_hz=bandpass,
    )
    plot_alignment_diagnostics(
        outdir / "diagnostics_expected_physical",
        unit_files,
        clocks,
        ref_pick,
        expected_plot,
        ref_unit=args.ref,
        highpass_hz=args.highpass_hz,
        bandpass_hz=bandpass,
    )

    with (outdir / "impulse_buffer_offsets_summary.txt").open("w") as f:
        f.write("# ARU impulse buffer offset fit\n")
        f.write(f"event_dir = {event_dir}\n")
        f.write(f"ref_unit = {args.ref}\n")
        f.write(f"source_lat = {args.source_lat:.9f}\n")
        f.write(f"source_lon = {args.source_lon:.9f}\n")
        f.write(f"source_x_m = {source_xy[0]:.6f}\n")
        f.write(f"source_y_m = {source_xy[1]:.6f}\n")
        f.write(f"event_pick_time_s = {ref_pick.time_s:.9f}\n")
        f.write(f"event_pick_sample = {ref_pick.sample}\n")
        f.write(f"event_pick_abs_time = {ref_pick.abs_time.isoformat()}\n")
        f.write(f"event_pick_snr = {ref_pick.snr:.6f}\n")
        f.write(f"event_pick_method = {pick_method}\n")
        f.write(f"gcc_phat_exponent = {args.gcc_phat_exponent:.6f}\n")
        f.write("offset_sign = corrected_tdoa = raw_tdoa - fitted_offset_s\n")
        f.write(f"offsets_csv = {outdir / 'impulse_buffer_offsets.csv'}\n")
        if not args.event_ref_offset:
            f.write(f"auto_impulse_candidates_csv = {outdir / 'auto_impulse_candidates.csv'}\n")

    print(f"Impulse pick: ref={args.ref} time={ref_pick.time_s:.6f}s sample={ref_pick.sample} snr={ref_pick.snr:.2f}")
    print(f"Wrote offsets: {outdir / 'impulse_buffer_offsets.csv'}")
    print(f"Wrote summary: {outdir / 'impulse_buffer_offsets_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
