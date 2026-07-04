#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from aru_io import (
    find_unit_files, load_station_locations, stations_to_xy, xy_to_latlon, latlon_to_xy_m,
    fit_clock_maps, audio_info, parse_offset_seconds, read_kv_txt
)
from align import pick_impulse_in_unit, fine_align_impulse
from tdoa import build_tdoa_from_alignment, expected_tdoa_s
from calibration import (
    fit_geometry_and_timing,
    load_calibration_txt,
    project_relative_text,
    session_calibration_output_path,
    update_tables_with_event,
    write_calibration_txt,
)
from maps import write_station_fit_map
from viz import plot_alignment_diagnostics


def event_id_from_dir(event_dir: Path) -> str:
    return event_dir.name


def source_from_args(args, stations, ref_unit, ref_latlon, pos_xy):
    if args.source_lat is not None and args.source_lon is not None:
        sx, sy = latlon_to_xy_m(float(args.source_lat), float(args.source_lon), ref_latlon)
        return {
            "source_type": "coordinate",
            "source_unit": "",
            "source_lat": f"{float(args.source_lat):.9f}",
            "source_lon": f"{float(args.source_lon):.9f}",
            "source_x_m": f"{sx:.6f}",
            "source_y_m": f"{sy:.6f}",
        }, np.asarray([sx, sy], dtype=float)
    su = args.source_unit or ref_unit
    if su not in stations:
        raise SystemExit(f"source unit {su!r} has no location")
    return {
        "source_type": "unit",
        "source_unit": su,
        "source_lat": f"{float(stations[su]['lat']):.9f}",
        "source_lon": f"{float(stations[su]['lon']):.9f}",
        "source_x_m": f"{float(pos_xy[su][0]):.6f}",
        "source_y_m": f"{float(pos_xy[su][1]):.6f}",
    }, np.asarray(pos_xy[su], dtype=float)


def main() -> int:
    p = argparse.ArgumentParser(description="Create/update an ARU TXT calibration file from a known clap event.")
    p.add_argument("event_dir", help="Fetched calibration event directory from rotate_fetch_clips_txt.py")
    p.add_argument(
        "--calibration-txt",
        default=None,
        help="Calibration TXT to create/update. Default: data/calibrations/station_offset_calibrations/session_calibration.txt.",
    )
    p.add_argument("--ref", default="five")
    p.add_argument("--source-unit", choices=["zero", "one", "four", "five"], default="five", help="Unit where the clap occurred. Default: five.")
    p.add_argument("--source-lat", type=float, default=None)
    p.add_argument("--source-lon", type=float, default=None)
    p.add_argument("--event-ref-offset", default=None, help="Offset into ref-unit clip, e.g. 5.2 or 0:05.2. Default: middle of clip.")
    p.add_argument("--units", default="zero,one,four,five")
    p.add_argument("--locations", default=None, help="Optional station override TXT with [stations] unit lat lon rows.")
    p.add_argument("--sound-speed", type=float, default=343.0)
    p.add_argument("--position-prior-sigma-m", type=float, default=2.0)
    p.add_argument("--timing-prior-sigma-s", type=float, default=0.003)
    p.add_argument("--max-tau-s", type=float, default=0.10)
    p.add_argument("--highpass-hz", type=float, default=300.0)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    event_dir = Path(args.event_dir)
    outdir = Path(args.out_dir) if args.out_dir else event_dir / "calibration_output"
    outdir.mkdir(parents=True, exist_ok=True)
    units = [u.strip() for u in args.units.split(",") if u.strip()]
    unit_files = find_unit_files(event_dir, units)
    if args.ref not in unit_files:
        raise SystemExit(f"Reference unit {args.ref!r} missing from event dir")
    stations = load_station_locations(event_dir, units=units, override_txt=Path(args.locations) if args.locations else None, default_sigma_m=args.position_prior_sigma_m)
    missing = [u for u in unit_files if u not in stations]
    if missing:
        raise SystemExit(f"Missing station locations for: {missing}")
    ref_latlon = (float(stations[args.ref]["lat"]), float(stations[args.ref]["lon"]))
    input_latlon = {u: (float(stations[u]["lat"]), float(stations[u]["lon"])) for u in stations}
    pos_xy = stations_to_xy(stations, args.ref)
    source_meta, source_xy = source_from_args(args, stations, args.ref, ref_latlon, pos_xy)

    clocks = fit_clock_maps(unit_files, units=units, draws=args.draws)
    fs, _, dur_s = audio_info(unit_files[args.ref].flac_path)
    offset_s = parse_offset_seconds(args.event_ref_offset) if args.event_ref_offset else 0.5 * dur_s
    ref_pick = pick_impulse_in_unit(args.ref, unit_files[args.ref], clocks[args.ref], offset_s, highpass_hz=args.highpass_hz)

    tdoa_rows: List[Dict[str, Any]] = []
    tdoa_for_plot = {args.ref: 0.0}
    for u in units:
        if u == args.ref or u not in unit_files:
            continue
        aln = fine_align_impulse(u, args.ref, unit_files, clocks, ref_pick, max_tau_s=args.max_tau_s, highpass_hz=args.highpass_hz)
        est = build_tdoa_from_alignment(aln, timing_offsets_s=None)
        exp = expected_tdoa_s(pos_xy, source_xy, u, args.ref, args.sound_speed)
        tdoa_rows.append({
            "event_id": event_id_from_dir(event_dir),
            "unit": u,
            "ref_unit": args.ref,
            "measured_tdoa_s": f"{est.raw_s:+.9f}",
            "sigma_s": f"{est.sigma_s:.9f}",
            "expected_from_input_geometry_s": f"{exp:+.9f}",
            "raw_minus_expected_s": f"{(est.raw_s-exp):+.9f}",
            "gcc_tau_s": f"{est.tau_s:+.9f}",
            "coarse_dt_s": f"{est.coarse_dt_s:+.9f}",
            "quality_peak": f"{est.quality_peak:.6f}",
            "quality_secondary_ratio": f"{est.quality_secondary_ratio:.6f}",
        })
        tdoa_for_plot[u] = est.raw_s

    plot_alignment_diagnostics(outdir / "diagnostics", unit_files, clocks, ref_pick, tdoa_for_plot, ref_unit=args.ref, highpass_hz=args.highpass_hz)

    cal_path = Path(args.calibration_txt).expanduser() if args.calibration_txt else session_calibration_output_path()
    existing_kv, existing_tables = load_calibration_txt(cal_path)
    event_meta = {
        "event_id": event_id_from_dir(event_dir),
        "event_dir": project_relative_text(event_dir),
        "ref_unit": args.ref,
        **source_meta,
    }
    tables = update_tables_with_event(existing_tables, event_meta, tdoa_rows)
    fit_pos_xy, timing_offsets, residual_rows = fit_geometry_and_timing(
        pos_xy, input_latlon, tables, args.ref, ref_latlon,
        sound_speed_m_s=args.sound_speed,
        position_prior_sigma_m=args.position_prior_sigma_m,
        timing_prior_sigma_s=args.timing_prior_sigma_s,
    )
    write_calibration_txt(cal_path, args.ref, args.sound_speed, input_latlon, fit_pos_xy, timing_offsets, ref_latlon, tables, residual_rows, args.position_prior_sigma_m, args.timing_prior_sigma_s)

    fit_latlon = {u: xy_to_latlon(float(fit_pos_xy[u][0]), float(fit_pos_xy[u][1]), ref_latlon) for u in fit_pos_xy}
    source_ll = {event_id_from_dir(event_dir): (float(source_meta["source_lat"]), float(source_meta["source_lon"]))}
    map_path = outdir / "station_fit_map.html"
    write_station_fit_map(str(map_path), input_latlon, fit_latlon, source_ll)

    print(f"Wrote/updated calibration: {cal_path}")
    print(f"Wrote station-fit map: {map_path}")
    print(f"Wrote diagnostics: {outdir / 'diagnostics'}")
    print("Timing offsets relative to ref:")
    for u in sorted(timing_offsets):
        print(f"  {u:>4}: {timing_offsets[u]*1e6:+.1f} us")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
