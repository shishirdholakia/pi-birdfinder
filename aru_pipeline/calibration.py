#!/usr/bin/env python3
"""Calibration-file read/write and geometry/timing fit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
from scipy import optimize

from aru_io import read_sectioned_tables, write_sectioned_tables, xy_to_latlon
from tdoa import expected_tdoa_s


@dataclass
class CalibrationEvent:
    event_id: str
    source_type: str          # "unit" or "coordinate"
    source_unit: str
    source_lat: float
    source_lon: float
    tdoa_rows: List[Dict[str, Any]]


def load_calibration_txt(path: Path) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, str]]]]:
    return read_sectioned_tables(path)


def extract_timing_offsets_from_tables(tables: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
    out = {}
    for row in tables.get("stations", []):
        unit = row.get("unit", "")
        if unit:
            try: out[unit] = float(row.get("timing_offset_s", "0") or 0.0)
            except Exception: out[unit] = 0.0
    return out


def update_tables_with_event(existing_tables: Dict[str, List[Dict[str, str]]], event_meta: Dict[str, Any], tdoa_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    tables: Dict[str, List[Dict[str, Any]]] = {k: [dict(r) for r in v] for k, v in existing_tables.items()}
    eid = str(event_meta["event_id"])
    tables["events"] = [r for r in tables.get("events", []) if str(r.get("event_id")) != eid]
    tables["tdoa"] = [r for r in tables.get("tdoa", []) if str(r.get("event_id")) != eid]
    tables.setdefault("events", []).append(event_meta)
    tables.setdefault("tdoa", []).extend(tdoa_rows)
    return tables


def fit_geometry_and_timing(
    initial_pos_xy: Dict[str, np.ndarray],
    station_latlon_input: Dict[str, Tuple[float, float]],
    tables: Dict[str, List[Dict[str, Any]]],
    ref_unit: str,
    ref_latlon: Tuple[float, float],
    sound_speed_m_s: float = 343.0,
    position_prior_sigma_m: float = 2.0,
    timing_prior_sigma_s: float = 0.003,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[Dict[str, Any]]]:
    units = sorted(initial_pos_xy.keys())
    nonref = [u for u in units if u != ref_unit]
    unit_index = {u: i for i, u in enumerate(units)}
    off_index = {u: i for i, u in enumerate(nonref)}
    npos = 2 * len(units)
    noff = len(nonref)

    # Event metadata lookup.
    events = {str(r.get("event_id")): r for r in tables.get("events", [])}
    trows = [dict(r) for r in tables.get("tdoa", [])]

    def unpack(v):
        pos = {}
        for u in units:
            i = unit_index[u]
            dx, dy = v[2*i], v[2*i+1]
            pos[u] = initial_pos_xy[u] + np.asarray([dx, dy], dtype=float)
        offsets = {ref_unit: 0.0}
        for u in nonref:
            offsets[u] = float(v[npos + off_index[u]])
        return pos, offsets

    def source_xy_for_event(event_row, pos):
        stype = str(event_row.get("source_type", "unit"))
        if stype == "unit":
            su = str(event_row.get("source_unit", ref_unit))
            return pos[su]
        sx = float(event_row.get("source_x_m", "nan"))
        sy = float(event_row.get("source_y_m", "nan"))
        if not math.isfinite(sx) or not math.isfinite(sy):
            # If only lat/lon available, approximate by nearest stored fixed source x/y being absent.
            raise ValueError(f"Coordinate event {event_row.get('event_id')} lacks source_x_m/source_y_m")
        return np.asarray([sx, sy], dtype=float)

    def residuals(v):
        pos, offsets = unpack(v)
        rr = []
        # TDOA residuals.
        for row in trows:
            u = str(row.get("unit"))
            if u == ref_unit or u not in pos:
                continue
            eid = str(row.get("event_id"))
            if eid not in events:
                continue
            obs = float(row.get("measured_tdoa_s", row.get("corrected_tdoa_s", 0.0)))
            sig = max(float(row.get("sigma_s", 0.0005) or 0.0005), 1e-6)
            src = source_xy_for_event(events[eid], pos)
            pred = expected_tdoa_s(pos, src, u, ref_unit, sound_speed_m_s) + offsets.get(u, 0.0) - offsets.get(ref_unit, 0.0)
            rr.append((pred - obs) / sig)
        # Priors: station displacements and timing offsets.
        for u in units:
            i = unit_index[u]
            rr.extend([v[2*i] / position_prior_sigma_m, v[2*i+1] / position_prior_sigma_m])
        for u in nonref:
            rr.append(v[npos + off_index[u]] / timing_prior_sigma_s)
        return np.asarray(rr, dtype=float)

    x0 = np.zeros(npos + noff, dtype=float)
    # Initialize offsets from single-event residuals if possible.
    try:
        for u in nonref:
            vals = []
            for row in trows:
                if str(row.get("unit")) != u: continue
                eid = str(row.get("event_id")); ev = events.get(eid)
                if ev is None: continue
                src = source_xy_for_event(ev, initial_pos_xy)
                exp = expected_tdoa_s(initial_pos_xy, src, u, ref_unit, sound_speed_m_s)
                obs = float(row.get("measured_tdoa_s", 0.0))
                vals.append(obs - exp)
            if vals:
                x0[npos + off_index[u]] = float(np.median(vals))
    except Exception:
        pass
    res = optimize.least_squares(residuals, x0, loss="soft_l1", f_scale=2.0, max_nfev=10000)
    pos_fit, offsets_fit = unpack(res.x)

    residual_rows: List[Dict[str, Any]] = []
    for row in trows:
        u = str(row.get("unit"))
        eid = str(row.get("event_id"))
        if u == ref_unit or u not in pos_fit or eid not in events:
            continue
        src = source_xy_for_event(events[eid], pos_fit)
        obs = float(row.get("measured_tdoa_s", 0.0))
        pred_geom = expected_tdoa_s(pos_fit, src, u, ref_unit, sound_speed_m_s)
        pred = pred_geom + offsets_fit.get(u, 0.0) - offsets_fit.get(ref_unit, 0.0)
        residual_rows.append({
            "event_id": eid,
            "unit": u,
            "ref_unit": ref_unit,
            "measured_tdoa_s": f"{obs:.9f}",
            "model_tdoa_s": f"{pred:.9f}",
            "geometric_tdoa_s": f"{pred_geom:.9f}",
            "residual_s": f"{(obs - pred):+.9f}",
            "sigma_s": row.get("sigma_s", ""),
        })
    return pos_fit, offsets_fit, residual_rows


def write_calibration_txt(
    path: Path,
    ref_unit: str,
    sound_speed_m_s: float,
    input_latlon: Dict[str, Tuple[float, float]],
    fit_pos_xy: Dict[str, np.ndarray],
    timing_offsets: Dict[str, float],
    ref_latlon: Tuple[float, float],
    tables: Dict[str, List[Dict[str, Any]]],
    residual_rows: List[Dict[str, Any]],
    position_prior_sigma_m: float,
    timing_prior_sigma_s: float,
) -> None:
    stations = []
    for u in sorted(input_latlon):
        fit_ll = xy_to_latlon(float(fit_pos_xy[u][0]), float(fit_pos_xy[u][1]), ref_latlon)
        in_xy = fit_pos_xy[u] * 0.0  # placeholder overwritten below
        stations.append({
            "unit": u,
            "input_lat": f"{input_latlon[u][0]:.9f}",
            "input_lon": f"{input_latlon[u][1]:.9f}",
            "fit_lat": f"{fit_ll[0]:.9f}",
            "fit_lon": f"{fit_ll[1]:.9f}",
            "timing_offset_s": f"{timing_offsets.get(u, 0.0):+.9f}",
            "position_prior_sigma_m": f"{position_prior_sigma_m:.3f}",
        })
    out_tables: Dict[str, List[Dict[str, Any]]] = {}
    out_tables["stations"] = stations
    out_tables["events"] = tables.get("events", [])
    out_tables["tdoa"] = tables.get("tdoa", [])
    out_tables["fit_residuals"] = residual_rows
    kv = {
        "format": "ARU_SESSION_CALIBRATION_TXT_v1",
        "ref_unit": ref_unit,
        "sound_speed_m_s": f"{sound_speed_m_s:.6f}",
        "position_prior_sigma_m": f"{position_prior_sigma_m:.3f}",
        "timing_prior_sigma_s": f"{timing_prior_sigma_s:.6f}",
    }
    write_sectioned_tables(path, kv, out_tables, header="Created/updated by aru_calibrate_session.py")
