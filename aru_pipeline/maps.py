#!/usr/bin/env python3
"""Polished Folium maps for station calibration and event localization."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Any
import math
import json
from pathlib import Path
import numpy as np

from aru_io import xy_to_latlon, stations_to_xy


def _folium_base(center: Tuple[float, float], zoom_start: int = 20):
    import folium
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=None, max_zoom=22, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite imagery",
        overlay=False,
        control=True,
        max_zoom=22,
        max_native_zoom=20,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True, max_zoom=22).add_to(m)
    try:
        folium.map.CustomPane("contours", z_index=410).add_to(m)
        folium.map.CustomPane("constraints", z_index=620).add_to(m)
        folium.map.CustomPane("locations", z_index=820).add_to(m)
    except Exception:
        pass
    return m


def _center_from_latlons(latlons: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    pts = list(latlons)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))




def _geojson_feature(geometry: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _pt_feature(lat: float, lon: float, properties: dict[str, Any]) -> dict[str, Any]:
    return _geojson_feature({"type": "Point", "coordinates": [float(lon), float(lat)]}, properties)


def _line_feature(latlons: List[Tuple[float, float]], properties: dict[str, Any]) -> dict[str, Any]:
    return _geojson_feature({"type": "LineString", "coordinates": [[float(lon), float(lat)] for lat, lon in latlons]}, properties)


def _poly_feature(latlons: List[Tuple[float, float]], properties: dict[str, Any]) -> dict[str, Any]:
    coords = [[float(lon), float(lat)] for lat, lon in latlons]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return _geojson_feature({"type": "Polygon", "coordinates": [coords]}, properties)


def _write_geojson_companion(out_html: str, features: List[dict[str, Any]]) -> None:
    out = Path(out_html).with_suffix(".geojson")
    fc = {"type": "FeatureCollection", "features": features}
    out.write_text(json.dumps(fc, indent=2), encoding="utf-8")

def hyperbola_segments(pos_xy: Dict[str, np.ndarray], ref_unit: str, unit: str, delta_m: float, ref_latlon: Tuple[float, float], extent_m: Optional[float] = None, n: int = 500) -> List[List[Tuple[float, float]]]:
    import matplotlib.pyplot as plt
    if ref_unit not in pos_xy or unit not in pos_xy:
        return []
    pts = np.stack(list(pos_xy.values()))
    dmax = float(np.max(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1))) if len(pts) else 100.0
    extent = float(extent_m or max(3*dmax, 120.0))
    sxr, syr = pos_xy[ref_unit]
    sxu, syu = pos_xy[unit]
    baseline = float(np.linalg.norm(pos_xy[unit] - pos_xy[ref_unit]))
    if baseline > 0 and abs(delta_m) > baseline:
        # Clip very slightly for display if close; skip if impossible by a lot.
        if abs(delta_m) - baseline > 2.0:
            return []
        delta_m = math.copysign(max(0.0, baseline - 0.05), delta_m)
    xs = np.linspace(-extent, extent, int(n))
    ys = np.linspace(-extent, extent, int(n))
    X, Y = np.meshgrid(xs, ys)
    Z = (np.hypot(X - sxu, Y - syu) - np.hypot(X - sxr, Y - syr)) - float(delta_m)
    fig = plt.figure()
    try:
        cs = plt.contour(X, Y, Z, levels=[0.0])
        segs = cs.allsegs[0] if cs.allsegs else []
    finally:
        plt.close(fig)
    out = []
    for seg in segs:
        arr = np.asarray(seg)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        step = max(1, int(math.ceil(arr.shape[0] / 600)))
        out.append([xy_to_latlon(float(x), float(y), ref_latlon) for x, y in arr[::step]])
    return out


def write_station_fit_map(out_html: str, input_latlon: Dict[str, Tuple[float, float]], fit_latlon: Dict[str, Tuple[float, float]], source_latlons: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
    import folium
    center = _center_from_latlons(list(input_latlon.values()) + list(fit_latlon.values()) + list((source_latlons or {}).values()))
    m = _folium_base(center, zoom_start=20)
    features: List[dict[str, Any]] = []
    fg_prior = folium.FeatureGroup(name="Input GPS/location-summary stations", show=True)
    fg_fit = folium.FeatureGroup(name="Best-fit stations", show=True)
    fg_lines = folium.FeatureGroup(name="Station displacement vectors", show=True)
    for u, ll in input_latlon.items():
        folium.CircleMarker(ll, radius=6, color="#1f77b4", fill=True, fill_opacity=0.9, tooltip=f"input {u}", popup=f"Input {u}<br>{ll[0]:.8f}, {ll[1]:.8f}", pane="locations").add_to(fg_prior)
        features.append(_pt_feature(ll[0], ll[1], {"kind": "input_station", "unit": u, "name": f"input {u}", "radius": 6}))
    for u, ll in fit_latlon.items():
        folium.Marker(ll, tooltip=f"fit {u}", popup=f"Best-fit {u}<br>{ll[0]:.8f}, {ll[1]:.8f}", icon=folium.Icon(color="green", icon="ok-sign"), pane="locations").add_to(fg_fit)
        features.append(_pt_feature(ll[0], ll[1], {"kind": "fit_station", "unit": u, "name": f"fit {u}", "radius": 8}))
        if u in input_latlon:
            folium.PolyLine([input_latlon[u], ll], color="#ffffff", weight=5, opacity=0.7, pane="constraints").add_to(fg_lines)
            folium.PolyLine([input_latlon[u], ll], color="#2ca02c", weight=2, opacity=0.95, pane="constraints").add_to(fg_lines)
            features.append(_line_feature([input_latlon[u], ll], {"kind": "station_displacement", "unit": u, "name": f"station displacement {u}"}))
    if source_latlons:
        fg_src = folium.FeatureGroup(name="Calibration clap/source locations", show=True)
        for name, ll in source_latlons.items():
            folium.Marker(ll, tooltip=f"source {name}", popup=f"Calibration source {name}<br>{ll[0]:.8f}, {ll[1]:.8f}", icon=folium.Icon(color="red", icon="star"), pane="locations").add_to(fg_src)
            features.append(_pt_feature(ll[0], ll[1], {"kind": "calibration_source", "name": str(name), "radius": 9}))
        fg_src.add_to(m)
    fg_prior.add_to(m); fg_lines.add_to(m); fg_fit.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    _write_geojson_companion(out_html, features)


def write_event_map(out_html: str, station_latlon: Dict[str, Tuple[float, float]], pos_xy: Dict[str, np.ndarray], ref_unit: str, source_xy: np.ndarray, tdoa_s: Dict[str, float], sound_speed_m_s: float, samples_xy: Optional[np.ndarray] = None, ellipse95_xy: Optional[np.ndarray] = None, loo_xy: Optional[Dict[str, np.ndarray]] = None) -> None:
    import folium
    ref_latlon = station_latlon[ref_unit]
    source_ll = xy_to_latlon(float(source_xy[0]), float(source_xy[1]), ref_latlon)
    lls = list(station_latlon.values()) + [source_ll]
    center = _center_from_latlons(lls)
    m = _folium_base(center, zoom_start=20)
    features: List[dict[str, Any]] = []

    # Lowest z-order: uncertainty samples/contours.
    if samples_xy is not None and samples_xy.size:
        fg_samples = folium.FeatureGroup(name="Localization MC samples", show=False)
        stride = max(1, len(samples_xy)//500)
        for xy in samples_xy[::stride]:
            ll = xy_to_latlon(float(xy[0]), float(xy[1]), ref_latlon)
            folium.CircleMarker(ll, radius=2, color="#ffff66", fill=True, fill_opacity=0.35, weight=0, pane="contours").add_to(fg_samples)
            features.append(_pt_feature(ll[0], ll[1], {"kind": "mc_sample", "name": "MC source sample", "radius": 2}))
        fg_samples.add_to(m)
    if ellipse95_xy is not None:
        poly = [xy_to_latlon(float(x), float(y), ref_latlon) for x, y in ellipse95_xy]
        folium.Polygon(poly, color="#ffff66", fill=True, fill_opacity=0.12, weight=2, tooltip="~95% location contour", pane="contours").add_to(m)
        features.append(_poly_feature(poly, {"kind": "ellipse95", "name": "~95% location contour"}))

    # Constraints above contours.
    fg_h = folium.FeatureGroup(name="TDOA hyperbola constraints", show=True)
    colors = ["#00ffff", "#ff7f0e", "#ff00ff", "#00ff00", "#ffffff"]
    for j, (u, tau) in enumerate(tdoa_s.items()):
        if u == ref_unit:
            continue
        for k, seg in enumerate(hyperbola_segments(pos_xy, ref_unit, u, sound_speed_m_s * float(tau), ref_latlon)):
            folium.PolyLine(seg, color=colors[j % len(colors)], weight=3, opacity=0.95, tooltip=f"{u}-{ref_unit}", pane="constraints").add_to(fg_h)
            features.append(_line_feature(seg, {"kind": "tdoa_hyperbola", "unit": u, "ref_unit": ref_unit, "name": f"{u}-{ref_unit}", "tdoa_s": float(tau), "segment": k}))
    fg_h.add_to(m)

    # Stations and source on top.
    fg_st = folium.FeatureGroup(name="Stations", show=True)
    for u, ll in station_latlon.items():
        folium.Marker(ll, tooltip=u, popup=f"Station {u}<br>{ll[0]:.8f}, {ll[1]:.8f}", icon=folium.Icon(color="blue", icon="info-sign"), pane="locations").add_to(fg_st)
        folium.PolyLine([source_ll, ll], color="#ffffff", weight=2, opacity=0.6, pane="constraints").add_to(fg_st)
        features.append(_pt_feature(ll[0], ll[1], {"kind": "station", "unit": u, "name": f"station {u}", "radius": 7}))
        features.append(_line_feature([source_ll, ll], {"kind": "source_station_line", "unit": u, "name": f"source to {u}"}))
    fg_st.add_to(m)
    folium.Marker(source_ll, tooltip="best-fit source", popup=f"Best-fit source<br>{source_ll[0]:.8f}, {source_ll[1]:.8f}", icon=folium.Icon(color="red", icon="star"), pane="locations").add_to(m)
    features.append(_pt_feature(source_ll[0], source_ll[1], {"kind": "best_fit_source", "name": "best-fit source", "radius": 9}))

    if loo_xy:
        fg_loo = folium.FeatureGroup(name="Leave-one-out localizations", show=True)
        for drop, xy in loo_xy.items():
            ll = xy_to_latlon(float(xy[0]), float(xy[1]), ref_latlon)
            folium.CircleMarker(ll, radius=7, color="#ffffff", weight=3, fill=True, fill_color="#ffcc00", fill_opacity=0.95, tooltip=f"LOO drop {drop}", popup=f"Leave-one-out localization<br>Dropped {drop}<br>{ll[0]:.8f}, {ll[1]:.8f}", pane="locations").add_to(fg_loo)
            features.append(_pt_feature(ll[0], ll[1], {"kind": "leave_one_out", "dropped_unit": drop, "name": f"LOO drop {drop}", "radius": 7}))
        fg_loo.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    _write_geojson_companion(out_html, features)
