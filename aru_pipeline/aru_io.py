#!/usr/bin/env python3
"""Shared I/O and geometry helpers for the ARU localization/calibration scripts.

The text files in this pipeline are deliberately simple:
  - key/value files:  key = value
  - sectioned tables: lines beginning with [section] followed by tab-separated headers/rows

The readers are forgiving and also understand legacy JSON files so older event
clip directories can still be processed while new outputs are TXT/H5/CSV/HTML.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

EARTH_RADIUS_M = 6378137.0
TS_RE = re.compile(r"(?P<d>\d{4}-\d{2}-\d{2})[_ T](?P<h>\d{2}[-:]\d{2}[-:]\d{2})(?P<frac>\.\d+)?")
HMS_RE = re.compile(r"^(?P<h>\d{2})[-:](?P<m>\d{2})[-:](?P<s>\d{2})(?P<frac>\.\d+)?$")
NAME_RE = re.compile(
    r"(?P<start>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?)--(?P<unit>[^-].*?)--(?P<end>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?)\.(?P<ext>flac|tracking|txt)$",
    re.IGNORECASE,
)


def parse_sbts_dt(text: str) -> datetime:
    s = str(text).strip().replace("__", "_")
    m = TS_RE.search(s)
    if m:
        date = m.group("d")
        hms = m.group("h").replace("-", ":")
        frac = m.group("frac") or ""
        fmt = "%Y-%m-%d %H:%M:%S.%f" if frac else "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(f"{date} {hms}{frac}", fmt)
    # Deliberately small fallback set to avoid dateutil dependency here.
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d_%H-%M-%S.%f", "%Y-%m-%d_%H-%M-%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse SBTS datetime: {text!r}")


def format_sbts_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H-%M-%S.%f")


def parse_offset_seconds(s: str) -> float:
    s = str(s).strip()
    if ":" not in s:
        return float(s)
    parts = s.split(":")
    if len(parts) == 2:
        return 60.0 * float(parts[0]) + float(parts[1])
    if len(parts) == 3:
        return 3600.0 * float(parts[0]) + 60.0 * float(parts[1]) + float(parts[2])
    raise ValueError(f"Bad offset string: {s!r}")


def maybe_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def maybe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


def write_kv_txt(path: Path, data: Dict[str, Any], header: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if header:
        for line in header.splitlines():
            lines.append(f"# {line}")
    for k, v in data.items():
        if isinstance(v, (dict, list, tuple)):
            v = json.dumps(v, sort_keys=True)
        lines.append(f"{k} = {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_kv_txt(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_legacy_json_or_txt(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return read_kv_txt(path)


def write_sectioned_tables(path: Path, kv: Dict[str, Any], tables: Dict[str, List[Dict[str, Any]]], header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if header:
        for line in header.splitlines():
            lines.append(f"# {line}")
    for k, v in kv.items():
        lines.append(f"{k} = {v}")
    for name, rows in tables.items():
        lines.append("")
        lines.append(f"[{name}]")
        if not rows:
            continue
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        lines.append("\t".join(keys))
        for row in rows:
            lines.append("\t".join(str(row.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_sectioned_tables(path: Path) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, str]]]]:
    kv: Dict[str, str] = {}
    tables: Dict[str, List[Dict[str, str]]] = {}
    if not path.exists():
        return kv, tables
    section: Optional[str] = None
    header: Optional[List[str]] = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            tables.setdefault(section, [])
            header = None
            continue
        if section is None:
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if header is None:
            header = parts
            continue
        row = {k: (parts[i] if i < len(parts) else "") for i, k in enumerate(header)}
        tables[section].append(row)
    return kv, tables


@dataclass
class UnitFiles:
    unit: str
    unit_dir: Path
    flac_path: Path
    tracking_path: Path
    clip_metadata_path: Optional[Path]
    location_summary_path: Optional[Path]


def find_unit_files(event_dir: Path, units: Optional[Sequence[str]] = None) -> Dict[str, UnitFiles]:
    event_dir = Path(event_dir)
    out: Dict[str, UnitFiles] = {}
    candidates = [p for p in event_dir.iterdir() if p.is_dir()]
    if units:
        candidates = [event_dir / u for u in units if (event_dir / u).is_dir()]
    for udir in candidates:
        unit = udir.name
        flacs = sorted(udir.glob("*.flac"))
        trks = sorted(udir.glob("*.tracking"))
        if not flacs or not trks:
            continue
        metas = sorted(udir.glob("*.clip_metadata.txt")) or sorted(udir.glob("*.clip_metadata.json"))
        locs = [udir / "location_summary.txt", udir / "location_summary.json"]
        loc_path = next((p for p in locs if p.exists()), None)
        out[unit] = UnitFiles(
            unit=unit,
            unit_dir=udir,
            flac_path=flacs[0],
            tracking_path=trks[0],
            clip_metadata_path=metas[0] if metas else None,
            location_summary_path=loc_path,
        )
    return out


def read_clip_metadata(unit_files: UnitFiles) -> Dict[str, Any]:
    if unit_files.clip_metadata_path is None:
        return {}
    return read_legacy_json_or_txt(unit_files.clip_metadata_path)


def read_location_summary(unit_files: UnitFiles) -> Dict[str, Any]:
    if unit_files.location_summary_path is None:
        return {}
    d = read_legacy_json_or_txt(unit_files.location_summary_path)
    flat = flatten_dict(d) if isinstance(d, dict) else {}
    # Preserve original scalar keys as well as flattened keys.
    if isinstance(d, dict):
        flat.update({k: v for k, v in d.items() if not isinstance(v, dict)})
    return flat


def extract_lat_lon(d: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    lat_keys = ["lat", "latitude", "mean_lat", "avg_lat", "gps_mean_lat", "location.lat", "location.latitude", "mean.lat"]
    lon_keys = ["lon", "lng", "longitude", "mean_lon", "avg_lon", "gps_mean_lon", "location.lon", "location.longitude", "mean.lon"]
    lat = next((maybe_float(d.get(k)) for k in lat_keys if k in d and math.isfinite(maybe_float(d.get(k)))), float("nan"))
    lon = next((maybe_float(d.get(k)) for k in lon_keys if k in d and math.isfinite(maybe_float(d.get(k)))), float("nan"))
    if math.isfinite(lat) and math.isfinite(lon):
        return lat, lon
    return None


def extract_position_sigma_m(d: Dict[str, Any], default: float = 3.0) -> float:
    keys = ["sigma_m", "position_sigma_m", "gps_sigma_m", "cep68_m", "accuracy_m", "mean_accuracy_m", "horizontal_accuracy_m"]
    for k in keys:
        if k in d:
            val = maybe_float(d[k])
            if math.isfinite(val) and val > 0:
                return val
    # cep95 is wider; convert roughly to one-sigma radial scale.
    for k in ["cep95_m", "radius95_m", "accuracy95_m"]:
        if k in d:
            val = maybe_float(d[k])
            if math.isfinite(val) and val > 0:
                return max(0.1, val / 2.45)
    return float(default)


def load_station_locations(event_dir: Path, units: Optional[Sequence[str]] = None, override_txt: Optional[Path] = None, default_sigma_m: float = 3.0) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    files = find_unit_files(event_dir, units)
    for unit, uf in files.items():
        d = read_location_summary(uf)
        ll = extract_lat_lon(d)
        if ll:
            out[unit] = {
                "unit": unit,
                "lat": float(ll[0]),
                "lon": float(ll[1]),
                "sigma_m": extract_position_sigma_m(d, default_sigma_m),
                "source": str(uf.location_summary_path) if uf.location_summary_path else "location_summary",
            }
    if override_txt and Path(override_txt).exists():
        _, tables = read_sectioned_tables(Path(override_txt))
        rows = tables.get("stations", []) or tables.get("locations", [])
        for row in rows:
            unit = row.get("unit", "").strip()
            if not unit:
                continue
            lat = maybe_float(row.get("lat", row.get("fit_lat", row.get("input_lat"))))
            lon = maybe_float(row.get("lon", row.get("fit_lon", row.get("input_lon"))))
            if math.isfinite(lat) and math.isfinite(lon):
                out[unit] = {
                    "unit": unit,
                    "lat": lat,
                    "lon": lon,
                    "sigma_m": maybe_float(row.get("position_sigma_m", row.get("sigma_m")), default_sigma_m),
                    "source": str(override_txt),
                }
    return out


def latlon_to_xy_m(lat: float, lon: float, ref_latlon: Tuple[float, float]) -> Tuple[float, float]:
    lat0, lon0 = ref_latlon
    lat0_rad = math.radians(lat0)
    dlat_rad = math.radians(lat - lat0)
    dlon_rad = math.radians(lon - lon0)
    return (EARTH_RADIUS_M * dlon_rad * math.cos(lat0_rad), EARTH_RADIUS_M * dlat_rad)


def xy_to_latlon(x_m: float, y_m: float, ref_latlon: Tuple[float, float]) -> Tuple[float, float]:
    lat0, lon0 = ref_latlon
    lat0_rad = math.radians(lat0)
    dlat = (y_m / EARTH_RADIUS_M) * 180.0 / math.pi
    dlon = (x_m / (EARTH_RADIUS_M * math.cos(lat0_rad))) * 180.0 / math.pi
    return lat0 + dlat, lon0 + dlon


def stations_to_xy(stations: Dict[str, Dict[str, Any]], ref_unit: str) -> Dict[str, np.ndarray]:
    if ref_unit not in stations:
        raise ValueError(f"Reference unit {ref_unit!r} has no station location")
    ref_latlon = (float(stations[ref_unit]["lat"]), float(stations[ref_unit]["lon"]))
    return {u: np.asarray(latlon_to_xy_m(float(s["lat"]), float(s["lon"]), ref_latlon), dtype=float) for u, s in stations.items()}


def xy_stations_to_latlon(pos_xy: Dict[str, np.ndarray], ref_latlon: Tuple[float, float]) -> Dict[str, Tuple[float, float]]:
    return {u: xy_to_latlon(float(p[0]), float(p[1]), ref_latlon) for u, p in pos_xy.items()}


def read_mono_segment(path: Path, start_s: float, dur_s: float) -> Tuple[np.ndarray, int, int]:
    if sf is None:
        raise RuntimeError("Missing soundfile. Install with: pip install soundfile")
    info = sf.info(str(path))
    fs = int(info.samplerate)
    n = int(info.frames)
    start_sample = max(0, min(n, int(round(float(start_s) * fs))))
    stop = max(0, min(n, start_sample + int(round(float(dur_s) * fs))))
    audio, _ = sf.read(str(path), start=start_sample, stop=stop, always_2d=True)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32), fs, start_sample
    mono = audio.mean(axis=1).astype(np.float32)
    return mono, fs, start_sample


def audio_info(path: Path) -> Tuple[int, int, float]:
    if sf is None:
        raise RuntimeError("Missing soundfile. Install with: pip install soundfile")
    info = sf.info(str(path))
    fs = int(info.samplerate)
    n = int(info.frames)
    return fs, n, n / fs


@dataclass
class SimpleClock:
    unit: str
    ref_epoch: datetime
    x0: float
    a: float
    b: float
    buffer_samples: int
    sigma_s: float

    def time_from_sample(self, sample_index: float) -> datetime:
        bidx = float(sample_index) / float(self.buffer_samples)
        seconds = self.a + self.b * (bidx - self.x0)
        return self.ref_epoch + timedelta(seconds=seconds)

    def sample_from_time(self, dt: datetime) -> float:
        y = (dt - self.ref_epoch).total_seconds()
        bidx = self.x0 + (y - self.a) / max(self.b, 1e-12)
        return bidx * float(self.buffer_samples)


def read_tracking_rows(path: Path, date_hint: Optional[datetime] = None) -> Tuple[np.ndarray, List[datetime]]:
    idxs: List[int] = []
    dts: List[datetime] = []
    if date_hint is None:
        m = TS_RE.search(path.name)
        date_hint = parse_sbts_dt(m.group(0)) if m else datetime.now()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except Exception:
            continue
        ts = parts[1]
        try:
            if TS_RE.search(ts):
                dt = parse_sbts_dt(ts)
            else:
                m = HMS_RE.match(ts)
                if not m:
                    continue
                us = int(((m.group("frac") or ".0")[1:] + "000000")[:6])
                dt = datetime(date_hint.year, date_hint.month, date_hint.day, int(m.group("h")), int(m.group("m")), int(m.group("s")), us)
        except Exception:
            continue
        idxs.append(idx)
        dts.append(dt)
    if not idxs:
        raise RuntimeError(f"No tracking rows parsed from {path}")
    order = np.argsort(np.asarray(idxs))
    idx = np.asarray(idxs, dtype=int)[order]
    dts_sorted = [dts[int(i)] for i in order]
    fixed = [dts_sorted[0]]
    day_offset = 0
    for dt in dts_sorted[1:]:
        cur = dt + timedelta(days=day_offset)
        if cur < fixed[-1] - timedelta(seconds=1):
            day_offset += 1
            cur = dt + timedelta(days=day_offset)
        fixed.append(cur)
    return idx, fixed


def fit_simple_clock(unit: str, flac_path: Path, tracking_path: Path, buffer_samples: Optional[int] = None) -> SimpleClock:
    fs, nframes, _ = audio_info(flac_path)
    idx, dts = read_tracking_rows(tracking_path)
    ts = np.asarray([d.timestamp() for d in dts], dtype=float)
    ref_epoch = datetime.fromtimestamp(float(ts[0]))
    y = ts - ts[0]
    x0 = float(np.median(idx))
    x = idx.astype(float) - x0
    if len(idx) >= 3:
        # Robust-ish fallback without scipy dependency in this module.
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = float(np.median(np.diff(y) / np.maximum(np.diff(idx), 1))) if len(idx) > 1 else 2048 / fs
        intercept = float(np.median(y - slope * x))
    resid = y - (intercept + slope * x)
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) if resid.size else 1e-4
    if buffer_samples is None:
        inferred = int(round(float(slope) * fs)) if slope > 0 else 2048
        span = int(round(nframes / max(1, int(idx.max() - idx.min() + 1))))
        buffer_samples = span if inferred * 0.7 <= span <= inferred * 1.3 else inferred
    return SimpleClock(unit, ref_epoch, x0, float(intercept), float(slope), int(buffer_samples), max(float(sigma), 1e-6))


def fit_clock_maps(unit_files: Dict[str, UnitFiles], *, units: Sequence[str], draws: int = 2000, max_fit_rows: int = 2500, buffer_samples: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """Fit clocks. Prefer bayesian_clock_map_scipy.py if importable; otherwise fallback.

    Returned objects all implement time_from_sample(sample) and sample_from_time(datetime)
    through the wrapper class below.
    """
    try:
        import bayesian_clock_map_scipy as bcm  # type: ignore
    except Exception:
        bcm = None

    class BCMWrapper:
        def __init__(self, fit: Any):
            self.fit = fit
            self.unit = fit.summary.unit
            self.sigma_s = float(fit.summary.sigma_s_median)
        def time_from_sample(self, sample_index: float) -> datetime:
            pred = self.fit.predict_time_from_sample(float(sample_index))
            return parse_sbts_dt(str(pred["time_median_iso"]))
        def sample_from_time(self, dt: datetime) -> float:
            pred = self.fit.predict_sample_from_time(dt)
            return float(pred["sample_median"])

    out: Dict[str, Any] = {}
    for k, unit in enumerate(units):
        if unit not in unit_files:
            continue
        uf = unit_files[unit]
        if bcm is not None:
            try:
                data = bcm.load_tracking_data(unit, str(uf.flac_path), str(uf.tracking_path), units)
                fit = bcm.fit_clock(
                    data,
                    nu=4.0,
                    max_fit_rows=max_fit_rows,
                    draws=draws,
                    seed=seed + 7919 * k,
                    prior_slope_frac=0.05,
                    prior_intercept_s=10.0,
                    prior_log_sigma_sd=2.0,
                    buffer_samples_hint=buffer_samples,
                )
                out[unit] = BCMWrapper(fit)
                continue
            except Exception:
                pass
        out[unit] = fit_simple_clock(unit, uf.flac_path, uf.tracking_path, buffer_samples=buffer_samples)
    return out
