#!/usr/bin/env python3
"""BirdNET location-prior species list helpers."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

from aru_io import find_unit_files, load_station_locations, parse_sbts_dt


def birdnet_week_from_datetime(dt: datetime) -> int:
    """Convert a date to BirdNET's 1..48 seasonal week bin."""
    doy = int(dt.timetuple().tm_yday)
    return max(1, min(48, int(math.ceil(doy / 7.0))))


def infer_event_datetime(event_dir: Path, unit: str = "five", units: Optional[Sequence[str]] = None) -> Optional[datetime]:
    unit_files = find_unit_files(Path(event_dir), units=units)
    uf = unit_files.get(unit) or next(iter(unit_files.values()), None)
    if uf is None:
        return None
    try:
        return parse_sbts_dt(uf.flac_path.name)
    except Exception:
        return None


def unit_location_from_event(
    event_dir: Path,
    *,
    unit: str = "five",
    units: Optional[Sequence[str]] = None,
    locations: Optional[Path] = None,
) -> Tuple[float, float]:
    stations = load_station_locations(Path(event_dir), units=units, override_txt=locations)
    if unit not in stations:
        available = ",".join(sorted(stations)) or "none"
        raise ValueError(f"No location for BirdNET geo unit {unit!r}; available units: {available}")
    return float(stations[unit]["lat"]), float(stations[unit]["lon"])


def write_species_list_from_location(
    out_path: Path,
    *,
    latitude: float,
    longitude: float,
    week: Optional[int],
    min_confidence: float = 0.03,
    device: str = "CPU",
) -> int:
    try:
        import birdnet  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import the birdnet package. Run this under the "
            "acoustic_camera conda environment with birdnet installed."
        ) from e

    geo_model = birdnet.load("geo", "2.4", "tf")
    result = geo_model.predict(
        float(latitude),
        float(longitude),
        week=int(week) if week is not None else None,
        min_confidence=float(min_confidence),
        device=str(device),
    )
    df = result.to_dataframe()
    species = sorted({str(s).strip() for s in df["species_name"].tolist() if str(s).strip()})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(species) + ("\n" if species else ""), encoding="utf-8")
    return len(species)


def build_event_species_list(
    event_dir: Path,
    out_path: Path,
    *,
    unit: str = "five",
    units: Optional[Sequence[str]] = None,
    locations: Optional[Path] = None,
    week: Optional[int] = None,
    min_confidence: float = 0.03,
    device: str = "CPU",
) -> tuple[Path, int, float, float, Optional[int]]:
    lat, lon = unit_location_from_event(event_dir, unit=unit, units=units, locations=locations)
    if week is None:
        dt = infer_event_datetime(event_dir, unit=unit, units=units)
        week = birdnet_week_from_datetime(dt) if dt is not None else None
    count = write_species_list_from_location(
        out_path,
        latitude=lat,
        longitude=lon,
        week=week,
        min_confidence=min_confidence,
        device=device,
    )
    return Path(out_path), count, lat, lon, week
