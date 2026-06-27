#!/usr/bin/env python3
"""BirdNET candidate detection for ARU birdcall localization."""
from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd

from align import BirdcallCandidate


def _load_birdnet_model():
    try:
        import birdnet  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import the birdnet package. Run this under the "
            "acoustic_camera conda environment with birdnet installed."
        ) from e
    return birdnet.load("acoustic", "2.4", "tf")


def _as_float_seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 3:
        return 3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
    if len(parts) == 2:
        return 60.0 * parts[0] + parts[1]
    raise ValueError(f"Bad BirdNET time value: {value!r}")


def _prediction_dataframe(predictions) -> pd.DataFrame:
    if hasattr(predictions, "to_dataframe"):
        df = predictions.to_dataframe()
    elif hasattr(predictions, "to_pandas"):
        df = predictions.to_pandas()
    else:
        raise RuntimeError("BirdNET prediction object has no to_dataframe() method")
    required = {"input", "start_time", "end_time", "species_name", "confidence"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"BirdNET predictions missing columns: {missing}")
    out = df.copy()
    out["start_time"] = out["start_time"].map(_as_float_seconds)
    out["end_time"] = out["end_time"].map(_as_float_seconds)
    out["confidence"] = out["confidence"].astype(float)
    out["species_name"] = out["species_name"].astype(str)
    return out


def run_birdnet_predictions(
    flac_path: Path,
    *,
    top_k: int = 5,
    threshold: float = 0.25,
    overlap_s: float = 2.8,
    bandpass_fmin: int = 1000,
    bandpass_fmax: int = 9000,
    batch_size: int = 8,
    n_workers: int = 1,
    device: str = "CPU",
    custom_species_list: Optional[Union[str, Path, Iterable[str]]] = None,
) -> pd.DataFrame:
    model = _load_birdnet_model()
    predictions = model.predict(
        str(Path(flac_path)),
        top_k=int(top_k),
        overlap_duration_s=float(overlap_s),
        batch_size=int(batch_size),
        n_workers=max(1, int(n_workers)),
        default_confidence_threshold=float(threshold),
        bandpass_fmin=int(bandpass_fmin),
        bandpass_fmax=int(bandpass_fmax),
        show_stats=None,
        device=str(device),
        custom_species_list=custom_species_list,
    )
    return _prediction_dataframe(predictions)


def merge_birdnet_candidates(
    df: pd.DataFrame,
    *,
    unit: str,
    threshold: float = 0.25,
    species_query: Optional[str] = None,
    merge_gap_s: float = 0.4,
    max_candidates: int = 8,
    hint_s: Optional[float] = None,
) -> List[BirdcallCandidate]:
    work = df.copy()
    if species_query:
        q = str(species_query).lower()
        work = work[work["species_name"].str.lower().str.contains(q, regex=False)]
    above = work[work["confidence"] >= float(threshold)].copy()
    fallback = False
    if above.empty:
        above = work.sort_values("confidence", ascending=False).head(max(1, int(max_candidates))).copy()
        fallback = True

    merged = []
    for species, grp in above.sort_values(["species_name", "start_time", "end_time"]).groupby("species_name"):
        cur = None
        for row in grp.itertuples(index=False):
            start = float(row.start_time)
            end = float(row.end_time)
            conf = float(row.confidence)
            if cur is None:
                cur = {
                    "species_name": str(species),
                    "start_s": start,
                    "end_s": end,
                    "max_conf": conf,
                    "sum_conf": conf,
                    "count": 1,
                }
                continue
            if start <= float(cur["end_s"]) + float(merge_gap_s):
                cur["end_s"] = max(float(cur["end_s"]), end)
                cur["max_conf"] = max(float(cur["max_conf"]), conf)
                cur["sum_conf"] = float(cur["sum_conf"]) + conf
                cur["count"] = int(cur["count"]) + 1
            else:
                merged.append(cur)
                cur = {
                    "species_name": str(species),
                    "start_s": start,
                    "end_s": end,
                    "max_conf": conf,
                    "sum_conf": conf,
                    "count": 1,
                }
        if cur is not None:
            merged.append(cur)

    candidates: List[BirdcallCandidate] = []
    for row in merged:
        start = float(row["start_s"])
        end = float(row["end_s"])
        center = 0.5 * (start + end)
        if hint_s is None:
            distance = float("nan")
        elif start <= float(hint_s) <= end:
            distance = 0.0
        else:
            distance = min(abs(float(hint_s) - start), abs(float(hint_s) - end))
        mean_conf = float(row["sum_conf"]) / max(1, int(row["count"]))
        score = float(row["max_conf"]) + 0.10 * mean_conf + 0.01 * math.log1p(int(row["count"]))
        quality = "below_threshold_fallback" if fallback else "ok"
        candidates.append(
            BirdcallCandidate(
                unit=unit,
                start_s=start,
                end_s=end,
                center_s=center,
                species_name=str(row["species_name"]),
                confidence=float(row["max_conf"]),
                score=score,
                source_count=int(row["count"]),
                quality=quality,
                distance_to_hint_s=distance,
            )
        )

    def sort_key(c: BirdcallCandidate):
        if hint_s is None:
            return (0, -float(c.score), -float(c.confidence), float(c.start_s))
        contains_rank = 0 if c.distance_to_hint_s == 0.0 else 1
        dist = c.distance_to_hint_s if math.isfinite(c.distance_to_hint_s) else float("inf")
        return (contains_rank, dist, -float(c.score), -float(c.confidence), float(c.start_s))

    candidates.sort(key=sort_key)
    return candidates[: int(max_candidates)] if max_candidates and max_candidates > 0 else candidates


def write_birdnet_candidates_csv(path: Path, candidates: Iterable[BirdcallCandidate], *, selected: Optional[BirdcallCandidate] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, cand in enumerate(candidates, start=1):
        row = asdict(cand)
        row["rank"] = rank
        row["selected"] = "yes" if selected is not None and cand == selected else ""
        rows.append(row)
    keys = [
        "rank",
        "selected",
        "unit",
        "start_s",
        "end_s",
        "center_s",
        "species_name",
        "confidence",
        "score",
        "source_count",
        "quality",
        "distance_to_hint_s",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
