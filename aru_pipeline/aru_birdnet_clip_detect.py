#!/usr/bin/env python3
"""Run BirdNET on event clips and write minimal cross-unit call detections."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from align import BirdcallCandidate
from aru_io import find_unit_files
from birdnet_detect import merge_birdnet_candidates, run_birdnet_predictions, write_birdnet_candidates_csv
from birdnet_species import build_event_species_list


CROSS_UNIT_SOURCE = "predictions:min_overlap"


@dataclass
class BirdnetClipDetectionResult:
    input_path: Path
    out_dir: Path
    out_txt: Path
    clips: Dict[str, Path]
    predictions_by_unit: Dict[str, object]
    candidates_by_unit: Dict[str, List[BirdcallCandidate]]
    calls: List[dict]
    species_list_path: Optional[Path]
    highpass_hz: int
    min_units: int
    cross_unit_source: str = CROSS_UNIT_SOURCE


def _default_out_txt(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "birdnet_calls.txt"
    return input_path.with_suffix(input_path.suffix + ".birdnet_calls.txt")


def _parse_units(text: str | None) -> List[str] | None:
    if not text:
        return None
    units = [u.strip() for u in str(text).split(",") if u.strip()]
    return units or None


def _input_clips(input_path: Path, units: Sequence[str] | None) -> Dict[str, Path]:
    if input_path.is_dir():
        unit_files = find_unit_files(input_path, units)
        if not unit_files:
            raise SystemExit(f"No unit FLAC/tracking pairs found under event directory: {input_path}")
        return {unit: uf.flac_path for unit, uf in sorted(unit_files.items())}
    unit = units[0] if units else (input_path.parent.name or input_path.stem)
    return {unit: input_path}


def _window_species_hits_from_predictions(
    predictions_by_unit: Dict[str, object],
    min_prediction_confidence: float,
    species_query: str | None,
) -> List[dict]:
    """Return one hit per unit/window/species after raw confidence filtering."""
    hits = []
    for unit, predictions in predictions_by_unit.items():
        work = predictions.copy()
        if species_query:
            q = str(species_query).lower()
            work = work[work["species_name"].astype(str).str.lower().str.contains(q, regex=False)]
        if work.empty:
            continue
        work["confidence"] = work["confidence"].astype(float)
        work["start_time"] = work["start_time"].astype(float)
        work["end_time"] = work["end_time"].astype(float)
        idx = work.groupby(["start_time", "end_time", "species_name"])["confidence"].idxmax()
        best_rows = work.loc[idx].copy()
        best_rows = best_rows[best_rows["confidence"] >= float(min_prediction_confidence)]
        for row in best_rows.itertuples(index=False):
            hits.append({
                "unit": unit,
                "species_name": str(row.species_name),
                "start_s": float(row.start_time),
                "end_s": float(row.end_time),
                "confidence": float(row.confidence),
                "score": float(row.confidence),
                "source_windows": 1,
                "quality": "prediction_window_best",
            })
    return hits


def _summarize_overlap_interval(species: str, start_s: float, end_s: float, active_hits: List[dict]) -> dict:
    per_unit = {}
    for unit in sorted({hit["unit"] for hit in active_hits}):
        unit_hits = [hit for hit in active_hits if hit["unit"] == unit]
        best = max(unit_hits, key=lambda h: float(h["confidence"]))
        per_unit[unit] = {
            "unit": unit,
            "confidence": float(best["confidence"]),
            "source_windows": len(unit_hits),
            "species_name": str(best["species_name"]),
        }
    confidences = [float(item["confidence"]) for item in per_unit.values()]
    source_windows = sum(int(item["source_windows"]) for item in per_unit.values())
    best_unit = max(per_unit, key=lambda u: float(per_unit[u]["confidence"])) if per_unit else ""
    other_confidences = [float(item["confidence"]) for u, item in per_unit.items() if u != best_unit]
    species_by_unit = ";".join(f"{u}:{item['species_name']}" for u, item in per_unit.items())
    return {
        "species_name": str(species),
        "start_s": float(start_s),
        "end_s": float(end_s),
        "n_units": len(per_unit),
        "units": ",".join(per_unit.keys()),
        "avg_confidence": sum(confidences) / max(1, len(confidences)),
        "max_confidence": max(confidences) if confidences else float("nan"),
        "source_windows": source_windows,
        "per_unit_confidence": ";".join(f"{u}:{float(item['confidence']):.6f}" for u, item in per_unit.items()),
        "cross_unit_source": CROSS_UNIT_SOURCE,
        "species_by_unit": species_by_unit,
        "best_unit": best_unit,
        "other_min_confidence": min(other_confidences) if other_confidences else float("nan"),
    }


def _minimal_overlap_calls(
    hits: List[dict],
    operating_units: Sequence[str],
    min_units: int,
    primary_confidence: float,
    other_confidence: float,
    max_calls: int,
) -> List[dict]:
    operating = set(operating_units)
    required_count = int(min_units) if int(min_units) > 0 else len(operating)
    out = []
    by_species: Dict[str, List[dict]] = {}
    for hit in hits:
        if hit["unit"] in operating:
            by_species.setdefault(str(hit["species_name"]), []).append(hit)

    for species, species_hits in by_species.items():
        boundaries = sorted({
            round(float(hit["start_s"]), 6)
            for hit in species_hits
        } | {
            round(float(hit["end_s"]), 6)
            for hit in species_hits
        })
        for start_s, end_s in zip(boundaries, boundaries[1:]):
            if end_s <= start_s:
                continue
            active = [
                hit for hit in species_hits
                if float(hit["start_s"]) <= start_s + 1e-9
                and float(hit["end_s"]) >= end_s - 1e-9
            ]
            present = {hit["unit"] for hit in active}
            if len(present) < required_count:
                continue
            call = _summarize_overlap_interval(species, start_s, end_s, active)
            if not math.isfinite(float(call["max_confidence"])) or float(call["max_confidence"]) < float(primary_confidence):
                continue
            if len(present) > 1:
                other_min = float(call.get("other_min_confidence", float("nan")))
                if not math.isfinite(other_min) or other_min < float(other_confidence):
                    continue
            out.append(call)
    out.sort(key=lambda c: (-float(c["max_confidence"]), float(c["start_s"]), str(c["species_name"])))
    if max_calls > 0:
        out = out[:max_calls]
    out.sort(key=lambda c: (float(c["start_s"]), str(c["species_name"])))
    return out


def _write_call_txt(path: Path, calls: List[dict], cross_unit_source: str = CROSS_UNIT_SOURCE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# BirdNET cross-unit call list\n")
        f.write(f"# cross_unit_source = {cross_unit_source}\n")
        f.write("# call_id\tstart_s\tend_s\tcenter_s\tspecies_name\tavg_confidence\tmax_confidence\tn_units\tunits\tsource_windows\tper_unit_confidence\tcross_unit_source\tspecies_by_unit\tbest_unit\tother_min_confidence\n")
        for idx, call in enumerate(calls, start=1):
            call_id = f"call_{idx:04d}"
            start_s = float(call["start_s"])
            end_s = float(call["end_s"])
            f.write(
                "\t".join(
                    [
                        call_id,
                        f"{start_s:.6f}",
                        f"{end_s:.6f}",
                        f"{0.5 * (start_s + end_s):.6f}",
                        str(call["species_name"]),
                        f"{float(call['avg_confidence']):.9f}",
                        f"{float(call['max_confidence']):.9f}",
                        str(int(call["n_units"])),
                        str(call["units"]),
                        str(int(call["source_windows"])),
                        str(call["per_unit_confidence"]),
                        str(call["cross_unit_source"]),
                        str(call.get("species_by_unit", "")),
                        str(call.get("best_unit", "")),
                        f"{float(call['other_min_confidence']):.9f}" if "other_min_confidence" in call and math.isfinite(float(call["other_min_confidence"])) else "",
                    ]
                )
                + "\n"
            )


def _write_calls_csv(path: Path, calls: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "call_id",
        "start_s",
        "end_s",
        "center_s",
        "species_name",
        "avg_confidence",
        "max_confidence",
        "n_units",
        "units",
        "source_windows",
        "per_unit_confidence",
        "cross_unit_source",
        "species_by_unit",
        "best_unit",
        "other_min_confidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for idx, call in enumerate(calls, start=1):
            start_s = float(call["start_s"])
            end_s = float(call["end_s"])
            row = dict(call)
            row["call_id"] = f"call_{idx:04d}"
            row["center_s"] = 0.5 * (start_s + end_s)
            writer.writerow(row)


def _resolve_species_list(
    input_path: Path,
    out_dir: Path,
    clips: Dict[str, Path],
    *,
    locations: Optional[Path],
    birdnet_species_list: Optional[Path],
    birdnet_species_unit: str,
    birdnet_geo_week: Optional[int],
    birdnet_geo_min_confidence: float,
    no_birdnet_location_species_list: bool,
    device: str,
) -> tuple[Optional[Path], Optional[int], str]:
    if birdnet_species_list:
        return Path(birdnet_species_list).expanduser().resolve(), None, str(birdnet_species_unit)
    if no_birdnet_location_species_list:
        return None, None, str(birdnet_species_unit)

    species_event_dir = input_path if input_path.is_dir() else input_path.parent.parent
    species_unit = str(birdnet_species_unit)
    if not input_path.is_dir() and species_unit not in clips and len(clips) == 1:
        species_unit = next(iter(clips))
    species_path, species_count, _lat, _lon, week = build_event_species_list(
        species_event_dir,
        out_dir / "species.txt",
        unit=species_unit,
        units=list(clips.keys()),
        locations=locations,
        week=birdnet_geo_week,
        min_confidence=birdnet_geo_min_confidence,
        device=device,
    )
    print(
        f"Wrote BirdNET species list: {species_path} "
        f"({species_count} species; unit={species_unit}; week={week if week is not None else 'year-round'})"
    )
    return species_path, species_count, species_unit


def run_clip_detection(
    input_path: Path,
    *,
    units: Sequence[str] | None = None,
    out_dir: Path | None = None,
    out_txt: Path | None = None,
    write_csv: bool = False,
    min_units: int = 0,
    window_primary_confidence: float = 0.70,
    window_other_confidence: float = 0.01,
    locations: Path | None = None,
    birdnet_species_list: Path | None = None,
    birdnet_species_unit: str = "five",
    birdnet_geo_week: int | None = None,
    birdnet_geo_min_confidence: float = 0.03,
    no_birdnet_location_species_list: bool = False,
    threshold: float = 0.25,
    top_k: int = 5,
    overlap_s: float = 2.8,
    highpass_hz: int = 0,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int = 9000,
    species_query: str | None = None,
    merge_gap_s: float = 0.4,
    max_calls: int = 0,
    max_candidates: int = 8,
    hint_s: float | None = None,
    batch_size: int = 8,
    workers: int = 1,
    device: str = "CPU",
    write_outputs: bool = True,
) -> BirdnetClipDetectionResult:
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    if out_dir:
        out_dir = Path(out_dir).expanduser().resolve()
        out_txt = Path(out_txt).expanduser().resolve() if out_txt else out_dir / "birdnet_calls.txt"
    else:
        out_txt = Path(out_txt).expanduser().resolve() if out_txt else _default_out_txt(input_path)
        out_dir = out_txt.parent

    clips = _input_clips(input_path, units)
    required_min_units = int(min_units) if int(min_units) > 0 else len(clips)
    if required_min_units < 1:
        raise SystemExit("--min-units must be >= 1, or 0 for all operating units")
    if required_min_units > len(clips):
        raise SystemExit(f"--min-units={required_min_units} exceeds operating unit count {len(clips)}")

    highpass = int(bandpass_fmin) if bandpass_fmin is not None else int(highpass_hz)
    if highpass < 0:
        raise SystemExit("--highpass-hz/--bandpass-fmin must be >= 0")

    out_dir.mkdir(parents=True, exist_ok=True)
    species_list_path, _species_count, _species_unit = _resolve_species_list(
        input_path,
        out_dir,
        clips,
        locations=Path(locations).expanduser().resolve() if locations else None,
        birdnet_species_list=Path(birdnet_species_list).expanduser().resolve() if birdnet_species_list else None,
        birdnet_species_unit=birdnet_species_unit,
        birdnet_geo_week=birdnet_geo_week,
        birdnet_geo_min_confidence=birdnet_geo_min_confidence,
        no_birdnet_location_species_list=no_birdnet_location_species_list,
        device=device,
    )

    predictions_by_unit = {}
    candidates_by_unit = {}
    prediction_threshold = min(float(threshold), float(window_other_confidence))
    for unit, audio_path in clips.items():
        print(f"Running BirdNET: unit={unit} clip={audio_path}")
        predictions = run_birdnet_predictions(
            audio_path,
            threshold=prediction_threshold,
            top_k=top_k,
            overlap_s=overlap_s,
            bandpass_fmin=highpass,
            bandpass_fmax=bandpass_fmax,
            batch_size=batch_size,
            n_workers=workers,
            device=device,
            custom_species_list=str(species_list_path) if species_list_path else None,
        )
        candidates = merge_birdnet_candidates(
            predictions,
            unit=unit,
            threshold=threshold,
            species_query=species_query,
            merge_gap_s=merge_gap_s,
            max_candidates=max_candidates,
            hint_s=hint_s,
        )
        predictions_by_unit[unit] = predictions
        candidates_by_unit[unit] = candidates

    hits = _window_species_hits_from_predictions(
        predictions_by_unit,
        min_prediction_confidence=window_other_confidence,
        species_query=species_query,
    )
    calls = _minimal_overlap_calls(
        hits,
        operating_units=list(clips.keys()),
        min_units=required_min_units,
        primary_confidence=window_primary_confidence,
        other_confidence=window_other_confidence,
        max_calls=int(max_calls),
    )

    result = BirdnetClipDetectionResult(
        input_path=input_path,
        out_dir=out_dir,
        out_txt=out_txt,
        clips=clips,
        predictions_by_unit=predictions_by_unit,
        candidates_by_unit=candidates_by_unit,
        calls=calls,
        species_list_path=species_list_path,
        highpass_hz=highpass,
        min_units=required_min_units,
    )

    if write_outputs:
        _write_call_txt(out_txt, calls)
        print(f"Wrote BirdNET call list: {out_txt}")
        print(f"Operating units: {','.join(clips.keys())}")
        print(
            "Filter: "
            f"min_units={required_min_units}, "
            f"window_primary_confidence={float(window_primary_confidence):.3f}, "
            f"window_other_confidence={float(window_other_confidence):.3f}"
        )
        print(f"Cross-unit source: {CROSS_UNIT_SOURCE}")
        print(f"High-pass Hz: {highpass}")
        print(f"Species list: {species_list_path}")
        print(f"Calls written: {len(calls)}")

    if write_csv:
        calls_csv = out_dir / "birdnet_calls.csv"
        _write_calls_csv(calls_csv, calls)
        print(f"Wrote filtered calls CSV: {calls_csv}")
        for unit in clips:
            pred_path = out_dir / f"birdnet_predictions_{unit}.csv"
            cand_path = out_dir / f"birdnet_candidates_{unit}.csv"
            predictions_by_unit[unit].to_csv(pred_path, index=False)
            write_birdnet_candidates_csv(cand_path, candidates_by_unit[unit])
            print(f"Wrote raw predictions for {unit}: {pred_path}")
            print(f"Wrote merged candidates for {unit}: {cand_path}")

    return result


def select_reference_candidate(
    result: BirdnetClipDetectionResult,
    ref_unit: str,
    *,
    hint_s: float | None = None,
    hint_fallback_half_s: float | None = None,
) -> tuple[BirdcallCandidate, List[BirdcallCandidate], dict]:
    diagnostic_candidates = list(result.candidates_by_unit.get(ref_unit, []))
    calls = [
        (idx, call) for idx, call in enumerate(result.calls, start=1)
        if ref_unit in {u for u in str(call.get("units", "")).split(",") if u}
    ]
    if not calls:
        raise SystemExit(f"BirdNET produced no minimal overlap calls containing reference unit {ref_unit!r}")

    def distance_to_hint(call: dict) -> float:
        if hint_s is None:
            return float("nan")
        start_s = float(call["start_s"])
        end_s = float(call["end_s"])
        if start_s <= float(hint_s) <= end_s:
            return 0.0
        return min(abs(float(hint_s) - start_s), abs(float(hint_s) - end_s))

    if hint_s is not None:
        calls.sort(key=lambda item: (
            0 if float(item[1]["start_s"]) <= float(hint_s) <= float(item[1]["end_s"]) else 1,
            distance_to_hint(item[1]),
            float(item[1]["end_s"]) - float(item[1]["start_s"]),
            -float(item[1]["max_confidence"]),
            str(item[1]["species_name"]),
        ))
    else:
        calls.sort(key=lambda item: (
            -float(item[1]["max_confidence"]),
            float(item[1]["start_s"]),
            float(item[1]["end_s"]) - float(item[1]["start_s"]),
            str(item[1]["species_name"]),
        ))

    call_id, call = calls[0]
    start_s = float(call["start_s"])
    end_s = float(call["end_s"])
    distance = distance_to_hint(call)
    candidate = BirdcallCandidate(
        unit=ref_unit,
        start_s=start_s,
        end_s=end_s,
        center_s=0.5 * (start_s + end_s),
        species_name=str(call["species_name"]),
        confidence=float(call["max_confidence"]),
        score=float(call["avg_confidence"]),
        source_count=int(call["source_windows"]),
        quality=f"minimal_overlap_call_{call_id:04d}",
        distance_to_hint_s=distance,
    )
    selected_call = dict(call)
    selected_call["call_id"] = f"call_{call_id:04d}"
    selected_call["distance_to_hint_s"] = distance
    return candidate, diagnostic_candidates, selected_call


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run birdnet-team/birdnet on an event directory or one clip and write minimal-overlap call IDs with times."
    )
    p.add_argument("input", help="Input event directory or one audio clip, e.g. FLAC/WAV.")
    p.add_argument("--units", default=None, help="Comma-separated units to process for event-directory input. Default: all discovered units.")
    p.add_argument("--unit", default="", help="Single-clip unit label. Ignored when --units is set.")
    p.add_argument("--out-txt", default=None, help="Output tab-delimited text file. Defaults beside the input.")
    p.add_argument("--out-dir", default=None, help="Directory for birdnet_calls.txt and optional CSV outputs.")
    p.add_argument("--write-csv", action="store_true", help="Also write filtered calls, merged candidates, and raw predictions CSV files.")
    p.add_argument("--min-units", type=int, default=0, help="Minimum units that must identify the call. 0 means all operating units.")
    p.add_argument("--window-primary-confidence", type=float, default=0.70, help="Require at least one unit at or above this confidence.")
    p.add_argument("--window-other-confidence", type=float, default=0.01, help="Require other identifying units at or above this confidence.")
    p.add_argument("--locations", default=None, help="Optional station override TXT with [stations] for generated species.txt.")
    p.add_argument("--birdnet-species-list", default=None, help="Optional custom BirdNET species list. If omitted, generate out-dir/species.txt before detection.")
    p.add_argument("--birdnet-species-unit", default="five", help="Unit whose location is used for generated species.txt.")
    p.add_argument("--birdnet-geo-week", type=int, default=None, help="BirdNET 1..48 week override for generated species.txt. Default: infer from event clip date.")
    p.add_argument("--birdnet-geo-min-confidence", type=float, default=0.03, help="Minimum BirdNET geo prior confidence for generated species.txt.")
    p.add_argument("--no-birdnet-location-species-list", action="store_true", help="Disable automatic generated species.txt.")
    p.add_argument("--threshold", type=float, default=0.25, help="Merged-candidate threshold. Prediction-window filtering uses --window-other-confidence.")
    p.add_argument("--top-k", type=int, default=5, help="Top BirdNET labels retained per analysis window.")
    p.add_argument("--overlap-s", type=float, default=2.8, help="BirdNET overlap duration. 2.8 gives a 0.2 s hop for 3 s windows.")
    p.add_argument("--highpass-hz", type=int, default=0, help="BirdNET preprocessing high-pass frequency. Default 0 disables high-pass filtering.")
    p.add_argument("--bandpass-fmin", type=int, default=None, help="Deprecated alias/override for --highpass-hz.")
    p.add_argument("--bandpass-fmax", type=int, default=9000, help="BirdNET preprocessing bandpass high edge.")
    p.add_argument("--species-query", default=None, help="Optional case-insensitive substring filter for species_name.")
    p.add_argument("--merge-gap-s", type=float, default=0.4, help="Merge windows of the same species for per-unit candidate CSVs.")
    p.add_argument("--max-calls", type=int, default=0, help="Maximum calls to write. 0 means no limit.")
    p.add_argument("--max-candidates", type=int, default=8, help="Maximum per-unit merged candidates to keep.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="CPU")
    args = p.parse_args()

    units = _parse_units(args.units) or _parse_units(args.unit)
    run_clip_detection(
        Path(args.input),
        units=units,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        out_txt=Path(args.out_txt) if args.out_txt else None,
        write_csv=args.write_csv,
        min_units=args.min_units,
        window_primary_confidence=args.window_primary_confidence,
        window_other_confidence=args.window_other_confidence,
        locations=Path(args.locations) if args.locations else None,
        birdnet_species_list=Path(args.birdnet_species_list) if args.birdnet_species_list else None,
        birdnet_species_unit=args.birdnet_species_unit,
        birdnet_geo_week=args.birdnet_geo_week,
        birdnet_geo_min_confidence=args.birdnet_geo_min_confidence,
        no_birdnet_location_species_list=args.no_birdnet_location_species_list,
        threshold=args.threshold,
        top_k=args.top_k,
        overlap_s=args.overlap_s,
        highpass_hz=args.highpass_hz,
        bandpass_fmin=args.bandpass_fmin,
        bandpass_fmax=args.bandpass_fmax,
        species_query=args.species_query,
        merge_gap_s=args.merge_gap_s,
        max_calls=args.max_calls,
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        write_outputs=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
