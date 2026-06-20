#!/usr/bin/env python3
"""Idempotent acquisition layer for ARU calibration/localization workflows.

This module is intended to be the single acquisition prerequisite used by the
CLI scripts and by the FastAPI dashboard on unit five.

Core contract
-------------
ensure_event_clips(timestamp, units, clip_half_s, ...)
    1. Computes the canonical event directory under data/event_clips/.
    2. Reuses already-fetched clips when they are complete and cover the
       requested timestamp/window.
    3. If clips are missing, inspects the ARU units to see whether the target
       timestamp lies in a file that is probably still being written.
    4. If active files are detected and rotation is allowed, rotates all
       selected units once, waits for finalization, then fetches missing clips.
    5. Writes TXT acquisition manifests/status files and returns the event dir.

The implementation deliberately reuses rotate_fetch_clips_txt.py's tested
remote clipping worker and copy logic. The remote inspector uses JSON only as an
SSH/stdin process protocol; all persistent pipeline outputs remain TXT/CSV/H5/
PNG/HTML-friendly.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from aru_io import (
    format_sbts_dt,
    parse_sbts_dt,
    read_legacy_json_or_txt,
    write_kv_txt,
    write_sectioned_tables,
)
from rotate_fetch_clips_txt import (
    UnitConfig,
    event_run_dir,
    flatten_dict,
    format_ts,
    load_config,
    parse_event_time,
    process_unit,
    rotate_unit,
    units_from_config,
)

BASE_DIR = Path(__file__).resolve().parent

# Remote inspector: intentionally small and dependency-light. It returns enough
# information to decide whether a timestamp is in a finalized file, in a
# active/unfinalized sbts-aru file, or not found. It does not create files.
INSPECT_WORKER = r'''
from __future__ import annotations
import json, re, sys
from datetime import datetime, timedelta
from pathlib import Path

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?$")

def parse_ts(s):
    s = str(s).strip()
    if "." not in s: s += ".000000"
    base, frac = s.split(".", 1)
    frac = (frac + "000000")[:6]
    return datetime.strptime(base + "." + frac, "%Y-%m-%d_%H-%M-%S.%f")

def fmt(dt):
    return dt.strftime("%Y-%m-%d_%H-%M-%S.%f")

def split_completed_name(path):
    """Completed sbts-aru file: start--unit--end.flac."""
    if path.suffix.lower() != ".flac": return None
    parts = path.stem.split("--")
    if len(parts) < 3: return None
    start_s, unit, end_s = parts[0], parts[1], parts[-1]
    if not TS_RE.match(start_s) or not TS_RE.match(end_s): return None
    try: return parse_ts(start_s), unit, parse_ts(end_s)
    except Exception: return None

def split_active_name(path):
    """Active sbts-aru file: start.flac. The unit/end are added only after close/rename."""
    if path.suffix.lower() != ".flac": return None
    if "--" in path.stem: return None
    if not TS_RE.match(path.stem): return None
    try: return parse_ts(path.stem)
    except Exception: return None

def find_flacs_for_day(target):
    roots = [Path("/home/pi/disk"), Path("/disk"), Path.home()/"disk"]
    day, ym, year = target.strftime("%Y-%m-%d"), target.strftime("%Y-%m"), target.strftime("%Y")
    found = []
    seen = set()
    for root in roots:
        d = root/year/ym/day
        if d.is_dir():
            for f in sorted(d.glob("*.flac")):
                if str(f) not in seen:
                    found.append(f); seen.add(str(f))
    if found: return found
    for root in roots:
        if root.is_dir():
            try:
                for f in sorted(root.rglob("*.flac")):
                    if str(f) not in seen:
                        found.append(f); seen.add(str(f))
            except Exception: pass
    return found

def main():
    args = json.loads(sys.stdin.read() or "{}")
    target = parse_ts(args["target_sbts"])
    unit_name = str(args.get("unit_name") or "")
    now = datetime.now()
    flacs = find_flacs_for_day(target)
    candidates = []
    for flac in flacs:
        tracking = flac.with_suffix(".tracking")

        completed = split_completed_name(flac)
        if completed is not None:
            start, file_unit, end = completed
            contains_by_name = start <= target <= end
            if not contains_by_name:
                continue
            candidates.append({
                "score": 0,
                "filename_state": "completed",
                "source_flac": str(flac),
                "source_tracking": str(tracking),
                "source_start": fmt(start),
                "source_end": fmt(end),
                "source_unit": file_unit,
                "contains_by_filename": True,
                "active": False,
                "tracking_exists": tracking.exists(),
            })
            continue

        active_start = split_active_name(flac)
        if active_start is not None:
            # sbts-aru active files have no end in the filename. If the target is
            # between the active start and now, the file is still being written
            # and must be closed/renamed by HUP before the clip worker can read it.
            if active_start <= target <= now + timedelta(seconds=5):
                candidates.append({
                    "score": 1,
                    "filename_state": "active_unfinalized",
                    "source_flac": str(flac),
                    "source_tracking": str(tracking),
                    "source_start": fmt(active_start),
                    "source_end": "",
                    "source_unit": unit_name,
                    "contains_by_filename": False,
                    "active": True,
                    "tracking_exists": tracking.exists(),
                })
            continue

    # Prefer a finalized containing file if one exists; otherwise active files
    # deliberately trigger rotation in the acquisition layer.
    candidates.sort(key=lambda d: (d["score"], d["source_start"]), reverse=False)
    best = candidates[0] if candidates else None
    status = "not_found"
    if best:
        status = "active_contains_target" if best.get("active") else "finalized_contains_target"
    print(json.dumps({"ok": True, "status": status, "target_sbts": fmt(target), "best": best, "candidates": candidates[:8]}))

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)})); sys.exit(1)
'''


@dataclass
class AcquisitionResult:
    ok: bool
    event_dir: str
    reused: bool
    rotated: bool
    fetched_units: List[str]
    missing_units: List[str]
    manifest_path: str
    message: str


def _run_remote_inspect(unit: UnitConfig, target: datetime, timeout_s: float = 20.0) -> Dict[str, Any]:
    payload = {"target_sbts": format_ts(target), "unit_name": unit.name}
    if unit.transport == "local":
        cmd = ["python3", "-c", INSPECT_WORKER]
    else:
        remote_cmd = "python3 -c " + shlex.quote(INSPECT_WORKER)
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
            f"{unit.user}@{unit.host}", remote_cmd,
        ]
    proc = subprocess.run(cmd, input=json.dumps(payload), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    if proc.returncode != 0:
        return {"ok": False, "unit": unit.name, "status": "inspect_failed", "error": proc.stderr.strip() or proc.stdout.strip()}
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        d = json.loads(lines[-1])
    except Exception as e:
        return {"ok": False, "unit": unit.name, "status": "inspect_failed", "error": f"bad inspector output: {e}", "stdout": proc.stdout}
    d["unit"] = unit.name
    return d


def _event_unit_complete(unit_dir: Path, target: datetime, clip_half_s: float, require_full_window: bool = True) -> Tuple[bool, str]:
    if not unit_dir.is_dir():
        return False, "unit_dir_missing"
    flacs = sorted(unit_dir.glob("*.flac"))
    trks = sorted(unit_dir.glob("*.tracking"))
    metas = sorted(unit_dir.glob("*.clip_metadata.txt")) or sorted(unit_dir.glob("*.clip_metadata.json"))
    if not flacs: return False, "flac_missing"
    if not trks: return False, "tracking_missing"
    if not metas: return False, "clip_metadata_missing"
    meta = read_legacy_json_or_txt(metas[0])
    try:
        clip_start = parse_sbts_dt(str(meta.get("clip_start", "")))
        clip_end = parse_sbts_dt(str(meta.get("clip_end", "")))
    except Exception:
        return False, "bad_or_missing_clip_start_end"
    if not (clip_start <= target <= clip_end):
        return False, "target_not_inside_clip"
    if require_full_window:
        # Allow one 2048-sample buffer plus 0.2 s margin because clip boundaries
        # are intentionally snapped to tracking rows and may not be exact.
        fs = float(meta.get("sample_rate", 48000) or 48000)
        buf = float(meta.get("buffer_size", 2048) or 2048)
        tol = max(0.25, 2.0 * buf / fs)
        if clip_start > target - timedelta(seconds=max(0.0, clip_half_s - tol)):
            return False, "clip_starts_too_late_for_requested_half_width"
        if clip_end < target + timedelta(seconds=max(0.0, clip_half_s - tol)):
            return False, "clip_ends_too_early_for_requested_half_width"
    try:
        n_samples = int(float(meta.get("n_samples", 0)))
        n_rows = int(float(meta.get("n_tracking_rows", 0)))
        buf = int(float(meta.get("buffer_size", 0)))
        if n_samples and n_rows and buf and n_samples != n_rows * buf:
            return False, "sample_tracking_count_mismatch"
    except Exception:
        pass
    return True, "complete"


def inspect_existing_event_dir(event_dir: Path, units: Sequence[str], target: datetime, clip_half_s: float) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    rows: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for u in units:
        ok, reason = _event_unit_complete(event_dir / u, target, clip_half_s, require_full_window=True)
        rows[u] = {"unit": u, "ok": ok, "reason": reason, "unit_dir": str(event_dir / u)}
        all_ok = all_ok and ok
    return all_ok, rows


def _clear_unit_dir(event_dir: Path, unit: str) -> None:
    udir = event_dir / unit
    if udir.exists():
        shutil.rmtree(udir)
    udir.mkdir(parents=True, exist_ok=True)


def _write_acquire_manifest(
    path: Path,
    *,
    target: datetime,
    clip_half_s: float,
    units: Sequence[str],
    reused: bool,
    rotated: bool,
    complete_rows: Dict[str, Dict[str, Any]],
    inspect_rows: List[Dict[str, Any]],
    rotation_rows: List[Dict[str, Any]],
    fetch_rows: List[Dict[str, Any]],
    ok: bool,
    message: str,
) -> None:
    kv = {
        "ok": ok,
        "target_sbts": format_ts(target),
        "clip_half_s": clip_half_s,
        "units": ",".join(units),
        "reused": reused,
        "rotated": rotated,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "message": message,
    }
    tables = {
        "existing_clips": [flatten_dict(v) for v in complete_rows.values()],
        "remote_inspection": [flatten_dict(v) for v in inspect_rows],
        "rotation": [flatten_dict(v) for v in rotation_rows],
        "fetch": [flatten_dict(v) for v in fetch_rows],
    }
    write_sectioned_tables(path, kv, tables, header="ARU acquisition manifest TXT v1")


def ensure_event_clips(
    target_time: str | datetime,
    *,
    units: Optional[Sequence[str]] = None,
    clip_half_s: float = 30.0,
    output_root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
    allow_rotate_if_active: bool = True,
    reuse_existing: bool = True,
    force_refetch: bool = False,
    finalize_wait_s: float = 10.0,
    worker_timeout_s: float = 240.0,
    inspect_timeout_s: float = 20.0,
    rotate_workers: int = 4,
    clip_workers: int = 2,
) -> AcquisitionResult:
    """Ensure a timestamp has a complete fetched event directory.

    This is the function the dashboard should call before calibration or
    localization. It is safe to call repeatedly: when clips already exist and
    pass validation, it returns immediately without rotating or fetching.
    """
    cfg = config or load_config()
    target = target_time if isinstance(target_time, datetime) else parse_event_time(str(target_time))
    selected_units = list(units) if units is not None else list(cfg.get("units", {}).keys())
    unit_cfgs = units_from_config(cfg, selected_units)
    if not unit_cfgs:
        raise RuntimeError("No units selected")
    selected_units = [u.name for u in unit_cfgs]

    out_root = Path(output_root or cfg.get("clips", {}).get("output_root", str(BASE_DIR / "data" / "event_clips"))).expanduser()
    event_dir = event_run_dir(out_root, target)
    event_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = event_dir / "acquire_manifest.txt"

    all_complete, complete_rows = inspect_existing_event_dir(event_dir, selected_units, target, clip_half_s)
    if reuse_existing and all_complete and not force_refetch:
        msg = "Existing fetched clips are complete; reused without rotation/fetch."
        _write_acquire_manifest(
            manifest_path, target=target, clip_half_s=clip_half_s, units=selected_units,
            reused=True, rotated=False, complete_rows=complete_rows, inspect_rows=[],
            rotation_rows=[], fetch_rows=[], ok=True, message=msg,
        )
        return AcquisitionResult(True, str(event_dir), True, False, [], [], str(manifest_path), msg)

    missing_or_invalid = [u for u, row in complete_rows.items() if force_refetch or not row.get("ok")]

    inspect_rows: List[Dict[str, Any]] = []
    if missing_or_invalid:
        with cf.ThreadPoolExecutor(max_workers=min(len(unit_cfgs), max(1, clip_workers))) as ex:
            futs = {ex.submit(_run_remote_inspect, u, target, inspect_timeout_s): u for u in unit_cfgs if u.name in missing_or_invalid}
            for fut in cf.as_completed(futs):
                try: inspect_rows.append(fut.result())
                except Exception as e: inspect_rows.append({"ok": False, "unit": futs[fut].name, "status": "inspect_failed", "error": str(e)})

    active_units = [r.get("unit") for r in inspect_rows if r.get("status") == "active_contains_target"]
    rotated = False
    rotation_rows: List[Dict[str, Any]] = []
    if active_units:
        if not allow_rotate_if_active:
            msg = f"Target appears to be in active files on {active_units}; rotation disabled."
            _write_acquire_manifest(
                manifest_path, target=target, clip_half_s=clip_half_s, units=selected_units,
                reused=False, rotated=False, complete_rows=complete_rows, inspect_rows=inspect_rows,
                rotation_rows=[], fetch_rows=[], ok=False, message=msg,
            )
            return AcquisitionResult(False, str(event_dir), False, False, [], missing_or_invalid, str(manifest_path), msg)
        with cf.ThreadPoolExecutor(max_workers=rotate_workers) as ex:
            futs = {ex.submit(rotate_unit, u): u for u in unit_cfgs}
            for fut in cf.as_completed(futs):
                try: rotation_rows.append(fut.result())
                except Exception as e: rotation_rows.append({"unit": futs[fut].name, "ok": False, "error": str(e)})
        rotated = True
        time.sleep(float(finalize_wait_s))

    # Recompute existing status after potential rotation, then fetch missing.
    all_complete, complete_rows = inspect_existing_event_dir(event_dir, selected_units, target, clip_half_s)
    fetch_units = [u for u, row in complete_rows.items() if force_refetch or not row.get("ok")]
    for u in fetch_units:
        _clear_unit_dir(event_dir, u)

    fetch_rows: List[Dict[str, Any]] = []
    if fetch_units:
        unit_map = {u.name: u for u in unit_cfgs}
        with cf.ThreadPoolExecutor(max_workers=clip_workers) as ex:
            futs = {
                ex.submit(process_unit, unit_map[u], target, clip_half_s, event_dir, worker_timeout_s): u
                for u in fetch_units if u in unit_map
            }
            for fut in cf.as_completed(futs):
                try: res = fut.result()
                except Exception as e: res = {"unit": futs[fut], "ok": False, "error": str(e)}
                fetch_rows.append(res)

    all_complete_final, complete_rows_final = inspect_existing_event_dir(event_dir, selected_units, target, clip_half_s)
    missing_final = [u for u, row in complete_rows_final.items() if not row.get("ok")]

    msg = "Acquisition complete." if all_complete_final else f"Acquisition incomplete; missing/invalid units: {missing_final}"
    _write_acquire_manifest(
        manifest_path, target=target, clip_half_s=clip_half_s, units=selected_units,
        reused=False, rotated=rotated, complete_rows=complete_rows_final, inspect_rows=inspect_rows,
        rotation_rows=rotation_rows, fetch_rows=fetch_rows, ok=all_complete_final, message=msg,
    )
    if all_complete_final:
        (event_dir / ".complete").write_text(datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")
    return AcquisitionResult(
        bool(all_complete_final), str(event_dir), False, bool(rotated),
        [str(r.get("unit", "")) for r in fetch_rows if r.get("ok")], missing_final,
        str(manifest_path), msg,
    )


def _parse_units_arg(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    # Accept both comma-separated and shell-list-ish strings.
    out: List[str] = []
    for part in str(s).replace(",", " ").split():
        p = part.strip()
        if p:
            out.append(p)
    return out or None


def main() -> int:
    p = argparse.ArgumentParser(description="Idempotently ensure fetched ARU event clips exist for a timestamp.")
    p.add_argument("timestamp", help='Absolute target time, e.g. "2026-05-05 20:31:00" or "today 20:31:00"')
    p.add_argument("--clip-half-s", type=float, default=30.0)
    p.add_argument("--units", default=None, help="Comma- or space-separated unit list. Default: config units.")
    p.add_argument("--output-root", default=None)
    p.add_argument("--no-rotate-if-active", action="store_true", help="Fail instead of rotating if timestamp is in an active file.")
    p.add_argument("--no-reuse", action="store_true", help="Ignore reusable existing clips and fetch again if needed.")
    p.add_argument("--force-refetch", action="store_true", help="Delete/refetch selected unit clips even if they look complete.")
    p.add_argument("--finalize-wait-s", type=float, default=10.0)
    p.add_argument("--worker-timeout-s", type=float, default=240.0)
    p.add_argument("--inspect-timeout-s", type=float, default=45.0)
    p.add_argument("--clip-workers", type=int, default=2)
    p.add_argument("--rotate-workers", type=int, default=4)
    args = p.parse_args()

    res = ensure_event_clips(
        args.timestamp,
        units=_parse_units_arg(args.units),
        clip_half_s=args.clip_half_s,
        output_root=args.output_root,
        allow_rotate_if_active=not args.no_rotate_if_active,
        reuse_existing=not args.no_reuse,
        force_refetch=args.force_refetch,
        finalize_wait_s=args.finalize_wait_s,
        worker_timeout_s=args.worker_timeout_s,
        inspect_timeout_s=args.inspect_timeout_s,
        clip_workers=args.clip_workers,
        rotate_workers=args.rotate_workers,
    )
    print(f"event_dir = {res.event_dir}")
    print(f"ok = {res.ok}")
    print(f"reused = {res.reused}")
    print(f"rotated = {res.rotated}")
    print(f"fetched_units = {','.join(res.fetched_units)}")
    print(f"missing_units = {','.join(res.missing_units)}")
    print(f"manifest_path = {res.manifest_path}")
    print(f"message = {res.message}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
