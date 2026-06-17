#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from aru_io import write_kv_txt, flatten_dict

BASE_DIR = Path(__file__).resolve().parent
# Pipeline scripts may live either directly in ~/aru-dashboard or in
# ~/aru-dashboard/aru_pipeline. Prefer a config next to the script, then fall
# back to the dashboard root.
CONFIG_CANDIDATES = [BASE_DIR / "config.yaml", BASE_DIR.parent / "config.yaml"]
CONFIG_PATH = next((p for p in CONFIG_CANDIDATES if p.exists()), CONFIG_CANDIDATES[0])

# The remote worker still uses a JSON process protocol because it is a safe way
# to pass structured values through stdin/stdout over SSH. Its on-disk output is
# now TXT, and the local manifest/location outputs are TXT.
REMOTE_WORKER = r'''
from __future__ import annotations
import json, os, re, shutil, statistics, subprocess, sys, tempfile
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import Any

FULL_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?$")
FULL_TS_FIND_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?")
TIME_ONLY_FIND_RE = re.compile(r"(?<!\d)(\d{2})-(\d{2})-(\d{2})(?:\.(\d+))?(?!\d)")

def parse_ts(s: str) -> datetime:
    s = s.strip()
    if "." not in s: s = s + ".000000"
    base, frac = s.split(".", 1)
    frac = (frac + "000000")[:6]
    return datetime.strptime(base + "." + frac, "%Y-%m-%d_%H-%M-%S.%f")

def format_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H-%M-%S.%f")

def parse_time_only_to_dt(text: str, base_date: date) -> datetime | None:
    m = TIME_ONLY_FIND_RE.search(text)
    if not m:
        return None
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac = ((m.group(4) or "") + "000000")[:6]
    return datetime.combine(base_date, dtime(hh, mm, ss, int(frac)))

def ts_from_line(line: str, base_date: date | None = None):
    m = FULL_TS_FIND_RE.search(line)
    if m:
        try: return parse_ts(m.group(0))
        except Exception: return None
    if base_date is not None:
        try: return parse_time_only_to_dt(line, base_date)
        except Exception: return None
    return None

def split_sbts_name(path: Path):
    if path.suffix != ".flac": return None
    parts = path.stem.split("--")
    if len(parts) < 3: return None
    start_s, unit, end_s = parts[0], parts[1], parts[-1]
    if not FULL_TS_RE.match(start_s) or not FULL_TS_RE.match(end_s): return None
    try: return parse_ts(start_s), unit, parse_ts(end_s)
    except Exception: return None

def find_flacs_for_day(target: datetime):
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

def read_tracking_times(path: Path, base_dt: datetime | None = None):
    times = []
    base_date = (base_dt.date() if base_dt is not None else None)
    current_date = base_date
    prev = None
    with path.open("r", errors="replace") as f:
        for line in f:
            ts = ts_from_line(line, current_date)
            if ts is None:
                continue
            if prev is not None and ts < prev - timedelta(hours=12):
                ts = ts + timedelta(days=1)
                if current_date is not None:
                    current_date = current_date + timedelta(days=1)
            times.append(ts)
            prev = ts
    return times

def tracking_range_contains(path: Path, target: datetime, base_dt: datetime | None = None) -> bool:
    try: times = read_tracking_times(path, base_dt)
    except Exception: return False
    return bool(times) and min(times) <= target <= max(times)

def find_source_file(target: datetime):
    candidates = []
    for flac in find_flacs_for_day(target):
        parsed = split_sbts_name(flac)
        if parsed is None: continue
        start_dt, unit_name, end_dt = parsed
        tracking = flac.with_suffix(".tracking")
        if start_dt <= target <= end_dt and tracking.exists():
            candidates.append((start_dt, end_dt, unit_name, flac, tracking, "filename_interval"))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        start_dt, end_dt, unit_name, flac, tracking, method = candidates[0]
        return {"source_flac": str(flac), "source_tracking": str(tracking), "source_start": format_ts(start_dt), "source_end": format_ts(end_dt), "source_unit": unit_name, "match_method": method}
    matches = []
    for flac in find_flacs_for_day(target):
        parsed = split_sbts_name(flac)
        if parsed is None: continue
        start_dt, unit_name, end_dt = parsed
        tracking = flac.with_suffix(".tracking")
        if tracking.exists() and tracking_range_contains(tracking, target, start_dt):
            matches.append((start_dt, end_dt, unit_name, flac, tracking, "tracking_range"))
    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        start_dt, end_dt, unit_name, flac, tracking, method = matches[0]
        return {"source_flac": str(flac), "source_tracking": str(tracking), "source_start": format_ts(start_dt), "source_end": format_ts(end_dt), "source_unit": unit_name, "match_method": method}
    raise RuntimeError(f"No finalized FLAC+tracking file found containing {format_ts(target)}")

def get_sample_rate(flac: Path) -> int:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "default=nk=1:nw=1", str(flac)]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    if not out: raise RuntimeError(f"ffprobe could not determine sample rate for {flac}")
    return int(float(out[0]))

def infer_buffer_size(times, sample_rate):
    if len(times) < 3: raise RuntimeError("Need at least 3 tracking timestamps to infer buffer size")
    deltas = []
    last = times[0]
    for t in times[1:200]:
        dt = (t-last).total_seconds(); last = t
        if 0 < dt < 1: deltas.append(dt)
    if not deltas: raise RuntimeError("Could not infer tracking cadence")
    return int(round(statistics.median(deltas) * sample_rate))

def choose_clip_indices(times, target, half_s):
    start_target, end_target = target-timedelta(seconds=half_s), target+timedelta(seconds=half_s)
    start_i = next((i for i,t in enumerate(times) if t >= start_target), max(0, len(times)-1))
    end_i = next((i for i,t in enumerate(times) if t >= end_target), len(times))
    if end_i <= start_i: end_i = min(len(times), start_i+1)
    return start_i, end_i

def write_clipped_tracking(src, dst, start_i, end_i, times):
    n = 0
    with src.open("r", errors="replace") as f, dst.open("w") as g:
        for i, line in enumerate(f):
            if i < start_i: continue
            if i >= end_i: break
            parts = line.strip().split()
            ts = times[i] if i < len(times) else None
            if ts is not None:
                rest = " ".join(parts[2:]) if len(parts) >= 2 else ""
                g.write(f"{n} {format_ts(ts)}" + (f" {rest}" if rest else "") + "\n")
            else:
                g.write(line)
            n += 1
    return n

def run_ffmpeg_clip(src_flac, dst_flac, start_sample, end_sample):
    filter_arg = f"atrim=start_sample={start_sample}:end_sample={end_sample},asetpts=PTS-STARTPTS"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src_flac), "-map", "0:a:0", "-af", filter_arg, "-c:a", "flac", str(dst_flac)]
    if shutil.which("nice"): cmd = ["nice", "-n", "10"] + cmd
    if shutil.which("ionice"): cmd = ["ionice", "-c3"] + cmd
    subprocess.check_call(cmd)

def write_kv_txt(path, d):
    def emit(prefix, obj, lines):
        if isinstance(obj, dict):
            for k,v in obj.items(): emit(f"{prefix}.{k}" if prefix else str(k), v, lines)
        else:
            lines.append(f"{prefix} = {obj}")
    lines = ["# ARU clip metadata TXT v1"]
    emit("", d, lines)
    path.write_text("\n".join(lines)+"\n")

def main():
    args = json.loads(sys.stdin.read())
    target = parse_ts(args["target_sbts"])
    half_s = float(args.get("clip_half_s", 30.0))
    source = find_source_file(target)
    source_flac, source_tracking = Path(source["source_flac"]), Path(source["source_tracking"])
    source_start_dt = parse_ts(source["source_start"])
    unit_name = source["source_unit"]
    sample_rate = get_sample_rate(source_flac)
    times = read_tracking_times(source_tracking, source_start_dt)
    if not times: raise RuntimeError(f"No valid timestamps found in {source_tracking}")
    buffer_size = infer_buffer_size(times, sample_rate)
    start_i, end_i = choose_clip_indices(times, target, half_s)
    start_sample, end_sample = start_i*buffer_size, end_i*buffer_size
    clip_start_dt = times[start_i]
    clip_end_dt = times[end_i] if end_i < len(times) else times[end_i-1] + timedelta(seconds=buffer_size/sample_rate)
    tmpdir = Path(tempfile.mkdtemp(prefix="aru_clip_"))
    clip_base = f"{format_ts(clip_start_dt)}--{unit_name}--{format_ts(clip_end_dt)}"
    clip_flac = tmpdir / f"{clip_base}.flac"
    clip_tracking = tmpdir / f"{clip_base}.tracking"
    clip_metadata = tmpdir / f"{clip_base}.clip_metadata.txt"
    run_ffmpeg_clip(source_flac, clip_flac, start_sample, end_sample)
    n_tracking_rows = write_clipped_tracking(source_tracking, clip_tracking, start_i, end_i, times)
    metadata = {"target_sbts": format_ts(target), "clip_half_s_requested": half_s, "source": source, "sample_rate": sample_rate, "buffer_size": buffer_size, "tracking_start_index": start_i, "tracking_end_index_exclusive": end_i, "start_sample": start_sample, "end_sample": end_sample, "n_samples": end_sample-start_sample, "n_tracking_rows": n_tracking_rows, "clip_start": format_ts(clip_start_dt), "clip_end": format_ts(clip_end_dt), "clip_duration_s": (end_sample-start_sample)/sample_rate, "clip_flac": str(clip_flac), "clip_tracking": str(clip_tracking), "clip_metadata": str(clip_metadata), "alignment_note": "Audio was clipped by exact sample indices on tracking-buffer boundaries. Clipped tracking rows were cut over the same buffer interval and renumbered from zero. Tracking timestamps are written as full SBTS datetimes, even if the source tracking file used time-only rows."}
    write_kv_txt(clip_metadata, metadata)
    print(json.dumps({"ok": True, "clip_flac": str(clip_flac), "clip_tracking": str(clip_tracking), "clip_metadata": str(clip_metadata), "metadata": metadata}))
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)})); raise SystemExit(1)
'''


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def parse_event_time(raw: str) -> datetime:
    raw = " ".join(raw.strip().split())
    now = datetime.now(); lowered = raw.lower(); base_date = now.date()
    if lowered.startswith("today "):
        raw = raw[6:].strip(); base_date = now.date()
    elif lowered.startswith("yesterday "):
        raw = raw[10:].strip(); base_date = (now - timedelta(days=1)).date()
    full_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M"]
    time_formats = ["%H:%M:%S", "%H:%M", "%I:%M:%S%p", "%I:%M%p", "%I:%M:%S %p", "%I:%M %p", "%I%p", "%I %p"]
    for fmt in full_formats:
        try: return datetime.strptime(raw, fmt)
        except ValueError: pass
    for text in [raw.lower().replace(" ", ""), raw]:
        for fmt in time_formats:
            try: return datetime.combine(base_date, datetime.strptime(text, fmt).time())
            except ValueError: pass
    raise ValueError(f"Could not parse event time: {raw!r}")


def format_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H-%M-%S.%f")


def event_run_dir(root: Path, target: datetime) -> Path:
    return root / target.strftime("%Y") / target.strftime("%Y-%m") / target.strftime("%Y-%m-%d") / f"event_{target.strftime('%Y-%m-%d_%H-%M-%S')}"


@dataclass
class UnitConfig:
    name: str
    host: str
    user: str
    transport: str


def run_worker(unit: UnitConfig, input_json: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    remote_cmd = "python3 -c " + shlex.quote(REMOTE_WORKER)
    if unit.transport == "local":
        cmd = ["python3", "-c", REMOTE_WORKER]
    else:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", f"{unit.user}@{unit.host}", remote_cmd]
    proc = subprocess.run(cmd, input=json.dumps(input_json), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip() or proc.stdout.strip() or f"return code {proc.returncode}", "stdout": proc.stdout, "stderr": proc.stderr}
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


def rotate_unit(unit: UnitConfig, timeout_s: float = 10.0) -> dict[str, Any]:
    rotate_cmd = """
set -e
if sudo -n systemctl kill --kill-who=main -s HUP sbts-aru.service 2>/dev/null; then echo rotated_by=systemctl
elif sudo -n pkill -HUP -x sbts-aru 2>/dev/null; then echo rotated_by=sudo_pkill
elif pkill -HUP -x sbts-aru 2>/dev/null; then echo rotated_by=pkill
else echo rotated_by=failed; exit 1
fi
"""
    cmd = ["bash", "-lc", rotate_cmd] if unit.transport == "local" else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{unit.user}@{unit.host}", "bash -lc " + shlex.quote(rotate_cmd)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    return {"unit": unit.name, "ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def copy_remote_file(unit: UnitConfig, remote_path: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    if unit.transport == "local":
        src = Path(remote_path); dst = local_dir / src.name; shutil.copy2(src, dst); return dst
    subprocess.check_call(["scp", "-p", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{unit.user}@{unit.host}:{remote_path}", str(local_dir) + "/"])
    return local_dir / Path(remote_path).name


def cleanup_remote_files(unit: UnitConfig, paths: list[str]) -> None:
    if not paths: return
    quoted = " ".join(shlex.quote(p) for p in paths)
    cmd_str = f"rm -f {quoted}; rmdir {shlex.quote(str(Path(paths[0]).parent))} 2>/dev/null || true"
    if unit.transport == "local": subprocess.run(["bash", "-lc", cmd_str], check=False)
    else: subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{unit.user}@{unit.host}", "bash -lc " + shlex.quote(cmd_str)], check=False)


def load_location_summary(unit_name: str) -> dict[str, Any] | None:
    # Existing source may still be JSON; copied event output is TXT.
    # If scripts live in aru_pipeline/, summaries usually live one level up.
    search_dirs = [BASE_DIR / "data" / "location_summaries", BASE_DIR.parent / "data" / "location_summaries"]
    paths = []
    for d in search_dirs:
        paths.extend([d / f"{unit_name}.txt", d / f"{unit_name}.json"])
    for path in paths:
        if path.exists():
            try:
                if path.suffix == ".json": return json.loads(path.read_text())
                d = {}
                for line in path.read_text().splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1); d[k.strip()] = v.strip()
                return d
            except Exception:
                return None
    return None


def process_unit(unit: UnitConfig, target: datetime, clip_half_s: float, dest_root: Path, worker_timeout_s: float) -> dict[str, Any]:
    unit_dir = dest_root / unit.name; unit_dir.mkdir(parents=True, exist_ok=True)
    result = run_worker(unit, {"target_sbts": format_ts(target), "clip_half_s": clip_half_s}, worker_timeout_s)
    if not result.get("ok"):
        return {"unit": unit.name, "ok": False, "error": result.get("error", "unknown worker error")}
    copied = {}
    paths = [result["clip_flac"], result["clip_tracking"], result["clip_metadata"]]
    try:
        copied["flac"] = str(copy_remote_file(unit, paths[0], unit_dir))
        copied["tracking"] = str(copy_remote_file(unit, paths[1], unit_dir))
        copied["clip_metadata"] = str(copy_remote_file(unit, paths[2], unit_dir))
    finally:
        cleanup_remote_files(unit, paths)
    location_summary = load_location_summary(unit.name)
    if location_summary is not None:
        loc_path = unit_dir / "location_summary.txt"
        write_kv_txt(loc_path, flatten_dict(location_summary), header="ARU location summary TXT v1")
        copied["location_summary"] = str(loc_path)
    return {"unit": unit.name, "ok": True, "files": copied, "metadata": result.get("metadata", {})}


def units_from_config(config: dict[str, Any], only: list[str] | None = None) -> list[UnitConfig]:
    out = []
    for name, cfg in config["units"].items():
        if only and name not in only: continue
        out.append(UnitConfig(name=name, host=cfg.get("host", name), user=cfg.get("user", "shishir"), transport=cfg.get("transport", "ssh")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate SBTS units, make clipped FLAC/tracking files remotely, and fetch clips with TXT metadata/manifest.")
    parser.add_argument("event_time", help='Absolute event time, e.g. "2026-05-05 20:31:00"')
    parser.add_argument("--clip-half-s", type=float, default=30.0)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--finalize-wait-s", type=float, default=10.0)
    parser.add_argument("--units", nargs="*", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--worker-timeout-s", type=float, default=240.0)
    parser.add_argument("--rotate-workers", type=int, default=4)
    parser.add_argument("--clip-workers", type=int, default=2)
    args = parser.parse_args()
    config = load_config(); target = parse_event_time(args.event_time); target_sbts = format_ts(target)
    output_root = Path(args.output_root or config.get("clips", {}).get("output_root", str(BASE_DIR / "data" / "event_clips"))).expanduser()
    dest_root = event_run_dir(output_root, target); dest_root.mkdir(parents=True, exist_ok=True)
    units = units_from_config(config, args.units)
    if not units: raise SystemExit("No units selected.")
    manifest: dict[str, Any] = {"ok": None, "event_time_input": args.event_time, "target_sbts": target_sbts, "clip_half_s": args.clip_half_s, "created_at": datetime.now().isoformat(), "output_dir": str(dest_root), "notes": "Clips are stored separately from raw SBTS recordings; audio/tracking clipped on tracking-buffer boundaries."}
    rotation_rows = []; unit_rows = []
    print(f"Target event time: {target_sbts}\nOutput directory: {dest_root}\nUnits: {' '.join(u.name for u in units)}")
    if not args.no_rotate:
        print("\nRotating units...")
        with cf.ThreadPoolExecutor(max_workers=args.rotate_workers) as ex:
            futs = {ex.submit(rotate_unit, u): u for u in units}
            for fut in cf.as_completed(futs):
                u = futs[fut]
                try: res = fut.result()
                except Exception as e: res = {"unit": u.name, "ok": False, "error": str(e)}
                rotation_rows.append(res); print(f"  {u.name}: {'ok' if res.get('ok') else 'check'}")
        print(f"\nWaiting {args.finalize_wait_s:.1f} s for SBTS finalization..."); time.sleep(args.finalize_wait_s)
    else:
        print("\nSkipping rotation (--no-rotate).")
    print("\nCreating remote clips and fetching them...")
    with cf.ThreadPoolExecutor(max_workers=args.clip_workers) as ex:
        futs = {ex.submit(process_unit, u, target, args.clip_half_s, dest_root, args.worker_timeout_s): u for u in units}
        for fut in cf.as_completed(futs):
            u = futs[fut]
            try: res = fut.result()
            except Exception as e: res = {"unit": u.name, "ok": False, "error": str(e)}
            flat = flatten_dict(res)
            unit_rows.append(flat)
            print(f"  {u.name}: {'ok' if res.get('ok') else 'FAILED - ' + str(res.get('error'))}")
    manifest["ok"] = all(str(r.get("ok", "False")) == "True" or r.get("ok") is True for r in unit_rows)
    manifest_path = dest_root / "run_manifest.txt"
    write_kv_txt(manifest_path, manifest, header="ARU run manifest TXT v1")
    # Append rotation/unit tables to manifest.
    with manifest_path.open("a") as f:
        for section, rows in [("rotation", rotation_rows), ("units", unit_rows)]:
            f.write(f"\n[{section}]\n")
            if rows:
                keys = []
                for row in rows:
                    for k in row.keys():
                        if k not in keys: keys.append(k)
                f.write("\t".join(keys) + "\n")
                for row in rows:
                    f.write("\t".join(str(row.get(k, "")) for k in keys) + "\n")
    print(f"\nWrote manifest: {manifest_path}")
    return 0 if manifest["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
