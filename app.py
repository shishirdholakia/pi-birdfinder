#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import shlex
import statistics
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
DATA_DIR = BASE_DIR / "data"
FIX_DIR = DATA_DIR / "location_fixes"
SUMMARY_DIR = DATA_DIR / "location_summaries"
LOG_DIR = DATA_DIR / "logs"
JOB_DIR = DATA_DIR / "jobs"
EVENT_CLIPS_DIR = DATA_DIR / "event_clips"
CALIBRATION_DIR = DATA_DIR / "calibrations"
LOCALIZATION_DIR = DATA_DIR / "localizations"
LOCATION_OVERRIDE_DIR = DATA_DIR / "location_overrides"
MAP_LAYER_DIR = DATA_DIR / "map_layers"

for d in [DATA_DIR, FIX_DIR, SUMMARY_DIR, LOG_DIR, JOB_DIR, EVENT_CLIPS_DIR, CALIBRATION_DIR, LOCALIZATION_DIR, LOCATION_OVERRIDE_DIR, MAP_LAYER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REMOTE_STATUS_COMMAND = r"""
echo "__TIME__"
date --iso-8601=seconds 2>/dev/null || date

echo "__HOSTNAME__"
hostname 2>/dev/null || true

echo "__SBTS__"
systemctl is-active sbts-aru.service 2>/dev/null || true

echo "__JACKD__"
if systemctl is-active --quiet jackd.service 2>/dev/null; then
  echo active
elif systemctl is-active --quiet jack.service 2>/dev/null; then
  echo active
elif pgrep -x jackd >/dev/null 2>&1; then
  echo active
else
  echo inactive
fi

echo "__CHRONY_SOURCES__"
chronyc -n sources 2>/dev/null || true

echo "__CHRONY_TRACKING__"
chronyc tracking 2>/dev/null || true

echo "__GPSD_JSON__"
timeout 2 gpspipe -w -n 20 2>/dev/null || true

echo "__DISK__"
df -P /home/pi/disk 2>/dev/null || df -P /disk 2>/dev/null || df -P / 2>/dev/null || true

echo "__WIFI__"
iw dev wlan0 link 2>/dev/null || true
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CONFIG = load_config()

def _resolve_dashboard_path(value: Any, default: Path) -> Path:
    """Resolve config paths relative to the dashboard root unless absolute."""
    if value in (None, ""):
        return default
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

TILE_CACHE_DIR = _resolve_dashboard_path(
    CONFIG.get("map", {}).get("tile_cache_dir"),
    DATA_DIR / "tile_cache" / "esri_world_imagery",
)


STATE: dict[str, Any] = {
    "started_at": utc_now_iso(),
    "last_update": None,
    "units": {},
    "logs": [],
    "jobs": {},
    "location": {
        "state": "idle",
        "session_id": None,
        "started_at": None,
        "frozen_at": None,
        "fixes": {},
        "summaries": {},
    },
}


def add_log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    STATE["logs"].append(line)
    STATE["logs"] = STATE["logs"][-200:]
    with open(LOG_DIR / "dashboard.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def split_sections(stdout: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in stdout.splitlines():
        m = re.match(r"^__([A-Z0-9_]+)__$", line.strip())
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def parse_chrony_sources(text: str, pps_names: list[str]) -> dict[str, Any]:
    pps_selected = False
    selected_source = None
    pps_source = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("=") or line.startswith("MS "):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        marker, name = parts[0], parts[1]
        if "*" in marker:
            selected_source = name
        if marker == "#*" and name in pps_names:
            pps_selected = True
            pps_source = name
    return {"pps_selected": pps_selected, "pps_source": pps_source, "selected_source": selected_source}


def parse_chrony_tracking(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"chrony_tracking_ok": False, "leap_status": None, "system_time": None, "root_dispersion": None}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        key_l = key.lower()
        if key_l == "leap status":
            out["leap_status"] = val
            out["chrony_tracking_ok"] = val.lower() == "normal"
        elif key_l == "system time":
            out["system_time"] = val
        elif key_l == "root dispersion":
            out["root_dispersion"] = val
    return out


def parse_gpsd_json(text: str) -> dict[str, Any]:
    latest_tpv: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("class") == "TPV":
            latest_tpv = obj
    if not latest_tpv:
        return {"gps_has_fix": False, "gps_mode": 0, "gps": None}
    mode = int(latest_tpv.get("mode") or 0)
    lat = latest_tpv.get("lat")
    lon = latest_tpv.get("lon")
    gps = {
        "time": latest_tpv.get("time"),
        "mode": mode,
        "lat": lat,
        "lon": lon,
        "alt_m": latest_tpv.get("alt"),
        "eph_m": latest_tpv.get("eph"),
        "epv_m": latest_tpv.get("epv"),
        "speed_m_s": latest_tpv.get("speed"),
    }
    return {"gps_has_fix": mode >= 2 and lat is not None and lon is not None, "gps_mode": mode, "gps": gps}


def parse_disk(text: str) -> dict[str, Any]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {"disk_free_gb": None, "disk_mount": None}
    parts = lines[-1].split()
    if len(parts) < 6:
        return {"disk_free_gb": None, "disk_mount": None}
    try:
        free_gb = float(parts[3]) * 1024 / 1e9
    except ValueError:
        free_gb = None
    return {"disk_free_gb": round(free_gb, 2) if free_gb is not None else None, "disk_mount": parts[-1]}


def parse_wifi(text: str) -> dict[str, Any]:
    connected = "Connected to" in text
    signal_dbm = None
    m = re.search(r"signal:\s*(-?\d+)", text)
    if m:
        signal_dbm = int(m.group(1))
    return {"wifi_connected": connected, "wifi_signal_dbm": signal_dbm}


def compute_health(status: dict[str, Any], summary: dict[str, Any] | None) -> str:
    if not status.get("reachable"):
        return "unreachable"
    bad, warn = [], []
    if not status.get("sbts_running"):
        bad.append("sbts")
    if not status.get("jackd_running"):
        bad.append("jackd")
    if not status.get("gps_has_fix"):
        warn.append("gps")
    if not status.get("pps_selected"):
        warn.append("pps")
    if summary:
        if summary.get("quality") == "bad":
            warn.append("location")
    else:
        warn.append("location")
    if bad:
        return "bad"
    if warn:
        return "warn"
    return "good"


async def run_command_for_unit(unit: str, cfg: dict[str, Any]) -> tuple[int, str, str]:
    transport = cfg.get("transport", "ssh")
    timeout_s = float(CONFIG.get("polling", {}).get("command_timeout_s", 8))
    if transport == "local":
        cmd = ["bash", "-lc", REMOTE_STATUS_COMMAND]
    else:
        user = cfg.get("user", "shishir")
        host = cfg["host"]
        remote_cmd = f"bash -lc {shlex.quote(REMOTE_STATUS_COMMAND)}"
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=1", f"{user}@{host}", remote_cmd]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return 124, "", "command timed out"
    return proc.returncode or 0, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")



GPSD_RESTART_COMMAND = r"""
set -u
if command -v systemctl >/dev/null 2>&1; then
  if sudo -n systemctl restart gpsd.socket gpsd.service 2>/tmp/aru_gpsd_restart.err; then
    echo "restart_method=sudo_systemctl_socket_and_service"
  elif sudo -n systemctl restart gpsd.service 2>>/tmp/aru_gpsd_restart.err; then
    echo "restart_method=sudo_systemctl_service"
  elif sudo -n systemctl restart gpsd.socket 2>>/tmp/aru_gpsd_restart.err; then
    echo "restart_method=sudo_systemctl_socket"
  elif systemctl restart gpsd.socket gpsd.service 2>>/tmp/aru_gpsd_restart.err; then
    echo "restart_method=systemctl_socket_and_service"
  elif systemctl restart gpsd.service 2>>/tmp/aru_gpsd_restart.err; then
    echo "restart_method=systemctl_service"
  else
    echo "restart_method=failed"
    echo "stderr=$(cat /tmp/aru_gpsd_restart.err 2>/dev/null | tail -5 | tr '\n' ' ')"
    exit 1
  fi
else
  if sudo -n service gpsd restart 2>/tmp/aru_gpsd_restart.err; then
    echo "restart_method=sudo_service_gpsd"
  elif service gpsd restart 2>>/tmp/aru_gpsd_restart.err; then
    echo "restart_method=service_gpsd"
  else
    echo "restart_method=failed"
    echo "stderr=$(cat /tmp/aru_gpsd_restart.err 2>/dev/null | tail -5 | tr '\n' ' ')"
    exit 1
  fi
fi
sleep 1
echo "gpsd_socket=$(systemctl is-active gpsd.socket 2>/dev/null || true)"
echo "gpsd_service=$(systemctl is-active gpsd.service 2>/dev/null || true)"
"""


async def run_shell_for_unit(unit: str, cfg: dict[str, Any], command: str, timeout_s: float = 20.0) -> tuple[int, str, str]:
    """Run an arbitrary shell command on a configured unit via local shell or SSH."""
    transport = cfg.get("transport", "ssh")
    if transport == "local":
        cmd = ["bash", "-lc", command]
    else:
        user = cfg.get("user", "shishir")
        host = cfg["host"]
        remote_cmd = f"bash -lc {shlex.quote(command)}"
        cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=6",
            "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=1",
            f"{user}@{host}",
            remote_cmd,
        ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return 124, "", "command timed out"
    return proc.returncode or 0, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")


async def restart_gpsd_for_unit(unit: str, cfg: dict[str, Any]) -> dict[str, Any]:
    rc, stdout, stderr = await run_shell_for_unit(unit, cfg, GPSD_RESTART_COMMAND, timeout_s=25.0)
    return {
        "unit": unit,
        "ok": rc == 0,
        "returncode": rc,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }


async def poll_unit(unit: str, cfg: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    rc, stdout, stderr = await run_command_for_unit(unit, cfg)
    elapsed_ms = int((time.time() - started) * 1000)
    base: dict[str, Any] = {
        "unit": unit,
        "reachable": rc == 0,
        "last_seen": utc_now_iso() if rc == 0 else None,
        "poll_elapsed_ms": elapsed_ms,
        "error": stderr.strip()[-500:] if stderr.strip() else None,
        "raw_sections": {},
    }
    if rc != 0:
        base.update({
            "hostname": None, "sbts_running": False, "jackd_running": False,
            "gps_has_fix": False, "gps_mode": 0, "gps": None,
            "pps_selected": False, "pps_source": None, "selected_source": None,
            "chrony_tracking_ok": False, "disk_free_gb": None, "disk_mount": None,
            "wifi_connected": False, "wifi_signal_dbm": None,
        })
        return base
    sections = split_sections(stdout)
    base["raw_sections"] = sections
    pps_names = cfg.get("pps_names", ["PPS", "PPS0", "GPS0", "SHM1"])
    chrony = parse_chrony_sources(sections.get("CHRONY_SOURCES", ""), pps_names)
    tracking = parse_chrony_tracking(sections.get("CHRONY_TRACKING", ""))
    gps = parse_gpsd_json(sections.get("GPSD_JSON", ""))
    disk = parse_disk(sections.get("DISK", ""))
    wifi = parse_wifi(sections.get("WIFI", ""))
    sbts_text = sections.get("SBTS", "").strip()
    jack_text = sections.get("JACKD", "").strip()
    base.update({
        "hostname": sections.get("HOSTNAME", "").strip() or None,
        "remote_time": sections.get("TIME", "").strip() or None,
        "sbts_running": sbts_text == "active",
        "sbts_state": sbts_text or "unknown",
        "jackd_running": jack_text == "active",
        "jackd_state": jack_text or "unknown",
    })
    base.update(chrony); base.update(tracking); base.update(gps); base.update(disk); base.update(wifi)
    return base


def gps_fix_is_valid(status: dict[str, Any]) -> bool:
    gps = status.get("gps")
    if not gps:
        return False
    loc_cfg = CONFIG.get("location", {})
    max_eph_m = float(loc_cfg.get("max_eph_m", 10.0))
    if gps.get("mode") != 3:
        return False
    if gps.get("lat") is None or gps.get("lon") is None:
        return False
    try:
        lat_f = float(gps.get("lat")); lon_f = float(gps.get("lon"))
        if abs(lat_f) < 1e-9 and abs(lon_f) < 1e-9:
            return False
    except Exception:
        return False
    eph = gps.get("eph_m")
    if eph is not None:
        try:
            if float(eph) > max_eph_m:
                return False
        except Exception:
            return False
    speed = gps.get("speed_m_s")
    if speed is not None:
        try:
            if float(speed) > 0.5:
                return False
        except Exception:
            pass
    return True


def append_location_fix(unit: str, status: dict[str, Any]) -> None:
    gps = status.get("gps")
    if not gps:
        return
    session_id = STATE["location"]["session_id"]
    fix = {
        "session_id": session_id, "unit": unit, "received_time": utc_now_iso(),
        "gps_time": gps.get("time"), "mode": gps.get("mode"),
        "lat": gps.get("lat"), "lon": gps.get("lon"), "alt_m": gps.get("alt_m"),
        "eph_m": gps.get("eph_m"), "epv_m": gps.get("epv_m"), "speed_m_s": gps.get("speed_m_s"),
    }
    STATE["location"]["fixes"].setdefault(unit, []).append(fix)
    STATE["location"]["fixes"][unit] = STATE["location"]["fixes"][unit][-5000:]
    path = FIX_DIR / f"{session_id}_{unit}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(fix) + "\n")


def weighted_mean(values: list[float], weights: list[float]) -> float:
    s = sum(weights)
    if s <= 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / s


def summarize_location(unit: str) -> dict[str, Any]:
    fixes = STATE["location"]["fixes"].get(unit, [])
    if not fixes:
        return {"unit": unit, "state": STATE["location"]["state"], "n_fixes": 0, "quality": "bad"}
    loc_cfg = CONFIG.get("location", {})
    min_good = int(loc_cfg.get("min_fixes_for_good", 50))
    floor_m = float(loc_cfg.get("weight_floor_m", 2.0))
    lats = [float(f["lat"]) for f in fixes]
    lons = [float(f["lon"]) for f in fixes]
    alts = [float(f["alt_m"]) for f in fixes if f.get("alt_m") is not None]
    weights, ephs = [], []
    for f in fixes:
        eph = f.get("eph_m")
        try:
            eph_f = float(eph) if eph is not None else floor_m
        except Exception:
            eph_f = floor_m
        eph_f = max(eph_f, floor_m)
        weights.append(1.0 / (eph_f * eph_f))
        ephs.append(eph_f)
    lat_mean = weighted_mean(lats, weights)
    lon_mean = weighted_mean(lons, weights)
    alt_mean = weighted_mean(alts, weights[:len(alts)]) if alts else None
    r_earth = 6371000.0
    lat0 = math.radians(lat_mean)
    distances = []
    for lat, lon in zip(lats, lons):
        north = math.radians(lat - lat_mean) * r_earth
        east = math.radians(lon - lon_mean) * r_earth * math.cos(lat0)
        distances.append(math.hypot(east, north))
    scatter_m = statistics.median(distances) if distances else None
    median_eph = statistics.median(ephs) if ephs else None
    if len(fixes) >= min_good and (median_eph is None or median_eph <= 5.0):
        quality = "good"
    elif len(fixes) >= 10:
        quality = "warn"
    else:
        quality = "bad"
    return {
        "unit": unit,
        "state": STATE["location"]["state"],
        "session_id": STATE["location"].get("session_id"),
        "started_at": STATE["location"].get("started_at"),
        "frozen_at": STATE["location"].get("frozen_at"),
        "n_fixes": len(fixes),
        "lat_deg": lat_mean,
        "lon_deg": lon_mean,
        "alt_m": alt_mean,
        "median_eph_m": median_eph,
        "scatter_horizontal_m": scatter_m,
        "quality": quality,
    }


def update_location_summaries() -> None:
    summaries = {unit: summarize_location(unit) for unit in CONFIG.get("units", {}).keys()}
    STATE["location"]["summaries"] = summaries


def _write_location_summary_txt(unit: str, summary: dict[str, Any]) -> None:
    path = SUMMARY_DIR / f"{unit}.txt"
    lat = summary.get("lat_deg"); lon = summary.get("lon_deg")
    sigma = summary.get("scatter_horizontal_m") if summary.get("scatter_horizontal_m") is not None else summary.get("median_eph_m")
    if sigma is None:
        sigma = 10.0
    lines = [
        "# ARU location summary TXT v1",
        "# Generated by aru-dashboard location averaging",
        f"unit = {unit}",
        f"state = {summary.get('state', '')}",
        f"session_id = {summary.get('session_id', '')}",
        f"started_at = {summary.get('started_at', '')}",
        f"frozen_at = {summary.get('frozen_at', '')}",
        f"quality = {summary.get('quality', '')}",
        f"n_fixes = {summary.get('n_fixes', 0)}",
    ]
    if lat is not None and lon is not None:
        lines.extend([f"lat = {float(lat):.12f}", f"lon = {float(lon):.12f}", f"latitude = {float(lat):.12f}", f"longitude = {float(lon):.12f}"])
    if summary.get("alt_m") is not None:
        lines.append(f"alt_m = {float(summary['alt_m']):.4f}")
    if summary.get("median_eph_m") is not None:
        lines.append(f"median_eph_m = {float(summary['median_eph_m']):.4f}")
    if summary.get("scatter_horizontal_m") is not None:
        lines.append(f"scatter_horizontal_m = {float(summary['scatter_horizontal_m']):.4f}")
    lines.append(f"position_sigma_m = {float(sigma):.4f}")
    lines.append("source = aru_dashboard_location_averaging")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_location_summaries() -> None:
    update_location_summaries()
    for unit, summary in STATE["location"]["summaries"].items():
        with open(SUMMARY_DIR / f"{unit}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        _write_location_summary_txt(unit, summary)
    with open(SUMMARY_DIR / "all_units.json", "w", encoding="utf-8") as f:
        json.dump(STATE["location"]["summaries"], f, indent=2)
    with open(SUMMARY_DIR / "all_units.txt", "w", encoding="utf-8") as f:
        f.write("# ARU all-unit location summaries TXT v1\n")
        for unit, summary in STATE["location"]["summaries"].items():
            f.write(f"\n[{unit}]\n")
            for k, v in summary.items():
                f.write(f"{k} = {v}\n")


async def poll_loop() -> None:
    interval_s = float(CONFIG.get("polling", {}).get("interval_s", 5))
    add_log("poller started")
    while True:
        try:
            tasks = [poll_unit(unit, cfg) for unit, cfg in CONFIG.get("units", {}).items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for unit, result in zip(CONFIG.get("units", {}).keys(), results):
                if isinstance(result, Exception):
                    status = {"unit": unit, "reachable": False, "error": str(result), "last_seen": None}
                else:
                    status = result
                if STATE["location"]["state"] == "collecting" and gps_fix_is_valid(status):
                    append_location_fix(unit, status)
                STATE["units"][unit] = status
            update_location_summaries()
            for unit, status in STATE["units"].items():
                summary = STATE["location"]["summaries"].get(unit)
                status["health"] = compute_health(status, summary)
            STATE["last_update"] = utc_now_iso()
        except Exception as e:
            add_log(f"poll loop error: {e}")
        await asyncio.sleep(interval_s)


# -----------------------------------------------------------------------------
# Pipeline job system: acquire -> calibrate/localize -> display artifacts
# -----------------------------------------------------------------------------

ARTIFACT_EXTS = {".html", ".htm", ".png", ".jpg", ".jpeg", ".txt", ".csv", ".h5", ".npz", ".log", ".json", ".geojson"}


def _cfg_path(section: str, key: str, default: Path) -> Path:
    val = CONFIG.get(section, {}).get(key)
    return Path(val).expanduser() if val else default


def clips_root() -> Path:
    return Path(CONFIG.get("clips", {}).get("output_root", EVENT_CLIPS_DIR)).expanduser()


def jobs_root() -> Path:
    return _cfg_path("jobs", "root", JOB_DIR)


def calibrations_root() -> Path:
    return _cfg_path("calibrations", "root", CALIBRATION_DIR)


def localizations_root() -> Path:
    return _cfg_path("localizations", "root", LOCALIZATION_DIR)


def location_overrides_root() -> Path:
    return _cfg_path("location_overrides", "root", LOCATION_OVERRIDE_DIR)


def active_location_override_file() -> Path:
    return _cfg_path("location_overrides", "active_file", location_overrides_root() / "active_location_override.txt")


def location_override_enabled_file() -> Path:
    return location_overrides_root() / "active_location_override_enabled.txt"


def location_override_enabled() -> bool:
    p = location_override_enabled_file()
    if not p.exists():
        return False
    return p.read_text(encoding="utf-8", errors="replace").strip().lower() in {"1", "true", "yes", "on", "enabled"}



def _override_split_row(line: str, expected_cols: int | None = None) -> list[str]:
    """Split a station-table row written by the dashboard.

    Tabs are preferred because they match the pipeline's sectioned-table TXT
    files.  Whitespace splitting is accepted as a fallback so older hand-made
    files are still readable by the dashboard.
    """
    if "\t" in line:
        return [x.strip() for x in line.split("\t")]
    if expected_cols and expected_cols > 1:
        return [x.strip() for x in line.split(None, expected_cols - 1)]
    return line.split()


def _unit_from_section_name(name: str) -> str:
    name = name.strip()
    if name.lower().startswith("unit "):
        return name.split(None, 1)[1].strip()
    return name


def _float_or_original(value: Any) -> Any:
    try:
        return float(value)
    except Exception:
        return value


def _set_override_value(rec: dict[str, Any], key: str, value: Any) -> None:
    kl = str(key).strip().lower()
    if kl in {"lat", "latitude"}:
        rec["lat"] = _float_or_original(value)
    elif kl in {"lon", "longitude"}:
        rec["lon"] = _float_or_original(value)
    elif kl in {"position_sigma_m", "sigma_m", "accuracy_m", "elevation_m", "alt_m"}:
        rec[kl] = _float_or_original(value)
    elif kl == "enabled":
        rec["enabled"] = str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}
    else:
        rec[str(key).strip()] = value


def _normalized_override_units(units: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for unit, raw in (units or {}).items():
        if not unit:
            continue
        rec = dict(raw or {})
        if rec.get("enabled") is False:
            continue
        lat = rec.get("lat") if rec.get("lat") is not None else rec.get("latitude")
        lon = rec.get("lon") if rec.get("lon") is not None else rec.get("longitude")
        if lat in (None, "") or lon in (None, ""):
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            continue
        sigma = rec.get("position_sigma_m", rec.get("sigma_m", rec.get("accuracy_m", 1.5)))
        try:
            sigma_f = float(sigma)
        except Exception:
            sigma_f = 1.5
        out[str(unit).strip()] = {
            **rec,
            "enabled": rec.get("enabled", True),
            "lat": lat_f,
            "lon": lon_f,
            "position_sigma_m": sigma_f,
            "sigma_m": sigma_f,
            "source": rec.get("source", "aru_dashboard_map_override"),
        }
    return out


def parse_location_override_txt(path: Path | None = None) -> dict[str, Any]:
    """Read dashboard/pipeline location overrides.

    The current on-disk format is the pipeline-compatible sectioned table:

        [stations]
        unit\tlat\tlon\tposition_sigma_m\tsigma_m\tsource
        five\t...\t...\t...\t...\taru_dashboard_map_override

    For backwards compatibility, this also reads the older dashboard format with
    one section per unit, e.g. [unit five].  The API return shape is unchanged:
    {enabled, path, units}.
    """
    path = path or active_location_override_file()
    units: dict[str, Any] = {}
    if not path.exists():
        return {"enabled": location_override_enabled(), "path": str(path), "units": units}

    current_unit: str | None = None
    section: str | None = None
    station_headers: list[str] | None = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            lname = name.lower()
            station_headers = None
            current_unit = None
            if lname == "global":
                section = "global"
            elif lname == "stations":
                section = "stations"
            else:
                section = "unit"
                current_unit = _unit_from_section_name(name)
                units.setdefault(current_unit, {"enabled": True})
            continue

        if section == "stations":
            parts = _override_split_row(line, len(station_headers) if station_headers else None)
            if not parts:
                continue
            if station_headers is None:
                lowered = [p.lower() for p in parts]
                if "unit" in lowered and any(k in lowered for k in ("lat", "latitude")) and any(k in lowered for k in ("lon", "longitude")):
                    station_headers = parts
                continue
            if len(parts) < len(station_headers):
                parts += [""] * (len(station_headers) - len(parts))
            row = dict(zip(station_headers, parts))
            unit = _unit_from_section_name(str(row.get("unit") or row.get("station") or "").strip())
            if not unit:
                continue
            rec = units.setdefault(unit, {"enabled": True})
            for k, v in row.items():
                if str(k).strip().lower() in {"unit", "station"}:
                    continue
                _set_override_value(rec, k, v)
            continue

        if section == "unit" and current_unit and "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            rec = units.setdefault(current_unit, {"enabled": True})
            _set_override_value(rec, k, v)
            continue

        # [global] metadata is currently informational only for the dashboard.

    return {"enabled": location_override_enabled(), "path": str(path), "units": _normalized_override_units(units)}


def write_location_override_txt(units: dict[str, Any], enabled: bool = True, path: Path | None = None) -> Path:
    """Write a pipeline-compatible location override file.

    The localization/calibration scripts describe --locations as a TXT file with
    a [stations] section.  Writing the dashboard override in that format lets the
    same active_location_override.txt be used directly by both the dashboard map
    and the existing pipeline scripts.
    """
    path = path or active_location_override_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalized_override_units(units)

    lines = [
        "# ARU location override TXT v2",
        "# Written by aru-dashboard map interface",
        "# Pipeline-compatible sectioned table for --locations.",
        "",
        "[global]",
        "coordinate_system = wgs84",
        "default_position_sigma_m = 1.5",
        "",
        "[stations]",
        "unit\tlat\tlon\tposition_sigma_m\tsigma_m\tsource",
    ]

    for unit in sorted(normalized.keys()):
        rec = normalized[unit]
        source = str(rec.get("source") or "aru_dashboard_map_override")
        sigma = float(rec.get("position_sigma_m", rec.get("sigma_m", 1.5)))
        lines.append(
            "\t".join([
                unit,
                f"{float(rec['lat']):.12f}",
                f"{float(rec['lon']):.12f}",
                f"{sigma:.4f}",
                f"{sigma:.4f}",
                source,
            ])
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    location_override_enabled_file().parent.mkdir(parents=True, exist_ok=True)
    location_override_enabled_file().write_text("true\n" if enabled else "false\n", encoding="utf-8")
    return path


def ensure_pipeline_location_override_txt(path: Path | None = None) -> Path | None:
    """Normalize an existing dashboard override into the [stations] format.

    This is intentionally called before launching calibration/localization so an
    older active override file from a previous dashboard version cannot be passed
    to the pipeline in the legacy [unit five] format.
    """
    path = path or active_location_override_file()
    if not path.exists():
        return None
    parsed = parse_location_override_txt(path)
    units = parsed.get("units") or {}
    if not units:
        return path
    return write_location_override_txt(units, enabled=location_override_enabled(), path=path)

def resolve_locations_arg(payload: dict[str, Any], job: dict[str, Any] | None = None) -> Path | None:
    explicit = str(payload.get("locations") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        # If the user explicitly passes a dashboard override TXT, normalize it
        # so the existing calibration/localization scripts receive [stations].
        if p.exists() and p.suffix.lower() == ".txt" and "override" in p.name.lower():
            try:
                normalized = ensure_pipeline_location_override_txt(p)
                return normalized or p
            except Exception as e:
                if job is not None:
                    _append_job_log(job, f"[warn] could not normalize explicit location override {p}: {e}")
        return p
    use_override = bool(payload.get("use_location_override") or payload.get("use_map_overrides"))
    if use_override:
        p = active_location_override_file()
        if p.exists() and location_override_enabled():
            try:
                normalized = ensure_pipeline_location_override_txt(p)
                return normalized or p
            except Exception as e:
                if job is not None:
                    _append_job_log(job, f"[warn] could not normalize active location override {p}: {e}; passing original file.")
                return p
        if job is not None:
            _append_job_log(job, "[warn] use_location_override requested but no enabled override file exists; using event/default locations.")
    return None


def active_calibration_default() -> Path:
    return _cfg_path("calibrations", "active_file", calibrations_root() / "active_session_calibration.txt")


def active_calibration_pointer() -> Path:
    return calibrations_root() / "active_calibration_path.txt"


def resolve_active_calibration() -> Path | None:
    pointer = active_calibration_pointer()
    if pointer.exists():
        p = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
        if p.exists():
            return p
    default = active_calibration_default()
    return default if default.exists() else None


def pipeline_python() -> str:
    return str(Path(CONFIG.get("pipeline", {}).get("python", sys.executable)).expanduser())


def pipeline_script(name: str) -> Path:
    pcfg = CONFIG.get("pipeline", {})
    configured = {"acquire.py": pcfg.get("acquire_script"), "aru_calibrate_session.py": pcfg.get("calibrate_script"), "aru_localize_event.py": pcfg.get("localize_script")}.get(name)
    if configured:
        p = Path(configured).expanduser()
        if not p.is_absolute():
            p = Path(pcfg.get("script_dir", BASE_DIR / "pipeline")).expanduser() / p
        return p
    script_dir = Path(pcfg.get("script_dir", BASE_DIR / "pipeline")).expanduser()
    candidates = [script_dir / name, BASE_DIR / name, BASE_DIR / "aru_pipeline_scripts" / name]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def parse_units(value: Any) -> str:
    if value is None or str(value).strip() == "":
        default = CONFIG.get("pipeline", {}).get("default_units")
        if isinstance(default, list):
            return ",".join(default)
        if default:
            return str(default)
        return ",".join(CONFIG.get("units", {}).keys())
    return str(value).replace(" ", ",")


def safe_label(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(s).strip()).strip("_") or datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_label(s: str) -> str:
    return safe_label(str(s).replace(":", "-").replace(" ", "_"))


def parse_kv_stdout(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def artifact_info(path: Path) -> dict[str, Any]:
    rel = path.resolve().relative_to(DATA_DIR.resolve())
    rel_url = quote(str(rel).replace(os.sep, "/"), safe="/")
    kind = "file"
    if path.suffix.lower() in {".html", ".htm"}:
        kind = "html"
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        kind = "image"
    elif path.suffix.lower() in {".txt", ".csv", ".log"}:
        kind = "text"
    elif path.suffix.lower() in {".json", ".geojson"}:
        kind = "geojson" if path.suffix.lower() == ".geojson" else "json"
    return {"name": path.name, "path": str(rel), "url": f"/artifact/{rel_url}", "kind": kind, "size_bytes": path.stat().st_size if path.exists() else None, "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None}


def scan_artifacts(root: Path, limit: int = 80) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    items: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ARTIFACT_EXTS:
            items.append(p)
    items.sort(key=lambda p: (p.suffix.lower() not in {".html", ".png", ".txt", ".csv"}, str(p)))
    out: list[dict[str, Any]] = []
    for p in items[:limit]:
        try:
            out.append(artifact_info(p))
        except Exception:
            pass
    return out


def event_dirs() -> list[Path]:
    root = clips_root()
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("event_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def _job_path(job_id: str) -> Path:
    return jobs_root() / job_id


def _new_job(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{kind}_" + uuid.uuid4().hex[:8]
    jdir = _job_path(job_id); jdir.mkdir(parents=True, exist_ok=True)
    job = {"job_id": job_id, "kind": kind, "status": "queued", "created_at": utc_now_iso(), "started_at": None, "finished_at": None, "payload": payload, "job_dir": str(jdir), "log_path": str(jdir / "job.log"), "commands": [], "outputs": {}, "artifacts": [], "error": None, "log_tail": []}
    STATE["jobs"][job_id] = job
    (jdir / "payload.txt").write_text("\n".join(f"{k} = {v}" for k, v in payload.items()) + "\n", encoding="utf-8")
    return job


def _append_job_log(job: dict[str, Any], line: str) -> None:
    line = line.rstrip("\n")
    job["log_tail"].append(line); job["log_tail"] = job["log_tail"][-300:]
    with open(job["log_path"], "a", encoding="utf-8") as f:
        f.write(line + "\n")


async def _run_logged(job: dict[str, Any], cmd: list[str], cwd: Path | None = None) -> str:
    display = shlex.join([str(x) for x in cmd])
    job["commands"].append(display)
    _append_job_log(job, f"$ {display}")
    proc = await asyncio.create_subprocess_exec(*[str(x) for x in cmd], cwd=str(cwd or BASE_DIR), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    lines: list[str] = []
    assert proc.stdout is not None
    while True:
        b = await proc.stdout.readline()
        if not b:
            break
        line = b.decode(errors="replace").rstrip("\n")
        lines.append(line)
        _append_job_log(job, line)
    rc = await proc.wait()
    _append_job_log(job, f"[exit {rc}] {display}")
    text = "\n".join(lines)
    if rc != 0:
        raise RuntimeError(f"Command failed with exit {rc}: {display}")
    return text


def build_acquire_cmd(payload: dict[str, Any]) -> list[str]:
    timestamp = str(payload.get("timestamp") or payload.get("event_time") or "").strip()
    if not timestamp:
        raise ValueError("timestamp is required")
    cmd = [pipeline_python(), str(pipeline_script("acquire.py")), timestamp, "--clip-half-s", str(float(payload.get("clip_half_s", 30.0))), "--units", parse_units(payload.get("units")), "--output-root", str(clips_root()), "--finalize-wait-s", str(float(payload.get("finalize_wait_s", CONFIG.get("pipeline", {}).get("finalize_wait_s", 10.0))))]
    if payload.get("no_rotate_if_active"):
        cmd.append("--no-rotate-if-active")
    if payload.get("no_reuse"):
        cmd.append("--no-reuse")
    if payload.get("force_refetch"):
        cmd.append("--force-refetch")
    return cmd


async def ensure_clips_for_job(job: dict[str, Any], payload: dict[str, Any]) -> str:
    stdout = await _run_logged(job, build_acquire_cmd(payload), cwd=pipeline_script("acquire.py").parent)
    kv = parse_kv_stdout(stdout)
    event_dir = kv.get("event_dir")
    if not event_dir:
        raise RuntimeError("acquire.py did not print event_dir = ...")
    job["outputs"]["event_dir"] = event_dir
    return event_dir


def calibration_txt_from_payload(payload: dict[str, Any]) -> Path:
    requested = str(payload.get("calibration_txt") or "active").strip()
    if requested in {"", "active", "default"}:
        return active_calibration_default()
    return Path(requested).expanduser()


async def run_calibration_for_job(job: dict[str, Any], payload: dict[str, Any], event_dir: str) -> None:
    timestamp = str(payload.get("timestamp") or payload.get("event_time") or datetime.now().isoformat())
    out_dir = Path(payload.get("out_dir") or (calibrations_root() / f"calibration_{timestamp_label(timestamp)}_{job['job_id'][-8:]}"))
    cal_txt = calibration_txt_from_payload(payload)
    out_dir.mkdir(parents=True, exist_ok=True); cal_txt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [pipeline_python(), str(pipeline_script("aru_calibrate_session.py")), event_dir, "--calibration-txt", str(cal_txt), "--ref", str(payload.get("ref", "five")), "--source-unit", str(payload.get("source_unit", "five")), "--units", parse_units(payload.get("units")), "--out-dir", str(out_dir), "--sound-speed", str(float(payload.get("sound_speed", 343.0))), "--max-tau-s", str(float(payload.get("max_tau_s", 0.10))), "--highpass-hz", str(float(payload.get("highpass_hz", 300.0))), "--draws", str(int(payload.get("draws", 2000)))]
    if payload.get("event_ref_offset"):
        cmd.extend(["--event-ref-offset", str(payload["event_ref_offset"])])
    loc_path = resolve_locations_arg(payload, job)
    if loc_path is not None:
        cmd.extend(["--locations", str(loc_path)])
        job["outputs"]["locations_used"] = str(loc_path)
    if payload.get("source_lat") not in (None, "") and payload.get("source_lon") not in (None, ""):
        cmd.extend(["--source-lat", str(payload["source_lat"]), "--source-lon", str(payload["source_lon"])])
    await _run_logged(job, cmd, cwd=pipeline_script("aru_calibrate_session.py").parent)
    active_calibration_pointer().parent.mkdir(parents=True, exist_ok=True)
    active_calibration_pointer().write_text(str(cal_txt) + "\n", encoding="utf-8")
    job["outputs"].update({"calibration_txt": str(cal_txt), "calibration_out_dir": str(out_dir)})


async def run_localization_for_job(job: dict[str, Any], payload: dict[str, Any], event_dir: str) -> None:
    timestamp = str(payload.get("timestamp") or payload.get("event_time") or datetime.now().isoformat())
    out_dir = Path(payload.get("out_dir") or (localizations_root() / f"localization_{timestamp_label(timestamp)}_{job['job_id'][-8:]}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [pipeline_python(), str(pipeline_script("aru_localize_event.py")), event_dir, "--ref", str(payload.get("ref", "five")), "--mode", str(payload.get("mode", "impulse")), "--units", parse_units(payload.get("units")), "--out-dir", str(out_dir), "--sound-speed", str(float(payload.get("sound_speed", 343.0))), "--max-tau-s", str(float(payload.get("max_tau_s", 0.10))), "--highpass-hz", str(float(payload.get("highpass_hz", 300.0))), "--draws", str(int(payload.get("draws", 2000))), "--mc", str(int(payload.get("mc", 1000)))]
    if payload.get("event_ref_offset"):
        cmd.extend(["--event-ref-offset", str(payload["event_ref_offset"])])
    loc_path = resolve_locations_arg(payload, job)
    if loc_path is not None:
        cmd.extend(["--locations", str(loc_path)])
        job["outputs"]["locations_used"] = str(loc_path)
    if payload.get("bird_bandpass"):
        cmd.extend(["--bird-bandpass", str(payload["bird_bandpass"])])
    cal_mode = str(payload.get("calibration", "active")).strip()
    cal_path: Path | None = None
    if cal_mode not in {"", "none", "None", "false"}:
        cal_path = resolve_active_calibration() if cal_mode == "active" else Path(cal_mode).expanduser()
        if cal_path is not None and cal_path.exists():
            cmd.extend(["--calibration-txt", str(cal_path)])
        elif cal_mode == "active":
            _append_job_log(job, "[warn] No active calibration found; localizing without calibration.")
        else:
            raise RuntimeError(f"Requested calibration file does not exist: {cal_path}")
    await _run_logged(job, cmd, cwd=pipeline_script("aru_localize_event.py").parent)
    job["outputs"].update({"localization_out_dir": str(out_dir), "calibration_used": str(cal_path) if cal_path else ""})


async def run_pipeline_job(job_id: str, kind: str, payload: dict[str, Any]) -> None:
    job = STATE["jobs"][job_id]
    job["status"] = "running"; job["started_at"] = utc_now_iso()
    add_log(f"job {job_id} started: {kind}")
    try:
        event_dir = await ensure_clips_for_job(job, payload)
        if kind == "calibrate":
            await run_calibration_for_job(job, payload, event_dir)
        elif kind == "localize":
            await run_localization_for_job(job, payload, event_dir)
        elif kind != "acquire":
            raise ValueError(f"unknown job kind: {kind}")
        roots = [Path(event_dir)]
        for key in ["calibration_out_dir", "localization_out_dir"]:
            if job["outputs"].get(key):
                roots.append(Path(job["outputs"][key]))
        if job["outputs"].get("calibration_txt"):
            roots.append(Path(job["outputs"]["calibration_txt"]).parent)
        arts: list[dict[str, Any]] = []
        for root in roots:
            try:
                if root.exists() and DATA_DIR.resolve() in [root.resolve(), *root.resolve().parents]:
                    arts.extend(scan_artifacts(root, limit=80))
            except Exception:
                pass
        seen = set(); job["artifacts"] = []
        for a in arts:
            if a["path"] not in seen:
                seen.add(a["path"]); job["artifacts"].append(a)
        job["status"] = "succeeded"
        add_log(f"job {job_id} succeeded")
    except Exception as e:
        job["status"] = "failed"; job["error"] = str(e)
        _append_job_log(job, f"[error] {e}")
        add_log(f"job {job_id} failed: {e}")
    finally:
        job["finished_at"] = utc_now_iso()
        jdir = _job_path(job_id)
        (jdir / "status.txt").write_text("\n".join([f"job_id = {job_id}", f"kind = {kind}", f"status = {job['status']}", f"created_at = {job['created_at']}", f"started_at = {job.get('started_at') or ''}", f"finished_at = {job.get('finished_at') or ''}", f"error = {job.get('error') or ''}"]) + "\n", encoding="utf-8")


def start_pipeline_job(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind not in {"acquire", "calibrate", "localize"}:
        raise HTTPException(status_code=404, detail="unknown job type")
    job = _new_job(kind, payload)
    asyncio.create_task(run_pipeline_job(job["job_id"], kind, payload))
    return job


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(poll_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/status")
async def api_status():
    return JSONResponse(STATE)


@app.get("/tiles/esri/{z}/{x}/{y}.jpg")
async def cached_esri_tile(z: int, x: int, y: int):
    """Serve locally cached Esri/XYZ tiles from data/tile_cache.

    Expected layout:
      data/tile_cache/esri_world_imagery/<z>/<x>/<y>.jpg
    """
    if z < 0 or z > 23 or x < 0 or y < 0:
        raise HTTPException(status_code=404, detail="tile out of range")
    n = 2 ** z
    if x >= n or y >= n:
        raise HTTPException(status_code=404, detail="tile out of range")

    path = (TILE_CACHE_DIR / str(z) / str(x) / f"{y}.jpg").resolve()
    try:
        path.relative_to(TILE_CACHE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="tile path outside cache")

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="tile not cached")

    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.post("/api/location/reset")
async def api_location_reset():
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    STATE["location"] = {"state": "collecting", "session_id": session_id, "started_at": utc_now_iso(), "frozen_at": None, "fixes": {unit: [] for unit in CONFIG.get("units", {}).keys()}, "summaries": {}}
    update_location_summaries()
    add_log(f"location averaging reset; session={session_id}")
    return {"ok": True, "location": STATE["location"]}


@app.post("/api/location/freeze")
async def api_location_freeze():
    STATE["location"]["state"] = "frozen"
    STATE["location"]["frozen_at"] = utc_now_iso()
    write_location_summaries()
    add_log("location averaging frozen; wrote JSON and TXT location summaries")
    return {"ok": True, "location": STATE["location"]}


@app.post("/api/location/unfreeze")
async def api_location_unfreeze():
    if STATE["location"].get("session_id") is None:
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        STATE["location"]["session_id"] = session_id
        STATE["location"]["started_at"] = utc_now_iso()
        STATE["location"]["fixes"] = {unit: [] for unit in CONFIG.get("units", {}).keys()}
    STATE["location"]["state"] = "collecting"
    STATE["location"]["frozen_at"] = None
    add_log("location averaging unfrozen / collecting")
    return {"ok": True, "location": STATE["location"]}



@app.post("/api/gpsd/restart")
async def api_gpsd_restart():
    """Restart gpsd.socket/gpsd.service on all configured units.

    Requires passwordless sudo for systemctl restart gpsd.socket/gpsd.service on
    remote units unless the dashboard user already has permission.
    """
    units_cfg = CONFIG.get("units", {})
    tasks = [restart_gpsd_for_unit(unit, cfg) for unit, cfg in units_cfg.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[str, Any] = {}
    ok_all = True
    for unit, result in zip(units_cfg.keys(), results):
        if isinstance(result, Exception):
            out[unit] = {"unit": unit, "ok": False, "error": str(result)}
            ok_all = False
        else:
            out[unit] = result
            if not result.get("ok"):
                ok_all = False
    msg = "gpsd restart requested on all configured units"
    if not ok_all:
        msg += " (some units failed; check returned stderr and sudo permissions)"
    add_log(msg)
    return {"ok": ok_all, "results": out}


@app.get("/api/map/locations")
async def api_map_locations():
    summaries = STATE.get("location", {}).get("summaries", {}) or {}
    units_out: dict[str, Any] = {}
    for unit in CONFIG.get("units", {}).keys():
        status = STATE.get("units", {}).get(unit, {}) or {}
        summary = summaries.get(unit, {}) or {}
        latest = status.get("gps", {}) or {}
        lat = summary.get("lat_deg")
        lon = summary.get("lon_deg")
        units_out[unit] = {
            "unit": unit,
            "avg_lat": lat,
            "avg_lon": lon,
            "latest_lat": latest.get("lat"),
            "latest_lon": latest.get("lon"),
            "quality": summary.get("quality"),
            "n_fixes": summary.get("n_fixes", 0),
            "median_eph_m": summary.get("median_eph_m"),
            "scatter_horizontal_m": summary.get("scatter_horizontal_m"),
            "reachable": status.get("reachable"),
            "last_seen": status.get("last_seen"),
        }
    return {"ok": True, "location_state": STATE.get("location", {}).get("state"), "units": units_out, "overrides": parse_location_override_txt()}


@app.get("/api/map/location-overrides")
async def api_get_location_overrides():
    return parse_location_override_txt()


@app.post("/api/map/location-overrides")
async def api_save_location_overrides(payload: dict[str, Any] = Body(default={})):
    units = payload.get("units") or {}
    if not isinstance(units, dict):
        raise HTTPException(status_code=400, detail="units must be an object keyed by unit name")
    enabled = bool(payload.get("enabled", True))
    p = write_location_override_txt(units, enabled=enabled)
    add_log(f"location override pins saved to {p}; enabled={enabled}")
    return {"ok": True, "path": str(p), "enabled": enabled, "overrides": parse_location_override_txt(p)}


@app.delete("/api/map/location-overrides")
async def api_clear_location_overrides():
    p = active_location_override_file()
    if p.exists():
        p.unlink()
    location_override_enabled_file().write_text("false\n", encoding="utf-8")
    add_log("location override pins cleared")
    return {"ok": True, "path": str(p), "enabled": False}


def _latest_result_dir(kind: str) -> Path | None:
    if kind == "calibration":
        roots = [p for p in calibrations_root().rglob("calibration_*") if p.is_dir()] if calibrations_root().exists() else []
    elif kind == "localization":
        roots = [p for p in localizations_root().rglob("localization_*") if p.is_dir()] if localizations_root().exists() else []
    else:
        roots = []
    if not roots:
        return None
    return sorted(roots, key=lambda p: p.stat().st_mtime, reverse=True)[0]


@app.get("/api/map/result-layers")
async def api_map_result_layers(kind: str = "localization"):
    d = _latest_result_dir(kind)
    if d is None:
        return {"ok": True, "kind": kind, "dir": "", "artifacts": [], "html": [], "geojson": []}
    artifacts = scan_artifacts(d, limit=100)
    html = [a for a in artifacts if a.get("kind") == "html"]
    geojson = [a for a in artifacts if a.get("kind") == "geojson"]
    return {"ok": True, "kind": kind, "dir": str(d), "artifacts": artifacts, "html": html, "geojson": geojson}


@app.get("/api/dashboard-config")
async def api_dashboard_config():
    return {"units": list(CONFIG.get("units", {}).keys()), "clips_root": str(clips_root()), "calibrations_root": str(calibrations_root()), "localizations_root": str(localizations_root()), "active_calibration": str(resolve_active_calibration() or ""), "location_override": parse_location_override_txt(), "pipeline_scripts": {"acquire": str(pipeline_script("acquire.py")), "calibrate": str(pipeline_script("aru_calibrate_session.py")), "localize": str(pipeline_script("aru_localize_event.py"))}}


@app.post("/api/jobs/acquire")
async def api_job_acquire(payload: dict[str, Any] = Body(default={})):
    return start_pipeline_job("acquire", payload)


@app.post("/api/jobs/calibrate")
async def api_job_calibrate(payload: dict[str, Any] = Body(default={})):
    return start_pipeline_job("calibrate", payload)


@app.post("/api/jobs/localize")
async def api_job_localize(payload: dict[str, Any] = Body(default={})):
    return start_pipeline_job("localize", payload)


@app.get("/api/jobs")
async def api_jobs():
    jobs = sorted(STATE["jobs"].values(), key=lambda j: j.get("created_at") or "", reverse=True)
    return {"jobs": jobs[:100]}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = STATE["jobs"].get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
async def api_job_log(job_id: str):
    job = STATE["jobs"].get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    p = Path(job["log_path"])
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


@app.get("/api/events")
async def api_events():
    rows = []
    for d in event_dirs()[:100]:
        rows.append({"id": d.name, "path": str(d), "mtime": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"), "artifacts": scan_artifacts(d, limit=60)})
    return {"events": rows}


@app.get("/api/calibrations")
async def api_calibrations():
    root = calibrations_root(); root.mkdir(parents=True, exist_ok=True)
    active = resolve_active_calibration()
    rows = []
    for p in sorted(root.rglob("*"), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
        if p.is_file() and p.name.endswith(".txt") and ("calibration" in p.name or p.name == "active_session_calibration.txt"):
            try:
                art = artifact_info(p)
            except Exception:
                continue
            art["active"] = bool(active and p.resolve() == active.resolve())
            rows.append(art)
    return {"active": str(active or ""), "calibrations": rows[:100]}


@app.post("/api/calibrations/activate")
async def api_activate_calibration(payload: dict[str, Any] = Body(default={})):
    rel_or_path = payload.get("path")
    if not rel_or_path:
        raise HTTPException(status_code=400, detail="path is required")
    p = Path(str(rel_or_path)).expanduser()
    if not p.is_absolute():
        p = DATA_DIR / p
    p = p.resolve()
    try:
        p.relative_to(DATA_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="calibration must be under dashboard data directory")
    if not p.exists():
        raise HTTPException(status_code=404, detail="calibration file not found")
    active_calibration_pointer().parent.mkdir(parents=True, exist_ok=True)
    active_calibration_pointer().write_text(str(p) + "\n", encoding="utf-8")
    add_log(f"active calibration set to {p}")
    return {"ok": True, "active": str(p)}


@app.get("/api/localizations")
async def api_localizations():
    root = localizations_root(); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted([p for p in root.rglob("localization_*") if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[:100]:
        rows.append({"id": d.name, "path": str(d), "mtime": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"), "artifacts": scan_artifacts(d, limit=60)})
    return {"localizations": rows}


@app.get("/artifact/{rel_path:path}")
async def artifact(rel_path: str):
    target = (DATA_DIR / rel_path).resolve()
    try:
        target.relative_to(DATA_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="artifact path outside data directory")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(target)
