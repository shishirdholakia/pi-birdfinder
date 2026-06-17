#!/usr/bin/env python3
"""
Cache a small, bounded set of Esri World Imagery tiles for offline ARU dashboard use.

Default behavior downloads ONE zoom level over a square bounding box that covers a
radius around a center latitude/longitude. The dashboard then serves these tiles
from /tiles/esri/{z}/{x}/{y}.jpg.

Use only for tile sources and terms that allow the intended offline/cache use.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EARTH_RADIUS_M = 6378137.0
DEFAULT_ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
    "/tile/{z}/{y}/{x}"
)


def clamp_lat(lat: float) -> float:
    # Web Mercator valid latitude range.
    return max(min(lat, 85.05112878), -85.05112878)


def latlon_to_tile(lat_deg: float, lon_deg: float, z: int) -> tuple[int, int]:
    lat_deg = clamp_lat(lat_deg)
    n = 2 ** z
    x = int((lon_deg + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat_deg)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tile_to_latlon(x: int, y: int, z: int) -> tuple[float, float]:
    n = 2 ** z
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def bbox_for_radius(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Return min_lat, min_lon, max_lat, max_lon covering a radius in meters."""
    dlat = math.degrees(radius_m / EARTH_RADIUS_M)
    cos_lat = max(1e-9, math.cos(math.radians(lat)))
    dlon = math.degrees(radius_m / (EARTH_RADIUS_M * cos_lat))
    return clamp_lat(lat - dlat), lon - dlon, clamp_lat(lat + dlat), lon + dlon


def tile_range_for_bbox(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, z: int
) -> tuple[int, int, int, int]:
    # y increases southward, so max_lat gives smaller y.
    x0, y_top = latlon_to_tile(max_lat, min_lon, z)
    x1, y_bottom = latlon_to_tile(min_lat, max_lon, z)
    return min(x0, x1), max(x0, x1), min(y_top, y_bottom), max(y_top, y_bottom)


def tile_center_distance_m(lat: float, lon: float, x: int, y: int, z: int) -> float:
    # Approximate center as midpoint between tile NW and SE corners.
    lat_n, lon_w = tile_to_latlon(x, y, z)
    lat_s, lon_e = tile_to_latlon(x + 1, y + 1, z)
    clat = 0.5 * (lat_n + lat_s)
    clon = 0.5 * (lon_w + lon_e)
    dlat = math.radians(clat - lat)
    dlon = math.radians(clon - lon)
    lat0 = math.radians(lat)
    return EARTH_RADIUS_M * math.hypot(dlat, dlon * math.cos(lat0))


def iter_tiles(lat: float, lon: float, radius_m: float, z: int, circle: bool) -> list[tuple[int, int]]:
    min_lat, min_lon, max_lat, max_lon = bbox_for_radius(lat, lon, radius_m)
    x_min, x_max, y_min, y_max = tile_range_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
    out: list[tuple[int, int]] = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if circle:
                # Keep tiles whose center is inside radius plus one approximate tile diagonal cushion.
                lat_n, lon_w = tile_to_latlon(x, y, z)
                lat_s, lon_e = tile_to_latlon(x + 1, y + 1, z)
                lat0 = math.radians(lat)
                tile_h = abs(math.radians(lat_n - lat_s) * EARTH_RADIUS_M)
                tile_w = abs(math.radians(lon_e - lon_w) * EARTH_RADIUS_M * math.cos(lat0))
                cushion = 0.5 * math.hypot(tile_w, tile_h)
                if tile_center_distance_m(lat, lon, x, y, z) > radius_m + cushion:
                    continue
            out.append((x, y))
    return out


def download_one(url: str, dest: Path, timeout_s: float, user_agent: str) -> tuple[bool, str]:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read()
        if status != 200:
            return False, f"HTTP {status}"
        if not data:
            return False, "empty response"
        if "image" not in ctype.lower() and not data.startswith((b"\xff\xd8", b"\x89PNG")):
            return False, f"unexpected content-type {ctype!r}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        tmp.replace(dest)
        return True, f"ok {len(data)} bytes"
    except urllib.error.HTTPError as e:
        return False, f"HTTPError {e.code}"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download a small Esri imagery tile cache around a field site.")
    ap.add_argument("--lat", type=float, required=True, help="center latitude in decimal degrees")
    ap.add_argument("--lon", type=float, required=True, help="center longitude in decimal degrees")
    ap.add_argument("--radius-m", type=float, default=500.0, help="radius in meters; default 500")
    ap.add_argument("--zoom", type=int, default=20, help="single zoom level to cache; default 20")
    ap.add_argument("--out-root", default="data/tile_cache/esri_world_imagery", help="output root")
    ap.add_argument("--url-template", default=DEFAULT_ESRI_URL, help="tile URL template using {z}, {x}, {y}; Esri URL uses z/y/x")
    ap.add_argument("--max-tiles", type=int, default=1500, help="safety cap; increase intentionally if needed")
    ap.add_argument("--sleep-s", type=float, default=0.08, help="delay between downloads")
    ap.add_argument("--timeout-s", type=float, default=12.0, help="per-tile timeout")
    ap.add_argument("--force", action="store_true", help="redownload existing tiles")
    ap.add_argument("--dry-run", action="store_true", help="show tile count and exit")
    ap.add_argument("--square", action="store_true", help="cache full square bbox instead of circle-filtered tiles")
    ap.add_argument("--yes-accept-terms", action="store_true", help="confirm your intended tile use is permitted")
    ap.add_argument("--user-agent", default="aru-dashboard-tile-cache/1.0", help="HTTP User-Agent")
    args = ap.parse_args(argv)

    if not args.yes_accept_terms and not args.dry_run:
        print(
            "Refusing to download until you pass --yes-accept-terms. "
            "Use only with sources/terms that allow your intended small offline cache.",
            file=sys.stderr,
        )
        return 2

    if not (-85.05112878 <= args.lat <= 85.05112878):
        print("Latitude is outside Web Mercator valid range.", file=sys.stderr)
        return 2
    if not (-180.0 <= args.lon <= 180.0):
        print("Longitude must be in [-180, 180].", file=sys.stderr)
        return 2
    if args.zoom < 0 or args.zoom > 23:
        print("Zoom should be between 0 and 23.", file=sys.stderr)
        return 2

    tiles = iter_tiles(args.lat, args.lon, args.radius_m, args.zoom, circle=not args.square)
    print(f"center = {args.lat:.8f}, {args.lon:.8f}")
    print(f"radius_m = {args.radius_m:.1f}")
    print(f"zoom = {args.zoom}")
    print(f"tiles = {len(tiles)}")
    if tiles:
        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]
        print(f"x range = {min(xs)}..{max(xs)}")
        print(f"y range = {min(ys)}..{max(ys)}")

    if len(tiles) > args.max_tiles:
        print(
            f"Tile count {len(tiles)} exceeds --max-tiles {args.max_tiles}. "
            "Reduce zoom/radius or increase the cap intentionally.",
            file=sys.stderr,
        )
        return 3

    if args.dry_run:
        return 0

    out_root = Path(args.out_root)
    ok = 0
    skipped = 0
    failed = 0
    for i, (x, y) in enumerate(tiles, 1):
        dest = out_root / str(args.zoom) / str(x) / f"{y}.jpg"
        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            skipped += 1
            print(f"[{i}/{len(tiles)}] skip {args.zoom}/{x}/{y}")
            continue
        url = args.url_template.format(z=args.zoom, x=x, y=y)
        success, msg = download_one(url, dest, args.timeout_s, args.user_agent)
        if success:
            ok += 1
            print(f"[{i}/{len(tiles)}] saved {args.zoom}/{x}/{y}: {msg}")
        else:
            failed += 1
            print(f"[{i}/{len(tiles)}] FAILED {args.zoom}/{x}/{y}: {msg}", file=sys.stderr)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    # Write a small manifest for human/debug use.
    manifest = out_root / f"cache_manifest_z{args.zoom}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        f.write("# ARU Esri tile cache manifest\n")
        f.write(f"center_lat = {args.lat:.10f}\n")
        f.write(f"center_lon = {args.lon:.10f}\n")
        f.write(f"radius_m = {args.radius_m}\n")
        f.write(f"zoom = {args.zoom}\n")
        f.write(f"tile_count = {len(tiles)}\n")
        f.write(f"downloaded = {ok}\n")
        f.write(f"skipped_existing = {skipped}\n")
        f.write(f"failed = {failed}\n")
        f.write(f"url_template = {args.url_template}\n")

    print(f"done: downloaded={ok} skipped={skipped} failed={failed}")
    print(f"manifest: {manifest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
