"""Command-line interface: ``pix2geo`` or ``python -m pix2geo``.

Example::

    pix2geo --dem n46_w122_1arc_v3.tif --geoid egm96 \\
        --lat 46.80 --lon -121.80 --alt 4200 --yaw 45 --pitch -25 --roll 0 \\
        --size 1920 1080 --hfov 60 --pixel 960 540 --pixel 100 900
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from typing import Union, get_args

import numpy as np

from ._typing import FloatArray
from .backends import BackendChoice
from .camera import CameraIntrinsics
from .geolocator import Geolocator, GeoResult
from .pose import AltRef, CameraPose
from .terrain import Terrain

#: Approximate length of one degree of latitude in kilometers.
_KM_PER_DEG_LAT = 111.0

#: One output row: pixel, position, range and hit flag. NaN becomes None.
Row = dict[str, Union[float, bool, None]]


def _geoid_arg(s: str) -> float | str:
    try:
        return float(s)
    except ValueError:
        return s


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pix2geo",
        description="Geolocate camera pixels on a terrain model.",
    )

    tg = p.add_argument_group("terrain")
    tg.add_argument(
        "--dem",
        action="append",
        required=True,
        help="elevation raster (GeoTIFF, DTED, ...); repeat to mosaic tiles",
    )
    tg.add_argument(
        "--geoid",
        type=_geoid_arg,
        default=None,
        help="geoid of the DEM heights: egm96, egm2008, EPSG code, grid file "
        "or constant undulation in meters (default: heights are ellipsoidal)",
    )
    tg.add_argument(
        "--radius",
        type=float,
        default=None,
        help="read only the DEM within this many km of the camera",
    )

    pg = p.add_argument_group("camera pose")
    pg.add_argument("--lat", type=float, required=True)
    pg.add_argument("--lon", type=float, required=True)
    pg.add_argument("--alt", type=float, required=True, help="camera altitude in meters")
    pg.add_argument("--alt-ref", choices=get_args(AltRef), default="ellipsoid")
    pg.add_argument("--yaw", type=float, required=True, help="degrees clockwise from north")
    pg.add_argument("--pitch", type=float, required=True, help="degrees, -90 is nadir")
    pg.add_argument("--roll", type=float, default=0.0, help="degrees, right side down")

    ig = p.add_argument_group("camera intrinsics")
    ig.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), required=True)
    fov = ig.add_mutually_exclusive_group(required=True)
    fov.add_argument("--hfov", type=float, help="horizontal field of view in degrees")
    fov.add_argument("--vfov", type=float, help="vertical field of view in degrees")
    fov.add_argument("--dfov", type=float, help="diagonal field of view in degrees")
    fov.add_argument(
        "--K",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help="focal lengths and principal point in pixels",
    )
    ig.add_argument(
        "--dist",
        type=float,
        nargs="+",
        default=None,
        help="OpenCV distortion coefficients k1 k2 p1 p2 [k3 [k4 k5 k6]]",
    )

    ig = p.add_argument_group("query")
    ig.add_argument(
        "--pixel",
        type=float,
        nargs=2,
        action="append",
        metavar=("U", "V"),
        required=True,
        help="pixel (column, row), top-left origin; repeatable",
    )
    ig.add_argument("--backend", default="auto", choices=get_args(BackendChoice))
    ig.add_argument(
        "--fallback-height",
        type=float,
        default=None,
        help="ellipsoidal height of the surface for rays that miss the DEM",
    )
    ig.add_argument("--json", action="store_true", help="print JSON")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    camera = _camera(args)
    terrain = _terrain(args)
    pose = CameraPose.from_euler(
        args.lat, args.lon, args.alt, args.yaw, args.pitch, args.roll, alt_ref=args.alt_ref
    )
    pixels = np.asarray(args.pixel, dtype=float)

    geo = Geolocator(terrain, backend=args.backend)
    result = geo.pixel_to_geo(camera, pose, pixels, fallback_height=args.fallback_height)

    rows = _rows(pixels, result)
    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
    else:
        _print_table(rows)

    return 0


def _camera(args: argparse.Namespace) -> CameraIntrinsics:
    w, h = args.size
    if args.K is not None:
        fx, fy, cx, cy = args.K
        return CameraIntrinsics(w, h, fx, fy, cx, cy, dist=args.dist)

    return CameraIntrinsics.from_fov(
        w,
        h,
        hfov=args.hfov,
        vfov=args.vfov,
        dfov=args.dfov,
        dist=args.dist,
    )


def _terrain(args: argparse.Namespace) -> Terrain:
    bounds = None
    if args.radius is not None:
        dlat = args.radius / _KM_PER_DEG_LAT
        dlon = dlat / max(math.cos(math.radians(args.lat)), 1e-6)
        bounds = (args.lon - dlon, args.lat - dlat, args.lon + dlon, args.lat + dlat)

    if len(args.dem) == 1:
        return Terrain.from_file(args.dem[0], bounds=bounds, geoid=args.geoid)

    return Terrain.from_files(args.dem, bounds=bounds, geoid=args.geoid)


def _rows(pixels: FloatArray, result: GeoResult) -> list[Row]:
    """One JSON-ready dict for each pixel. NaN becomes None."""

    def value(a: FloatArray, k: int) -> float | None:
        x = float(a[k])
        return None if math.isnan(x) else x

    return [
        {
            "u": float(u),
            "v": float(v),
            "lat": value(result.lat, k),
            "lon": value(result.lon, k),
            "alt": value(result.alt, k),
            "alt_msl": None if result.alt_msl is None else value(result.alt_msl, k),
            "range": value(result.range, k),
            "hit": bool(result.hit[k]),
        }
        for k, (u, v) in enumerate(pixels)
    ]


def _print_table(rows: list[Row]) -> None:
    def fmt(x: float | bool | None, spec: str) -> str:
        return "nan" if x is None else format(x, spec)

    print(f"{'u':>9} {'v':>9} {'lat':>14} {'lon':>15} {'alt':>10} {'range':>11}  hit")
    for r in rows:
        print(
            f"{r['u']:9.2f} {r['v']:9.2f} {fmt(r['lat'], '14.8f')} {fmt(r['lon'], '15.8f')} "
            f"{fmt(r['alt'], '10.2f')} {fmt(r['range'], '11.2f')}  {r['hit']}"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
