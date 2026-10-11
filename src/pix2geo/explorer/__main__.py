"""Command line of the explorer: ``python -m pix2geo.explorer``.

Examples::

    python -m pix2geo.explorer dem.tif --geoid egm96
    python -m pix2geo.explorer n40w106.dt2 n40w105.dt2 --geoid egm96 --radius 20 \\
        --lat 40.02 --lon -105.3 --alt 800 --yaw 250 --pitch -15
    python -m pix2geo.explorer  # a synthetic DEM

Without ``--lat`` and ``--lon``, the camera is above the center of the DEM.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from typing import get_args

import numpy as np
from affine import Affine

from ..backends import BackendChoice
from ..camera import CameraIntrinsics
from ..cli import _geoid_arg
from ..geolocator import Geolocator
from ..pose import AltRef, CameraPose
from ..terrain import Terrain
from .server import serve

#: Approximate length of one degree of latitude in kilometers.
_KM_PER_DEG_LAT = 111.0


def demo_terrain() -> Terrain:
    """A synthetic DEM near Boulder, Colorado: 1801 x 1801 posts, 1 arc-second apart.

    Fractal noise makes the relief. A massif rises in the north-east, and a
    flat lake at 1650 m fills a basin in the south-west.
    """
    n = 1801
    res = 1.0 / 3600
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / (n - 1)
    rng = np.random.default_rng(3)

    # A smooth warp of the coordinates breaks the regular pattern of the waves.
    tau = 2.0 * math.pi
    wx = x + 0.05 * np.sin(tau * (1.3 * y + 0.2)) + 0.02 * np.sin(tau * (3.1 * x + 2.3 * y))
    wy = y + 0.05 * np.sin(tau * (1.1 * x + 0.7)) + 0.02 * np.sin(tau * (2.7 * y - 1.9 * x))

    # Fractal noise: in each octave, nine waves in random directions. The
    # amplitude halves in each octave. Two middle octaves fold into ridges.
    h = np.zeros((n, n), dtype=np.float32)
    for octave in range(7):
        freq = 1.3 * 2.0**octave
        noise = np.zeros((n, n), dtype=np.float32)
        for _ in range(9):
            theta = rng.uniform(0.0, math.pi)
            phase = rng.uniform(0.0, tau)
            noise += np.cos(tau * freq * (math.cos(theta) * wx + math.sin(theta) * wy) + phase)
        noise /= math.sqrt(4.5)
        if octave in (2, 3):
            noise = (1.2 - 2.0 * np.abs(noise)).astype(np.float32)
        h += 0.5**octave * noise

    # Scale to meters, add the massif, and fill the basin with the lake.
    h = (h - h.min()) / (h.max() - h.min())
    massif = np.exp(-((wx - 0.7) ** 2 + (wy - 0.3) ** 2) / 0.04)
    basin = np.exp(-((wx - 0.27) ** 2 + (wy - 0.7) ** 2) / 0.03)
    h = 1700.0 + 1300.0 * h**1.6 + 1200.0 * massif * h - 600.0 * basin
    h = np.maximum(h, 1650.0)

    return Terrain.from_array(h, Affine(res, 0, -105.5, 0, -res, 40.25), 4326)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pix2geo.explorer",
        description="Explore a geolocation scene in 3D and from the camera.",
    )

    tg = p.add_argument_group("terrain")
    tg.add_argument("dem", nargs="*", help="elevation rasters; none gives a synthetic DEM")
    tg.add_argument(
        "--geoid",
        type=_geoid_arg,
        default=None,
        help="geoid of the DEM heights: egm96, egm2008, EPSG code, grid file or "
        "constant undulation in meters (default: heights are ellipsoidal)",
    )
    tg.add_argument(
        "--radius",
        type=float,
        default=None,
        help="read only the DEM within this many km of the camera (needs --lat and --lon)",
    )

    pg = p.add_argument_group("camera pose")
    pg.add_argument("--lat", type=float, default=None, help="default: the DEM center")
    pg.add_argument("--lon", type=float, default=None, help="default: the DEM center")
    pg.add_argument("--alt", type=float, default=1000.0, help="camera altitude in meters")
    pg.add_argument("--alt-ref", choices=get_args(AltRef), default="agl")
    pg.add_argument("--yaw", type=float, default=45.0, help="degrees clockwise from north")
    pg.add_argument("--pitch", type=float, default=-25.0, help="degrees, -90 is nadir")
    pg.add_argument("--roll", type=float, default=0.0, help="degrees, right side down")

    ig = p.add_argument_group("camera intrinsics")
    ig.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=(1920, 1080))
    fov = ig.add_mutually_exclusive_group()
    fov.add_argument("--hfov", type=float, help="horizontal field of view (default 60 degrees)")
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

    sg = p.add_argument_group("session")
    sg.add_argument("--backend", default="auto", choices=get_args(BackendChoice))
    sg.add_argument(
        "--fallback-height",
        type=float,
        default=None,
        help="ellipsoidal height of the surface for rays that miss the DEM",
    )
    sg.add_argument("--rays", type=int, default=256, help="rays in the drawn fan")
    sg.add_argument("--terrain-size", type=int, default=512, help="drawn posts on each axis")
    sg.add_argument("--host", default="127.0.0.1")
    sg.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    sg.add_argument("--no-browser", action="store_true", help="do not open a browser")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    terrain = _terrain(args)
    print(terrain)

    # The camera is above the DEM center unless the arguments place it.
    w, s, e, n = terrain.bounds_wgs84()
    lat = (s + n) / 2 if args.lat is None else args.lat
    lon = (w + e) / 2 if args.lon is None else args.lon
    pose = CameraPose.from_euler(
        lat,
        lon,
        args.alt,
        args.yaw,
        args.pitch,
        args.roll,
        alt_ref=args.alt_ref,
    )

    serve(
        Geolocator(terrain, backend=args.backend),
        _camera(args),
        pose,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        fallback_height=args.fallback_height,
        max_rays=args.rays,
        terrain_size=args.terrain_size,
        sources=list(args.dem),
        geoid=args.geoid,
    )

    return 0


def _camera(args: argparse.Namespace) -> CameraIntrinsics:
    w, h = args.size
    if args.K is not None:
        fx, fy, cx, cy = args.K
        return CameraIntrinsics(w, h, fx, fy, cx, cy, dist=args.dist)

    hfov = args.hfov
    if hfov is None and args.vfov is None and args.dfov is None:
        hfov = 60.0

    return CameraIntrinsics.from_fov(
        w,
        h,
        hfov=hfov,
        vfov=args.vfov,
        dfov=args.dfov,
        dist=args.dist,
    )


def _terrain(args: argparse.Namespace) -> Terrain:
    if not args.dem:
        return demo_terrain()

    bounds = None
    if args.radius is not None:
        if args.lat is None or args.lon is None:
            raise SystemExit("--radius needs --lat and --lon")
        dlat = args.radius / _KM_PER_DEG_LAT
        dlon = dlat / max(math.cos(math.radians(args.lat)), 1e-6)
        bounds = (args.lon - dlon, args.lat - dlat, args.lon + dlon, args.lat + dlat)

    if len(args.dem) == 1:
        return Terrain.from_file(args.dem[0], bounds=bounds, geoid=args.geoid)

    return Terrain.from_files(args.dem, bounds=bounds, geoid=args.geoid)


if __name__ == "__main__":
    raise SystemExit(main())
