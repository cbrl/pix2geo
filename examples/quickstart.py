"""Geolocate image pixels on a DEM.

Run with your own DEM::

    python examples/quickstart.py path/to/dem.tif --geoid egm96

Without a path, the script makes a synthetic DEM near Boulder, Colorado.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from affine import Affine

from pix2geo import CameraIntrinsics, CameraPose, Geolocator, Terrain, available_backends


def synthetic_dem(path: Path) -> Path:
    import rasterio

    n = 1800
    res = 1.0 / 3600  # 1 arc-second, about 30 m
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    h = 1700 + 300 * np.sin(xx / 150) * np.cos(yy / 210) + 40 * np.sin((xx + 2 * yy) / 23)
    transform = Affine(res, 0, -105.5, 0, -res, 40.25)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=n,
        height=n,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as ds:
        ds.write(h.astype(np.float32), 1)

    return path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("dem", nargs="?", help="elevation raster (default: synthetic)")
    parser.add_argument("--geoid", default=None, help="egm96, egm2008, grid file or a number")

    args = parser.parse_args()

    if args.dem is None:
        dem = synthetic_dem(Path(tempfile.mkdtemp()) / "synthetic_dem.tif")
    else:
        dem = Path(args.dem)

    terrain = Terrain.from_file(dem, geoid=args.geoid)
    print(terrain)

    # The camera is above the center of the DEM.
    w, s, e, n = terrain.bounds_wgs84()
    lat, lon = (s + n) / 2, (w + e) / 2

    # A 4K camera, 1000 m above the terrain, looking north-east and down 30 degrees.
    cam = CameraIntrinsics.from_fov(3840, 2160, hfov=65)
    pose = CameraPose.from_euler(lat, lon, 1000.0, yaw=45, pitch=-30, roll=0, alt_ref="agl")
    geo = Geolocator(terrain, backend="auto")
    print("backends on this system:", available_backends())

    pixels = np.array(
        [
            [cam.cx, cam.cy],
            [0, 0],
            [cam.width - 1, 0],
            [0, cam.height - 1],
            [cam.width - 1, cam.height - 1],
        ]
    )
    res = geo.pixel_to_geo(cam, pose, pixels)
    print(f"\n{'pixel':>16} {'lat':>12} {'lon':>13} {'alt [m]':>9} {'range [m]':>10}")
    for (u, v), la, lo, al, rg in zip(pixels, res.lat, res.lon, res.alt, res.range):
        print(f"({u:6.1f},{v:6.1f}) {la:12.7f} {lo:13.7f} {al:9.2f} {rg:10.1f}")

    # Project the hits back into the image: this checks the round trip.
    u, v, visible = geo.geo_to_pixel(cam, pose, res.lat, res.lon, res.alt)
    err = np.hypot(u - pixels[:, 0], v - pixels[:, 1])
    print(f"\nround-trip pixel error: max {np.nanmax(err):.2e} px")

    # A whole frame, every 4th pixel.
    full = geo.image_to_geo(cam, pose, step=4)
    print(f"full frame: {full.lat.size:,} rays, {full.hit.mean():.1%} hit the terrain")


if __name__ == "__main__":
    main()
