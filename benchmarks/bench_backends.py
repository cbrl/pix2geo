"""Compare the backends on a synthetic DEM and a full-frame image.

Usage::

    python benchmarks/bench_backends.py [--size 4096] [--width 1920] [--height 1080]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from affine import Affine

from pix2geo import (
    CameraIntrinsics,
    CameraPose,
    Geolocator,
    Terrain,
    available_backends,
)


def make_terrain(n: int) -> Terrain:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)

    h = 1200 + 400 * np.sin(xx / 211) * np.cos(yy / 157) + 80 * np.sin((xx - yy) / 29)
    h += rng.normal(0, 2, size=h.shape).astype(np.float32)

    res = 1.0 / 3600  # 1 arc-second posts, about 30 m
    return Terrain.from_array(h, Affine(res, 0, -105.5, 0, -res, 40.5), "EPSG:4326")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--python-rays", type=int, default=2000)

    args = parser.parse_args()

    t = time.perf_counter()
    terrain = make_terrain(args.size)
    print(
        f"terrain {terrain.shape}, {terrain.pyramid.num_levels} quadtree levels, "
        f"built in {time.perf_counter() - t:.2f} s"
    )

    cam = CameraIntrinsics.from_fov(args.width, args.height, hfov=60)
    poses = {
        "nadir": CameraPose.from_euler(40.0, -105.0, 4000.0, yaw=0, pitch=-90, roll=0),
        "oblique": CameraPose.from_euler(40.0, -105.0, 4000.0, yaw=30, pitch=-20, roll=0),
    }
    uu, vv = np.meshgrid(np.arange(cam.width, dtype=float), np.arange(cam.height, dtype=float))
    pixels = np.stack([uu, vv], -1).reshape(-1, 2)
    n = len(pixels)

    for view, pose in poses.items():
        print(f"\n{view} view, {n:,} rays")
        for name in available_backends():
            geo = Geolocator(terrain, backend=name)
            px = pixels

			# For the Python backend, we only test a subset of rays to avoid long runtimes.
            if name == "python":
                px = pixels[np.random.default_rng(1).choice(n, args.python_rays, replace=False)]

			# Warm up (JIT compile, GPU upload)
            geo.pixel_to_geo(cam, pose, px[:64])

            t = time.perf_counter()
            res = geo.pixel_to_geo(cam, pose, px)
            dt = time.perf_counter() - t

            rate = len(px) / dt
            print(f"  {name:7s} {dt:8.3f} s  {rate:12,.0f} rays/s  hit {res.hit.mean():.1%}")


if __name__ == "__main__":
    main()
