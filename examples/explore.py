"""Open the interactive explorer from Python, with a camera that has lens distortion.

Run with your own DEM::

    python examples/explore.py path/to/dem.tif --geoid egm96

Without a path, the script uses the synthetic DEM of ``python -m pix2geo.explorer``.
The page opens in the default browser. Press Ctrl+C to stop the server. For all
the options of the explorer, run ``python -m pix2geo.explorer --help``.
"""

from __future__ import annotations

import argparse

from pix2geo import CameraIntrinsics, CameraPose, Geolocator, Terrain
from pix2geo.explorer import serve
from pix2geo.explorer.__main__ import demo_terrain


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])

    parser.add_argument("dem", nargs="?", help="elevation raster (default: synthetic)")
    parser.add_argument("--geoid", default=None, help="egm96, egm2008, or a grid file")

    args = parser.parse_args()

    terrain = demo_terrain() if args.dem is None else Terrain.from_file(args.dem, geoid=args.geoid)

    # A wide lens with barrel distortion, 300 m above the DEM center, looking
    # at the horizon. The fallback surface catches the rays that leave the DEM.
    w, s, e, n = terrain.bounds_wgs84()
    cam = CameraIntrinsics.from_fov(1920, 1080, hfov=90, dist=[-0.25, 0.07, 0.0, 0.0, -0.01])
    pose = CameraPose.from_euler(
        (s + n) / 2,
        (w + e) / 2,
        300.0,
        yaw=135.0,
        pitch=-6.0,
        roll=0.0,
        alt_ref="agl",
    )

    serve(Geolocator(terrain), cam, pose, fallback_height=terrain.h_min, max_rays=400)


if __name__ == "__main__":
    main()
