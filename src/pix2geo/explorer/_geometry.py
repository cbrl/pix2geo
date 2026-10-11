"""Scene geometry of the explorer: the ENU frame, pixel sets and terrain windows.

The scene uses a local ENU frame: east, north and up, in meters. The origin is
the point on the WGS84 ellipsoid below the first camera position, so the up
coordinate is close to the ellipsoidal height near the camera. The frame is
flat and the earth is curved, so far points drop below their height (about
8 m at 10 km). The rays are straight lines in this frame, as in ECEF.
"""

from __future__ import annotations

import math

import numpy as np

from .._geodesy import geodetic_to_ecef
from .._typing import ArrayLike, Bounds, FloatArray, IntArray
from ..camera import CameraIntrinsics
from ..pose import ned_to_ecef_matrix
from ..terrain import Terrain

__all__ = [
    "Window",
    "bounds_window",
    "enu_frame",
    "image_border",
    "pixel_grid",
    "post_window",
    "to_enu",
]

#: A window of terrain posts: ``(col0, col1, row0, row1)``, inclusive.
Window = tuple[int, int, int, int]


def enu_frame(lat: float, lon: float) -> tuple[FloatArray, FloatArray]:
    """The ECEF origin and axes of the ENU frame on the ellipsoid at ``(lat, lon)``.

    The axes are the columns of a 3x3 matrix: east, north and up.
    """
    ned = ned_to_ecef_matrix(lat, lon)
    axes = np.column_stack([ned[:, 1], ned[:, 0], -ned[:, 2]])

    return geodetic_to_ecef(lat, lon, 0.0), axes


def to_enu(points: ArrayLike, origin: FloatArray, axes: FloatArray) -> FloatArray:
    """ECEF points ``(..., 3)`` to ENU, with the frame of :func:`enu_frame`."""
    # Row vectors: v @ axes gives the components of v along the axes.
    return (np.asarray(points, dtype=float) - origin) @ axes


def pixel_grid(camera: CameraIntrinsics, count: int) -> FloatArray:
    """At most ``count`` pixels on a regular grid over the image, shape ``(n, 2)``.

    The grid has the aspect ratio of the image. Each pixel is at the center
    of one tile of an even tiling of the image.
    """
    nu = int(np.clip(round(math.sqrt(count * camera.width / camera.height)), 1, count))
    nv = max(1, count // nu)

    # The image spans -0.5 to size - 0.5 on each axis.
    u = (np.arange(nu) + 0.5) * (camera.width / nu) - 0.5
    v = (np.arange(nv) + 0.5) * (camera.height / nv) - 0.5
    uu, vv = np.meshgrid(u, v)

    return np.stack([uu.ravel(), vv.ravel()], axis=-1)


def image_border(camera: CameraIntrinsics, edge_points: int) -> tuple[FloatArray, IntArray]:
    """A closed loop of pixels on the image edge, and the indices of the four corners.

    The loop runs clockwise from the top-left corner: the top, right, bottom
    and left edges. The last pixel repeats the first.
    """
    u0, u1 = -0.5, camera.width - 0.5
    v0, v1 = -0.5, camera.height - 0.5
    t = np.linspace(0.0, 1.0, edge_points, endpoint=False)
    n = edge_points

    # Each edge runs from one corner to the next, without the next corner.
    du, dv = (u1 - u0) * t, (v1 - v0) * t
    u = np.concatenate([u0 + du, np.full(n, u1), u1 - du, np.full(n, u0), [u0]])
    v = np.concatenate([np.full(n, v0), v0 + dv, np.full(n, v1), v1 - dv, [v0]])

    return np.stack([u, v], axis=-1), np.arange(4) * n


def post_window(
    terrain: Terrain,
    lon: FloatArray,
    lat: FloatArray,
    margin: float,
) -> Window | None:
    """The terrain posts around some points.

    The window grows by ``margin`` times its size plus one post, and it stops
    at the raster edge. None when the window has fewer than two posts on an axis.
    """
    x, y = terrain.lonlat_to_grid(lon, lat)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return None
    x, y = x[ok], y[ok]

    # Pad, round outward to whole posts, and clamp to the raster.
    pad = margin * max(np.ptp(x), np.ptp(y)) + 1.0
    rows, cols = terrain.shape
    c0, c1 = np.clip([math.floor(x.min() - pad), math.ceil(x.max() + pad)], 0, cols - 1)
    r0, r1 = np.clip([math.floor(y.min() - pad), math.ceil(y.max() + pad)], 0, rows - 1)
    if c1 <= c0 or r1 <= r0:
        return None

    return int(c0), int(c1), int(r0), int(r1)


def bounds_window(terrain: Terrain, bounds: Bounds) -> Window:
    """The terrain posts inside WGS84 ``(west, south, east, north)`` bounds."""
    # Points along the four edges of the box. The edges of the box can be
    # curves in a projected raster.
    w, s, e, n = bounds
    t = np.linspace(0.0, 1.0, 17)
    lon = np.concatenate([w + (e - w) * t, np.full_like(t, e), w + (e - w) * t, np.full_like(t, w)])
    lat = np.concatenate([np.full_like(t, n), s + (n - s) * t, np.full_like(t, s), s + (n - s) * t])

    window = post_window(terrain, lon, lat, margin=0.0)
    if window is None:
        raise ValueError("the bounds do not overlap the terrain")

    return window
