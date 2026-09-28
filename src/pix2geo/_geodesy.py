"""WGS84 constants and geodetic helpers."""

from __future__ import annotations

import numpy as np
import pymap3d

from ._typing import ArrayLike, FloatArray

#: WGS84 semi-major axis in meters.
WGS84_A = 6378137.0

#: WGS84 flattening.
WGS84_F = 1.0 / 298.257223563

#: WGS84 semi-minor axis in meters.
WGS84_B = WGS84_A * (1.0 - WGS84_F)

#: WGS84 first eccentricity squared.
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def wrap_lon(lon: ArrayLike, center: float) -> FloatArray:
    """Wrap longitudes in degrees into the interval ``[center - 180, center + 180)``."""
    return center + np.mod(np.asarray(lon, float) - center + 180.0, 360.0) - 180.0


def geodetic_to_ecef(lat: ArrayLike, lon: ArrayLike, alt: ArrayLike) -> FloatArray:
    """WGS84 latitude, longitude (degrees) and height (m) to ECEF, shape ``(..., 3)``."""
    x, y, z = pymap3d.geodetic2ecef(lat, lon, alt)

    return np.stack([np.asarray(c, float) for c in (x, y, z)], axis=-1)


def ecef_to_geodetic(points: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """ECEF points ``(..., 3)`` to WGS84 latitude, longitude (degrees) and height (m)."""
    p = np.asarray(points, float)

    lat, lon, alt = pymap3d.ecef2geodetic(p[..., 0], p[..., 1], p[..., 2])

    return np.asarray(lat, float), np.asarray(lon, float), np.asarray(alt, float)


def ray_ellipsoid_intersection(
    origin: ArrayLike,
    direction: ArrayLike,
    height: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Intersect rays with the WGS84 ellipsoid grown by ``height`` meters.

    The grown ellipsoid has semi-axes ``a + h`` and ``b + h``. It differs
    from the true surface of constant geodetic height by less than 3 cm for
    ``|h| < 20 km``.

    Returns ``(t_near, t_far)`` in units of ``direction``. Both are NaN when
    the ray line misses the ellipsoid.
    """
    # Scale each axis so that the grown ellipsoid becomes the unit sphere. The
    # scaling is linear, so the ray parameter t does not change.
    scale = np.array([WGS84_A + height, WGS84_A + height, WGS84_B + height])
    o = np.asarray(origin, float) / scale
    d = np.asarray(direction, float) / scale

    # |o + t d|^2 = 1 gives the quadratic qa t^2 + qb t + qc = 0.
    qa = np.einsum("...i,...i->...", d, d)
    qb = 2.0 * np.einsum("...i,...i->...", o, d)
    qc = np.einsum("...i,...i->...", o, o) - 1.0
    disc = qb * qb - 4.0 * qa * qc

    # This form of the quadratic formula prevents cancellation.
    # The roots are q / qa and qc / q.
    with np.errstate(invalid="ignore", divide="ignore"):
        q = -0.5 * (qb + np.copysign(np.sqrt(disc), qb))
        r1 = q / qa
        r2 = qc / q

    # A negative (or NaN) discriminant means that the ray line misses.
    miss = ~(disc >= 0)
    t_near = np.where(miss, np.nan, np.fmin(r1, r2))
    t_far = np.where(miss, np.nan, np.fmax(r1, r2))

    return t_near, t_far
