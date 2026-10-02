"""WGS84 constants and geodetic helpers."""

from __future__ import annotations

import math

import numpy as np

from ._typing import ArrayLike, FloatArray

#: WGS84 semi-major axis in meters.
WGS84_A = 6378137.0

#: WGS84 flattening.
WGS84_F = 1.0 / 298.257223563

#: WGS84 semi-minor axis in meters.
WGS84_B = WGS84_A * (1.0 - WGS84_F)

#: WGS84 first eccentricity squared.
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

#: WGS84 second eccentricity squared.
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)

# Radians in half a degree. It gives the half angles of the tangent form.
_HALF_DEGREE = math.pi / 360.0

# The types of one number. NumPy float64 is a subclass of float.
_NUMBER = (int, float)


def wrap_lon(lon: ArrayLike, center: float) -> FloatArray:
    """Wrap longitudes in degrees into the interval ``[center - 180, center + 180)``."""
    return center + np.mod(np.asarray(lon, float) - center + 180.0, 360.0) - 180.0


def geodetic_to_ecef(lat: ArrayLike, lon: ArrayLike, alt: ArrayLike) -> FloatArray:
    """WGS84 latitude, longitude (degrees) and height (m) to ECEF, shape ``(..., 3)``.

    The inputs broadcast against each other. Each angle needs one tan call and
    no sin or cos call: with t = tan(angle / 2), 1 + cos = 2 / (1 + t^2) and
    sin = t (1 + cos). The error is below 1e-8 m.
    """
    # One point is faster with Python floats than with NumPy calls. Infinite
    # angles go to the NumPy steps, because math.tan rejects them.
    if (
        isinstance(lat, _NUMBER)
        and isinstance(lon, _NUMBER)
        and isinstance(alt, _NUMBER)
        and math.isfinite(lat)
        and math.isfinite(lon)
    ):
        return np.array(_point_to_ecef(float(lat), float(lon), float(alt)))

    lat, lon, h = np.broadcast_arrays(
        np.asarray(lat, float),
        np.asarray(lon, float),
        np.asarray(alt, float),
    )
    shape = lat.shape

    # The steps work in place. On a large batch, each new array costs more
    # than a multiplication.
    out = np.empty((*shape, 3))

    # The sine and cosine of the latitude, from t = tan(lat / 2).
    sl = np.multiply(lat, _HALF_DEGREE, out=np.empty(shape))
    np.tan(sl, out=sl)
    cl = np.multiply(sl, sl, out=np.empty(shape))
    cl += 1.0
    np.divide(2.0, cl, out=cl)
    sl *= cl
    cl -= 1.0

    # The radius of curvature in the prime vertical, N = a / sqrt(1 - e2 sin^2(lat)).
    n = np.multiply(sl, sl, out=np.empty(shape))
    n *= -WGS84_E2
    n += 1.0
    np.sqrt(n, out=n)
    np.divide(WGS84_A, n, out=n)

    # The distance from the polar axis, r = (N + h) cos(lat), and
    # z = ((1 - e2) N + h) sin(lat).
    r = np.add(n, h, out=np.empty(shape))
    r *= cl
    n *= 1.0 - WGS84_E2
    n += h
    np.multiply(n, sl, out=out[..., 2])

    # The sine and cosine of the longitude, in the arrays of the latitude.
    # Then x = r cos(lon) and y = r sin(lon).
    so = np.multiply(lon, _HALF_DEGREE, out=sl)
    np.tan(so, out=so)
    co = np.multiply(so, so, out=cl)
    co += 1.0
    np.divide(2.0, co, out=co)
    so *= co
    co -= 1.0
    np.multiply(r, co, out=out[..., 0])
    np.multiply(r, so, out=out[..., 1])

    return out


def _point_to_ecef(lat: float, lon: float, h: float) -> tuple[float, float, float]:
    """``geodetic_to_ecef`` for one point, with the same steps."""
    t = math.tan(lat * _HALF_DEGREE)
    c1 = 2.0 / (1.0 + t * t)
    sl, cl = t * c1, c1 - 1.0
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    r = (n + h) * cl

    t = math.tan(lon * _HALF_DEGREE)
    c1 = 2.0 / (1.0 + t * t)

    return r * (c1 - 1.0), r * (t * c1), ((1.0 - WGS84_E2) * n + h) * sl


def ecef_to_geodetic(points: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """ECEF points ``(..., 3)`` to WGS84 latitude, longitude (degrees) and height (m).

    This is Bowring's method with two iterations, as in the kernel. The error
    is below 1e-8 m from 500 m below the surface up to 1000 km above it.
    """
    p = np.asarray(points, float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]

    # The distance from the polar axis, and the first guess of the reduced
    # latitude beta: tan(beta) = (a z) / (b r), as a sine and cosine pair.
    r = np.hypot(x, y)
    sn = WGS84_A * z
    cs = WGS84_B * r

    # Each iteration gets the latitude from beta, as tan(lat) = num / den, then
    # beta from the latitude, as tan(beta) = (b / a) tan(lat).
    for _ in range(2):
        inv = 1.0 / np.hypot(sn, cs)
        sb, cb = sn * inv, cs * inv
        num = z + (WGS84_EP2 * WGS84_B) * sb * sb * sb
        den = r - (WGS84_E2 * WGS84_A) * cb * cb * cb
        sn, cs = WGS84_B * num, WGS84_A * den

    # The height is along the ellipsoid normal. This form is stable at all
    # latitudes, also near the poles.
    inv = 1.0 / np.hypot(num, den)
    sl, cl = num * inv, den * inv
    h = r * cl + z * sl - WGS84_A * np.sqrt(1.0 - WGS84_E2 * sl * sl)

    return np.degrees(np.arctan2(num, den)), np.degrees(np.arctan2(y, x)), h


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
