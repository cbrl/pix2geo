"""Ray / heightfield intersection kernel, shared by all the backends.

The kernel is plain Python in the subset that both Numba and ``cupyx.jit``
compile. The function :func:`build_kernels` makes the kernel with a given
``jit`` decorator, so all the backends run the same algorithm:

* The Python backend passes an identity decorator.
* The Numba backend passes ``numba.njit``.
* The CuPy backend passes ``cupyx.jit.rawkernel(device=True)``. It also
  passes CuPy ufuncs in place of the ``math`` module, because ``cupyx.jit``
  cannot compile ``math`` functions. See :class:`KernelMath`.

Algorithm
---------
Each ray is an exact straight line in ECEF. The kernel clips it to the
height band of the terrain (two ray / ellipsoid intersections) and cuts the
rest into equal segments (see :func:`make_ray_limits`). For each segment,
the kernel:

1. Converts the segment end point from ECEF to geodetic coordinates
   (Bowring's method, two iterations, sub-micrometre accurate near the
   Earth surface), then to raster grid coordinates ``(x, y)``. The third
   grid coordinate ``z`` is the ellipsoidal height.
2. Walks the max quadtree (:class:`~pix2geo.quadtree.HeightPyramid`) for the
   grid-space segment, depth-first and near-to-far:

   a. Clip the segment against the node footprint (slab test).
   b. Skip the node when the lowest point of the clipped segment is above
      the node maximum. This rejects large empty areas in one step.
   c. At a leaf cell, solve the exact segment / bilinear-patch
      intersection. This is a quadratic equation in the segment parameter.

The visit order is near-to-far, so the first leaf hit is the nearest hit.
The kernel maps the hit back onto the exact ECEF ray parameter, and converts
the hit point to geodetic coordinates.

Step 1 uses float64. Step 2 uses the number types ``real`` and ``index``
that the backend gives (float32 and int32 on the GPU), with grid coordinates
relative to a grid point near the segment start. Small coordinates keep the
float32 error small, also on large rasters.

Grid mapping
------------
``params`` is a float64 vector that :func:`make_grid_params` makes:

* ``params[P_AFFINE:P_AFFINE + 6]``: affine ``(A, B, C, D, E, F)`` from
  (lon, lat) in degrees to fractional indices ``u = A*lon + B*lat + C`` and
  ``v = D*lon + E*lat + F``.
* ``params[P_WRAP]``: centre longitude for wrap-around, or NaN for no wrap.
* ``params[P_MODE]``: ``MODE_AFFINE`` when ``(u, v)`` are grid coordinates.
  ``MODE_LUT`` when ``(u, v)`` address the lookup tables ``lut_x`` and
  ``lut_y``, which hold grid coordinates on a regular lon/lat lattice (for
  projected CRSs).
* ``params[P_A]``, ``params[P_B]``, ``params[P_E2]``, ``params[P_EP2]``:
  ellipsoid semi-major and semi-minor axes, and first and second
  eccentricity squared. The kernel gets all four, so it does not compute
  ``b`` and ``ep2`` for each point.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar

import numpy as np

from .._geodesy import WGS84_A, WGS84_B, WGS84_E2, WGS84_EP2
from .._typing import (
    FloatArray,
    GridLut,
    GridParams,
    HeightGrid,
    LevelOffsets,
    LevelShapes,
    NodeStack,
    PyramidData,
    RayLimits,
)

if TYPE_CHECKING:
    from ..terrain import Terrain

_F = TypeVar("_F", bound=Callable[..., Any])


class Decorator(Protocol):
    """A function decorator that keeps the signature, such as ``numba.njit(...)``."""

    def __call__(self, func: _F, /) -> _F: ...


class KernelMath(Protocol):
    """The math functions that the kernel calls. The ``math`` module has all of them."""

    @property
    def nan(self) -> float: ...

    def sqrt(self, x: float, /) -> float: ...

    def atan2(self, y: float, x: float, /) -> float: ...

    def degrees(self, x: float, /) -> float: ...

    def isfinite(self, x: float, /) -> bool: ...


#: Signature of the ``trace_rays`` kernel: terrain arrays, the ray limits,
#: the ray batch, then the ``(4, n)`` output array for the hits.
TraceRaysKernel = Callable[
    [
        HeightGrid,
        PyramidData,
        LevelOffsets,
        LevelShapes,
        GridParams,
        GridLut,
        GridLut,
        RayLimits,
        FloatArray,
        FloatArray,
        FloatArray,
    ],
    None,
]


class Kernels(NamedTuple):
    """The kernel functions that :func:`build_kernels` makes."""

    bilinear_hit: Callable[..., float]
    clip_slab: Callable[..., tuple[float, float]]
    trace_segment: Callable[..., float]
    ecef_to_geodetic: Callable[..., tuple[float, float, float]]
    ecef_to_grid: Callable[..., tuple[float, float, float]]
    ellipsoid_hits: Callable[..., tuple[float, float]]
    clip_to_band: Callable[..., tuple[float, float]]
    trace_one: Callable[..., float]
    trace_ray: Callable[..., None]
    trace_rays: TraceRaysKernel


# Layout of the ``params`` vector.
P_AFFINE = 0
P_WRAP = 6
P_MODE = 7
P_A = 8
P_B = 9
P_E2 = 10
P_EP2 = 11
N_PARAMS = 12

MODE_AFFINE = 0.0
MODE_LUT = 1.0

# Layout of the ``limits`` vector.
L_H_MIN = 0
L_H_MAX = 1
L_T_MAX = 2
L_SEGMENT = 3
N_LIMITS = 4

#: The height band of the terrain grows by this margin (meters) on each side.
#: It covers the difference between a grown ellipsoid and a surface of
#: constant height (less than 3 cm below 20 km).
BAND_MARGIN = 1.0


def make_grid_params(
    affine: Sequence[float],
    wrap_center: float | None = None,
    use_lut: bool = False,
) -> GridParams:
    """Pack the (lon, lat) to grid mapping into the ``params`` vector."""
    params = np.empty(N_PARAMS)
    params[P_AFFINE : P_AFFINE + 6] = affine
    params[P_WRAP] = np.nan if wrap_center is None else wrap_center
    params[P_MODE] = MODE_LUT if use_lut else MODE_AFFINE
    params[P_A] = WGS84_A
    params[P_B] = WGS84_B
    params[P_E2] = WGS84_E2
    params[P_EP2] = WGS84_EP2
    return params


def make_ray_limits(
    terrain: Terrain,
    max_range: float | None,
    max_segment_length: float,
) -> RayLimits:
    """Pack the ray clip settings into the ``limits`` vector.

    * ``limits[L_H_MIN]``, ``limits[L_H_MAX]``: the height band of the
      terrain, with :data:`BAND_MARGIN`. The kernel clips each ray to it.
    * ``limits[L_T_MAX]``: the largest ray parameter (``max_range``), or
      infinity for no limit.
    * ``limits[L_SEGMENT]``: the longest segment (``max_segment_length``).
    """
    limits = np.empty(N_LIMITS)
    limits[L_H_MIN] = terrain.h_min - BAND_MARGIN
    limits[L_H_MAX] = terrain.h_max + BAND_MARGIN
    limits[L_T_MAX] = math.inf if max_range is None else max_range
    limits[L_SEGMENT] = max_segment_length

    return limits


def stack_size(num_levels: int) -> int:
    """Traversal stack entries that a quadtree with ``num_levels`` levels needs.

    Each pop pushes at most 4 children, so the depth-first stack never holds
    more than ``3 * num_levels + 1`` nodes.
    """
    return 4 * num_levels + 4


def run_cpu_kernel(
    trace_rays: TraceRaysKernel,
    terrain: Terrain,
    orig: FloatArray,
    dirs: FloatArray,
    limits: RayLimits,
) -> FloatArray:
    """Call a CPU ``trace_rays`` kernel for one ray batch and return the ``(4, n)`` hits."""
    params, lut_x, lut_y = terrain.grid_mapping()
    pyr = terrain.pyramid
    out = np.empty((4, len(dirs)))
    trace_rays(
        terrain.heights, pyr.data, pyr.offsets, pyr.shapes, params, lut_x, lut_y,
        limits, orig, dirs, out,
    )  # fmt: skip

    return out


def build_kernels(
    jit: Decorator,
    jit_parallel: Decorator,
    prange: Callable[[int], Iterable[int]],
    math: KernelMath = math,
    real: Callable[[Any], float] = float,
    index: Callable[[Any], int] = int,
) -> Kernels:
    """Make the kernel functions with the given decorators and parallel range.

    ``jit`` compiles the per-ray functions. ``jit_parallel`` compiles
    ``trace_rays``, whose ray loop uses ``prange``. ``math`` gives the math
    functions. The parameter has the name of the module that it replaces, so
    the kernel code reads as plain Python.

    ``real`` and ``index`` are the float and integer types of the quadtree
    walk (``trace_segment`` and ``bilinear_hit``). The CPU backends use
    ``float`` and ``int`` (64 bits). The CuPy backend uses float32 and int32,
    because consumer GPUs run 64-bit math much more slowly. The integer
    arrays of the walk (``shp`` and ``stack``) should have the type
    ``index`` too. The geodetic math always uses float64.
    """

    @jit
    def bilinear_hit(
        hgt: HeightGrid,
        i: int,
        j: int,
        u0: float,
        v0: float,
        z0: float,
        dx: float,
        dy: float,
        dz: float,
        ta: float,
        tb: float,
    ) -> float:
        # Smallest t in [ta, tb] where the segment (u0, v0, z0) + t (dx, dy, dz)
        # is on or below the bilinear patch of cell (i, j). (u0, v0) is the
        # segment start relative to the cell corner. Returns -1.0 if there is none.

        # The four corner posts of the cell.
        h00 = real(hgt[i, j])
        h10 = real(hgt[i, j + 1])
        h01 = real(hgt[i + 1, j])
        h11 = real(hgt[i + 1, j + 1])

        # The patch is H(u, v) = h00 + b u + c v + d u v, where (u, v) is the
        # position in the cell.
        b = h10 - h00
        c = h01 - h00
        d = h00 - h10 - h01 + h11

        # The height of the segment above the patch is a quadratic in t:
        # f(t) = z(t) - H(u(t), v(t)) = qa t^2 + qb t + qc
        qa = -d * dx * dy
        qb = dz - b * dx - c * dy - d * (u0 * dy + v0 * dx)
        qc = z0 - h00 - b * u0 - c * v0 - d * u0 * v0

        # The segment is already on or below the patch at ta.
        fa = (qa * ta + qb) * ta + qc
        if fa <= 0.0:
            return ta

        # Find the smallest root of f in [ta, tb].
        best = -1.0
        if qa == 0.0:
            # f is linear: the patch is a plane, or the segment moves along one grid axis.
            if qb != 0.0:
                t = -qc / qb
                if t >= ta and t <= tb:
                    best = t
        else:
            # This form of the quadratic formula prevents cancellation.
            # The roots are q / qa and qc / q.
            disc = qb * qb - 4.0 * qa * qc
            if disc >= 0.0:
                sq = math.sqrt(disc)
                q = -0.5 * (qb + sq) if qb >= 0.0 else -0.5 * (qb - sq)
                r1 = q / qa
                r2 = qc / q if q != 0.0 else r1
                if r1 > r2:
                    r1, r2 = r2, r1
                if r1 >= ta and r1 <= tb:
                    best = r1
                elif r2 >= ta and r2 <= tb:
                    best = r2

        if best < 0.0:
            # Round-off can hide a root at tb, so test the end point too.
            fb = (qa * tb + qb) * tb + qc
            if fb <= 0.0:
                best = tb

        return best

    @jit
    def clip_slab(
        p0: float, dp: float, lo: float, hi: float, ta: float, tb: float
    ) -> tuple[float, float]:
        # Clip the parameter interval [ta, tb] of one coordinate p0 + t * dp to
        # the slab lo <= p <= hi. An empty result has ta > tb.
        if dp != 0.0:
            # The parameters where the line crosses lo and hi, in increasing order.
            t0 = (lo - p0) / dp
            t1 = (hi - p0) / dp
            if t0 > t1:
                t0, t1 = t1, t0

            # Keep the part of [ta, tb] between the two crossings.
            if t0 > ta:
                ta = t0
            if t1 < tb:
                tb = t1
        elif p0 < lo or p0 > hi:
            # The line is parallel to the slab and outside it.
            return 1.0, 0.0

        return ta, tb

    @jit
    def trace_segment(
        hgt: HeightGrid,
        pyr: PyramidData,
        offs: LevelOffsets,
        shp: LevelShapes,
        row0: int,
        col0: int,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        stack: NodeStack,
    ) -> float:
        # First hit parameter t in [0, 1] on the grid-space segment from
        # (x0, y0, z0) to (x1, y1, z1), or -1.0. The caller owns the stack.
        # (x, y) are grid coordinates relative to the grid point (col0, row0).
        # An origin near the segment keeps them small, so float32 keeps its
        # precision on large rasters.
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        # Level 0 has one node for each cell. Edge node footprints clip to this size.
        rows = shp[0, 0]
        cols = shp[0, 1]

        # Row and column offset of the child that the segment enters first.
        near_i = 0 if dy >= 0.0 else 1
        near_j = 0 if dx >= 0.0 else 1

        # Start at the root node, which covers the whole raster.
        stack[0, 0] = shp.shape[0] - 1
        stack[0, 1] = 0
        stack[0, 2] = 0
        sp = 1
        while sp > 0:
            # Pop the next node: level k, row i, column j.
            sp -= 1
            k = stack[sp, 0]
            i = stack[sp, 1]
            j = stack[sp, 2]

            # The footprint of the node: 2^k x 2^k cells, relative to the origin.
            # The integer subtraction is exact.
            xlo = real((j << k) - col0)
            xhi = real(min((j + 1) << k, cols) - col0)
            ylo = real((i << k) - row0)
            yhi = real(min((i + 1) << k, rows) - row0)

            # Clip the segment to the footprint. Skip the node if the segment misses it.
            ta, tb = clip_slab(x0, dx, xlo, xhi, 0.0, 1.0)
            ta, tb = clip_slab(y0, dy, ylo, yhi, ta, tb)
            if ta > tb:
                continue

            # The lowest point of a straight segment is at one of its ends. Skip
            # the node if that point is above the highest terrain in the node.
            za = z0 + dz * ta
            zb = z0 + dz * tb
            zmin = za if za < zb else zb
            if zmin > pyr[offs[k] + i * shp[k, 1] + j]:
                continue

            # A leaf is one cell: do the exact test on its bilinear patch. The
            # footprint corner (xlo, ylo) is the cell corner.
            if k == 0:
                t = bilinear_hit(hgt, i, j, x0 - xlo, y0 - ylo, z0, dx, dy, dz, ta, tb)
                if t >= 0.0:
                    return t
                continue

            # Push the children far-to-near, so the near child pops first. n = 3
            # is the near child and n = 0 is the far child. A straight segment
            # crosses at most one of the other two, so their order has no effect.
            child_rows = shp[k - 1, 0]
            child_cols = shp[k - 1, 1]
            for n in range(4):
                ci = 2 * i + (near_i if n >= 2 else 1 - near_i)
                cj = 2 * j + (near_j if n % 2 == 1 else 1 - near_j)
                if ci < child_rows and cj < child_cols:
                    stack[sp, 0] = k - 1
                    stack[sp, 1] = ci
                    stack[sp, 2] = cj
                    sp += 1

        return -1.0

    @jit
    def ecef_to_geodetic(
        x: float,
        y: float,
        z: float,
        a: float,
        b: float,
        e2: float,
        ep2: float,
    ) -> tuple[float, float, float]:
        # Bowring's method with two iterations. Returns the latitude and the
        # longitude in radians and the height in meters. The ellipsoid has the
        # semi-axes a and b, and the first and second eccentricity squared e2
        # and ep2. Each angle comes from its tangent, as a sine and cosine pair.
        # Only the final angles need atan2, so the method uses no sin or cos.
        # Those are slow in float64 on GPUs.

        # The distance from the polar axis.
        p = math.sqrt(x * x + y * y)

        # The first guess of the reduced latitude beta comes from the point as if
        # it were on the ellipsoid: tan(beta) = (a z) / (b p).
        sn = a * z
        cs = b * p
        inv = 1.0 / math.sqrt(sn * sn + cs * cs)
        sb = sn * inv
        cb = cs * inv

        # Iteration 1: the latitude from beta, as tan(lat) = num / den. Then
        # beta from the latitude, as tan(beta) = (b / a) tan(lat).
        num = z + ep2 * b * sb * sb * sb
        den = p - e2 * a * cb * cb * cb
        sn = b * num
        cs = a * den
        inv = 1.0 / math.sqrt(sn * sn + cs * cs)
        sb = sn * inv
        cb = cs * inv

        # Iteration 2: the final latitude, with its sine and cosine.
        num = z + ep2 * b * sb * sb * sb
        den = p - e2 * a * cb * cb * cb
        inv = 1.0 / math.sqrt(num * num + den * den)
        sl = num * inv
        cl = den * inv

        # The longitude is exact. The height is along the ellipsoid normal. This
        # form is stable at all latitudes, also near the poles.
        lat = math.atan2(num, den)
        lon = math.atan2(y, x)
        h = p * cl + z * sl - a * math.sqrt(1.0 - e2 * sl * sl)

        return lat, lon, h

    @jit
    def ecef_to_grid(
        x: float,
        y: float,
        z: float,
        params: GridParams,
        lut_x: GridLut,
        lut_y: GridLut,
    ) -> tuple[float, float, float]:
        # ECEF point to grid coordinates (x, y) and ellipsoidal height.

        # Geodetic latitude and longitude in degrees.
        lat, lon, h = ecef_to_geodetic(
            x, y, z, params[P_A], params[P_B], params[P_E2], params[P_EP2]
        )
        lat = math.degrees(lat)
        lon = math.degrees(lon)

        c = params[P_WRAP]
        if c == c:  # not NaN: wrap longitude to within 180 deg of c
            lon = c + (lon - c + 180.0) % 360.0 - 180.0

        # The affine map from (lon, lat) to (u, v). In affine mode, (u, v) are
        # the grid coordinates.
        u = params[P_AFFINE] * lon + params[P_AFFINE + 1] * lat + params[P_AFFINE + 2]
        v = params[P_AFFINE + 3] * lon + params[P_AFFINE + 4] * lat + params[P_AFFINE + 5]
        if params[P_MODE] == MODE_AFFINE:
            return u, v, h

        # In LUT mode, (u, v) are fractional table indices. Points outside the
        # tables have no grid coordinates.
        rows = lut_x.shape[0]
        cols = lut_x.shape[1]
        if not (u >= 0.0 and u <= cols - 1 and v >= 0.0 and v <= rows - 1):
            return math.nan, math.nan, h

        # The table cell (i, j) that holds (u, v), and the position in it.
        j = min(int(u), cols - 2)
        i = min(int(v), rows - 2)
        fu = u - j
        fv = v - i

        # Bilinear interpolation: along u on the top and bottom rows, then along v.
        top = lut_x[i, j] * (1.0 - fu) + lut_x[i, j + 1] * fu
        bottom = lut_x[i + 1, j] * (1.0 - fu) + lut_x[i + 1, j + 1] * fu
        gx = top * (1.0 - fv) + bottom * fv
        top = lut_y[i, j] * (1.0 - fu) + lut_y[i, j + 1] * fu
        bottom = lut_y[i + 1, j] * (1.0 - fu) + lut_y[i + 1, j + 1] * fu
        gy = top * (1.0 - fv) + bottom * fv

        return gx, gy, h

    @jit
    def ellipsoid_hits(
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        a: float,
        b: float,
    ) -> tuple[float, float]:
        # The ray parameters (t_near, t_far) where the line (ox, oy, oz) + t (dx, dy, dz)
        # meets the ellipsoid with the semi-axes (a, a, b). NaN when it misses.

        # Scale each axis so that the ellipsoid becomes the unit sphere. Then
        # |o + t d|^2 = 1 gives the quadratic qa t^2 + qb t + qc = 0.
        ia = 1.0 / (a * a)
        ib = 1.0 / (b * b)
        qa = (dx * dx + dy * dy) * ia + dz * dz * ib
        qb = 2.0 * ((ox * dx + oy * dy) * ia + oz * dz * ib)
        qc = (ox * ox + oy * oy) * ia + oz * oz * ib - 1.0
        disc = qb * qb - 4.0 * qa * qc
        if not disc >= 0.0:
            return math.nan, math.nan

        # This form of the quadratic formula prevents cancellation.
        # The roots are q / qa and qc / q.
        sq = math.sqrt(disc)
        q = -0.5 * (qb + sq) if qb >= 0.0 else -0.5 * (qb - sq)
        r1 = q / qa
        r2 = qc / q if q != 0.0 else r1
        if r1 > r2:
            r1, r2 = r2, r1

        return r1, r2

    @jit
    def clip_to_band(
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        params: GridParams,
        limits: RayLimits,
    ) -> tuple[float, float]:
        # The ray parameter interval [t0, t1] inside the height band of the
        # terrain and inside the largest range. This removes the empty space
        # above the terrain, which can be hundreds of kilometers for a
        # satellite. The interval is empty (not t1 > t0) when the ray misses
        # the band.

        # The band is the shell between two grown ellipsoids.
        a = params[P_A]
        b = params[P_B]
        h_min = limits[L_H_MIN]
        h_max = limits[L_H_MAX]
        top_in, top_out = ellipsoid_hits(ox, oy, oz, dx, dy, dz, a + h_max, b + h_max)
        bottom_in, _ = ellipsoid_hits(ox, oy, oz, dx, dy, dz, a + h_min, b + h_min)

        # Start where the ray enters the top ellipsoid, or at the camera when the
        # camera is already inside it.
        t0 = top_in if top_in > 0.0 else 0.0

        # End where the ray leaves the top ellipsoid, or earlier where it enters
        # the bottom ellipsoid in front of the camera. NaN (a miss) compares False.
        t1 = top_out
        if bottom_in > 0.0 and bottom_in < t1:
            t1 = bottom_in
        if limits[L_T_MAX] < t1:
            t1 = limits[L_T_MAX]

        return t0, t1

    @jit
    def trace_one(
        hgt: HeightGrid,
        pyr: PyramidData,
        offs: LevelOffsets,
        shp: LevelShapes,
        params: GridParams,
        lut_x: GridLut,
        lut_y: GridLut,
        limits: RayLimits,
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        stack: NodeStack,
    ) -> float:
        # Ray parameter of the first hit of the ECEF ray (ox, oy, oz) + t (dx, dy, dz),
        # t >= 0, or NaN.

        # Clip the ray to the height band. Cut the rest into the fewest equal
        # segments that are each at most limits[L_SEGMENT] long.
        t_start, t_end = clip_to_band(ox, oy, oz, dx, dy, dz, params, limits)
        if not t_end > t_start:
            return math.nan
        nseg = int((t_end - t_start) / limits[L_SEGMENT]) + 1

        # The raster size in cells.
        rows = shp[0, 0]
        cols = shp[0, 1]

        # Grid coordinates of the start of the first segment. Each segment end
        # is the start of the next segment, so each point converts only once.
        step = (t_end - t_start) / nseg
        ta = t_start
        ax, ay, az = ecef_to_grid(ox + ta * dx, oy + ta * dy, oz + ta * dz, params, lut_x, lut_y)

        for k in range(nseg):
            # The last segment ends exactly at t_end, with no round-off.
            tb = t_end if k == nseg - 1 else t_start + (k + 1) * step
            bx, by, bz = ecef_to_grid(
                ox + tb * dx, oy + tb * dy, oz + tb * dz, params, lut_x, lut_y
            )

            # Skip a segment with an end outside the grid mapping (NaN), and a
            # segment with both ends on the same outer side of the raster. Thus
            # the walk only gets coordinates near the raster, which fit in int32.
            finite = (
                math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by)
            )
            overlaps = (
                (ax >= 0.0 or bx >= 0.0) and (ax <= cols or bx <= cols)
                and (ay >= 0.0 or by >= 0.0) and (ay <= rows or by <= rows)
            )  # fmt: skip
            if finite and overlaps:
                # The walk uses grid coordinates relative to the grid point next
                # to the segment start, in the number types of the walk.
                col0 = index(ax)
                row0 = index(ay)
                t = trace_segment(
                    hgt, pyr, offs, shp, row0, col0,
                    real(ax - col0), real(ay - row0), real(az),
                    real(bx - col0), real(by - row0), real(bz),
                    stack,
                )  # fmt: skip

                # The walk gives the hit as a fraction of the segment. Convert it
                # to the ray parameter.
                if t >= 0.0:
                    return ta + t * (tb - ta)

            ta = tb
            ax = bx
            ay = by
            az = bz

        return math.nan

    @jit
    def trace_ray(
        hgt: HeightGrid,
        pyr: PyramidData,
        offs: LevelOffsets,
        shp: LevelShapes,
        params: GridParams,
        lut_x: GridLut,
        lut_y: GridLut,
        limits: RayLimits,
        orig: FloatArray,
        dirs: FloatArray,
        r: int | np.int64,
        stack: NodeStack,
        out: FloatArray,
    ) -> None:
        # Trace ray r of the batch. Write the ray parameter of its hit to out[0, r],
        # and the latitude, longitude (degrees) and height of the hit point to
        # out[1:4, r]. A miss gives NaN.

        # orig has one row for each ray, or one row for all the rays.
        i = min(r, orig.shape[0] - 1)
        ox = orig[i, 0]
        oy = orig[i, 1]
        oz = orig[i, 2]
        dx = dirs[r, 0]
        dy = dirs[r, 1]
        dz = dirs[r, 2]
        t = trace_one(
            hgt, pyr, offs, shp, params, lut_x, lut_y, limits, ox, oy, oz, dx, dy, dz, stack
        )

        # The geodetic coordinates of the hit point.
        lat = math.nan
        lon = math.nan
        h = math.nan
        if t == t:  # not NaN
            lat, lon, h = ecef_to_geodetic(
                ox + t * dx, oy + t * dy, oz + t * dz,
                params[P_A], params[P_B], params[P_E2], params[P_EP2],
            )  # fmt: skip
            lat = math.degrees(lat)
            lon = math.degrees(lon)

        out[0, r] = t
        out[1, r] = lat
        out[2, r] = lon
        out[3, r] = h

    @jit_parallel
    def trace_rays(
        hgt: HeightGrid,
        pyr: PyramidData,
        offs: LevelOffsets,
        shp: LevelShapes,
        params: GridParams,
        lut_x: GridLut,
        lut_y: GridLut,
        limits: RayLimits,
        orig: FloatArray,
        dirs: FloatArray,
        out: FloatArray,
    ) -> None:
        # Ray r is orig[r] + t * dirs[r] for t >= 0. The hits go to out[:, r].
        n = dirs.shape[0]
        size = 4 * shp.shape[0] + 4  # stack_size(num_levels)

        # Each ray has its own traversal stack, so the rays can run in parallel.
        # The prange index of Numba is unsigned, and Numba mixes unsigned and
        # signed integers into float64. Thus the index becomes int64.
        for r in prange(n):
            stack = np.empty((size, 3), np.int64)
            ray = np.int64(r)
            trace_ray(
                hgt, pyr, offs, shp, params, lut_x, lut_y, limits, orig, dirs, ray, stack, out
            )

    return Kernels(
        bilinear_hit=bilinear_hit,
        clip_slab=clip_slab,
        trace_segment=trace_segment,
        ecef_to_geodetic=ecef_to_geodetic,
        ecef_to_grid=ecef_to_grid,
        ellipsoid_hits=ellipsoid_hits,
        clip_to_band=clip_to_band,
        trace_one=trace_one,
        trace_ray=trace_ray,
        trace_rays=trace_rays,
    )
