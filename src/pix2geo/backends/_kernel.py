"""Ray / heightfield intersection kernel, shared by the CPU backends.

The kernel is plain Python in the subset that Numba compiles. The function
:func:`build_kernels` makes the kernel with a given ``jit`` decorator. The
Numba backend passes ``numba.njit``. The Python backend passes an identity
decorator, so both backends run the same algorithm. The CuPy backend is a
line-by-line CUDA port.

Algorithm
---------
Each ray is an exact straight line in ECEF, clipped on the host to the
height band of the terrain and cut into ``nseg`` equal segments. For each
segment, the kernel:

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
The kernel maps the hit back onto the exact ECEF ray parameter.

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
* ``params[P_A]``, ``params[P_E2]``: ellipsoid semi-major axis and first
  eccentricity squared.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar

import numpy as np

from .._geodesy import WGS84_A, WGS84_E2
from .._typing import (
    FloatArray,
    GridLut,
    GridParams,
    HeightGrid,
    IntArray,
    LevelOffsets,
    LevelShapes,
    NodeStack,
    PyramidData,
)

if TYPE_CHECKING:
    from ..terrain import Terrain

_F = TypeVar("_F", bound=Callable[..., Any])


class Decorator(Protocol):
    """A function decorator that keeps the signature, such as ``numba.njit(...)``."""

    def __call__(self, func: _F, /) -> _F: ...


#: Signature of the ``trace_rays`` kernel: terrain arrays, then the ray batch,
#: then the output array for the hit parameters.
TraceRaysKernel = Callable[
    [
        HeightGrid,
        PyramidData,
        LevelOffsets,
        LevelShapes,
        GridParams,
        GridLut,
        GridLut,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        IntArray,
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
    trace_one: Callable[..., float]
    trace_rays: TraceRaysKernel


# Layout of the ``params`` vector.
P_AFFINE = 0
P_WRAP = 6
P_MODE = 7
P_A = 8
P_E2 = 9
N_PARAMS = 10

MODE_AFFINE = 0.0
MODE_LUT = 1.0


def make_grid_params(
    affine: Sequence[float], wrap_center: float | None = None, use_lut: bool = False
) -> GridParams:
    """Pack the (lon, lat) to grid mapping into the ``params`` vector."""
    params = np.empty(N_PARAMS)
    params[P_AFFINE : P_AFFINE + 6] = affine
    params[P_WRAP] = np.nan if wrap_center is None else wrap_center
    params[P_MODE] = MODE_LUT if use_lut else MODE_AFFINE
    params[P_A] = WGS84_A
    params[P_E2] = WGS84_E2
    return params


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
    t0: FloatArray,
    t1: FloatArray,
    nseg: IntArray,
) -> FloatArray:
    """Call a CPU ``trace_rays`` kernel for one ray batch and return the hit parameters."""
    params, lut_x, lut_y = terrain.grid_mapping()
    pyr = terrain.pyramid
    out = np.empty(len(orig))
    trace_rays(
        terrain.heights, pyr.data, pyr.offsets, pyr.shapes, params, lut_x, lut_y,
        orig, dirs, t0, t1, nseg, out,
    )  # fmt: skip

    return out


def build_kernels(
    jit: Decorator,
    jit_parallel: Decorator,
    prange: Callable[[int], Iterable[int]],
) -> Kernels:
    """Make the kernel functions with the given decorators and parallel range.

    ``jit`` compiles the per-ray functions. ``jit_parallel`` compiles
    ``trace_rays``, whose ray loop uses ``prange``.
    """

    @jit
    def bilinear_hit(
        hgt: HeightGrid,
        i: int,
        j: int,
        x0: float,
        y0: float,
        z0: float,
        dx: float,
        dy: float,
        dz: float,
        ta: float,
        tb: float,
    ) -> float:
        # Smallest t in [ta, tb] where the segment (x0, y0, z0) + t (dx, dy, dz)
        # is on or below the bilinear patch of cell (i, j). Returns -1.0 if there
        # is none.

        # The four corner posts of the cell.
        h00 = float(hgt[i, j])
        h10 = float(hgt[i, j + 1])
        h01 = float(hgt[i + 1, j])
        h11 = float(hgt[i + 1, j + 1])

        # The patch is H(u, v) = h00 + b u + c v + d u v, where (u, v) is the
        # position in the cell. The segment starts at (u0, v0).
        u0 = x0 - j
        v0 = y0 - i
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

            # The footprint of the node: 2^k x 2^k cells, in grid coordinates.
            xlo = float(j << k)
            xhi = float(min((j + 1) << k, cols))
            ylo = float(i << k)
            yhi = float(min((i + 1) << k, rows))

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

            # A leaf is one cell: do the exact test on its bilinear patch.
            if k == 0:
                t = bilinear_hit(hgt, i, j, x0, y0, z0, dx, dy, dz, ta, tb)
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
        x: float, y: float, z: float, a: float, e2: float
    ) -> tuple[float, float, float]:
        # Bowring's method with two iterations. Returns the latitude and the
        # longitude in radians and the height in metres.

        # The semi-minor axis b, the second eccentricity squared ep2, and the
        # distance p from the polar axis.
        b = a * math.sqrt(1.0 - e2)
        ep2 = e2 / (1.0 - e2)
        p = math.sqrt(x * x + y * y)

        # The longitude is exact. The first guess of the reduced latitude beta
        # comes from the point as if it were on the ellipsoid.
        lon = math.atan2(y, x)
        beta = math.atan2(z * a, p * b)
        lat = 0.0

        # Each iteration gets the latitude from beta, then beta from the latitude.
        for _ in range(2):
            sb = math.sin(beta)
            cb = math.cos(beta)
            lat = math.atan2(z + ep2 * b * sb * sb * sb, p - e2 * a * cb * cb * cb)
            beta = math.atan2(b * math.sin(lat), a * math.cos(lat))

        # The height along the ellipsoid normal. This form is stable at all
        # latitudes, also near the poles.
        sl = math.sin(lat)
        h = p * math.cos(lat) + z * sl - a * math.sqrt(1.0 - e2 * sl * sl)

        return lat, lon, h

    @jit
    def ecef_to_grid(
        x: float, y: float, z: float, params: GridParams, lut_x: GridLut, lut_y: GridLut
    ) -> tuple[float, float, float]:
        # ECEF point to grid coordinates (x, y) and ellipsoidal height.

        # Geodetic latitude and longitude in degrees.
        lat, lon, h = ecef_to_geodetic(x, y, z, params[P_A], params[P_E2])
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
    def trace_one(
        hgt: HeightGrid,
        pyr: PyramidData,
        offs: LevelOffsets,
        shp: LevelShapes,
        params: GridParams,
        lut_x: GridLut,
        lut_y: GridLut,
        ox: float,
        oy: float,
        oz: float,
        dx: float,
        dy: float,
        dz: float,
        t_start: float,
        t_end: float,
        nseg: int,
        stack: NodeStack,
    ) -> float:
        # Ray parameter of the first hit of the ECEF ray (ox, oy, oz) + t (dx, dy, dz)
        # on [t_start, t_end] in nseg segments, or NaN.

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

            # Skip a segment with an end outside the grid mapping (NaN). The kernel
            # gives the hit as a fraction of the segment. Convert it to the ray parameter.
            if math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by):
                t = trace_segment(hgt, pyr, offs, shp, ax, ay, az, bx, by, bz, stack)
                if t >= 0.0:
                    return ta + t * (tb - ta)

            ta = tb
            ax = bx
            ay = by
            az = bz

        return math.nan

    @jit_parallel
    def trace_rays(
        hgt: HeightGrid,
        pyr: PyramidData,
        offs: LevelOffsets,
        shp: LevelShapes,
        params: GridParams,
        lut_x: GridLut,
        lut_y: GridLut,
        orig: FloatArray,
        dirs: FloatArray,
        t0: FloatArray,
        t1: FloatArray,
        nseg: IntArray,
        out_t: FloatArray,
    ) -> None:
        # Ray r is orig[r] + t * dirs[r] on [t0[r], t1[r]] in nseg[r] segments.
        n = orig.shape[0]
        size = 4 * shp.shape[0] + 4  # stack_size(num_levels)

        for r in prange(n):
            out_t[r] = math.nan
            if nseg[r] <= 0:
                continue

            # Each ray has its own traversal stack, so the rays can run in parallel.
            stack = np.empty((size, 3), np.int64)
            out_t[r] = trace_one(
                hgt, pyr, offs, shp, params,
                lut_x, lut_y,
                orig[r, 0], orig[r, 1], orig[r, 2],
                dirs[r, 0], dirs[r, 1], dirs[r, 2],
                t0[r], t1[r], nseg[r], stack,
            )  # fmt: skip

    return Kernels(
        bilinear_hit=bilinear_hit,
        clip_slab=clip_slab,
        trace_segment=trace_segment,
        ecef_to_geodetic=ecef_to_geodetic,
        ecef_to_grid=ecef_to_grid,
        trace_one=trace_one,
        trace_rays=trace_rays,
    )
