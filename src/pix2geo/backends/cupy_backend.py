"""CuPy backend: the intersection kernel as a CUDA kernel, one thread per ray.

The CUDA code is a line-by-line port of :mod:`pix2geo.backends._kernel`,
function for function. Importing this module raises ``ImportError`` when
CuPy is not installed. The :func:`check_usable` check also needs a CUDA
device and a working compiler.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from .._typing import FloatArray, IntArray
from . import _kernel as K

if TYPE_CHECKING:
    from ..terrain import Terrain

#: Traversal stack entries for each thread. Enough for 47 quadtree levels.
STACK = 192

_DEFINES: dict[str, float] = {
    "STACK": STACK,
    "P_AFFINE": K.P_AFFINE,
    "P_WRAP": K.P_WRAP,
    "P_MODE": K.P_MODE,
    "P_A": K.P_A,
    "P_E2": K.P_E2,
    "MODE_AFFINE": K.MODE_AFFINE,
    "RAD_TO_DEG": 180.0 / math.pi,  # the factor of math.degrees
}

_SOURCE = r"""
// Each function ports the function of the same name in _kernel.py. The
// comments here are short. The Python versions have the full comments.

__device__ double bilinear_hit(
    const float* __restrict__ hgt,
    long long hcols,
    long long i,
    long long j,
    double x0,
	double y0,
    double z0,
    double dx,
	double dy,
    double dz,
    double ta,
	double tb
) {
    // The four corner posts of the cell.
    const double h00 = hgt[i * hcols + j];
    const double h10 = hgt[i * hcols + j + 1];
    const double h01 = hgt[(i + 1) * hcols + j];
    const double h11 = hgt[(i + 1) * hcols + j + 1];

    // The patch is H(u, v) = h00 + b u + c v + d u v.
    const double u0 = x0 - (double)j;
    const double v0 = y0 - (double)i;
    const double b = h10 - h00;
    const double c = h01 - h00;
    const double d = h00 - h10 - h01 + h11;

    // f(t) = z(t) - H(u(t), v(t)) = qa t^2 + qb t + qc
    const double qa = -d * dx * dy;
    const double qb = dz - b * dx - c * dy - d * (u0 * dy + v0 * dx);
    const double qc = z0 - h00 - b * u0 - c * v0 - d * u0 * v0;

    // The segment is on or below the patch at ta.
    const double fa = (qa * ta + qb) * ta + qc;
    if (fa <= 0.0) return ta;

    // Find the smallest root of f in [ta, tb].
    double best = -1.0;
    if (qa == 0.0) {
        if (qb != 0.0) {
            const double t = -qc / qb;
            if (t >= ta && t <= tb) best = t;
        }
    }
    else {
        const double disc = qb * qb - 4.0 * qa * qc;
        if (disc >= 0.0) {
            const double sq = sqrt(disc);
            const double q = (qb >= 0.0) ? -0.5 * (qb + sq) : -0.5 * (qb - sq);
            double r1 = q / qa;
            double r2 = (q != 0.0) ? qc / q : r1;
            if (r1 > r2) { const double tmp = r1; r1 = r2; r2 = tmp; }
            if (r1 >= ta && r1 <= tb) best = r1;
            else if (r2 >= ta && r2 <= tb) best = r2;
        }
    }

    // Round-off can hide a root at tb.
    if (best < 0.0) {
        const double fb = (qa * tb + qb) * tb + qc;
        if (fb <= 0.0) best = tb;
    }
    return best;
}

__device__ void clip_slab(
    double p0,
    double dp,
    double lo,
    double hi,
    double* ta,
    double* tb
) {
    if (dp != 0.0) {
        // Keep the part of [ta, tb] between the crossings of lo and hi.
        double t0 = (lo - p0) / dp, t1 = (hi - p0) / dp;
        if (t0 > t1) { const double tmp = t0; t0 = t1; t1 = tmp; }
        if (t0 > *ta) *ta = t0;
        if (t1 < *tb) *tb = t1;
    }
    else if (p0 < lo || p0 > hi) {
        // The line is parallel to the slab and outside it.
        *ta = 1.0; *tb = 0.0;
    }
}

__device__ double trace_segment(
    const float* __restrict__ hgt,
    long long hcols,
    const float* __restrict__ pyr,
    const long long* __restrict__ offs,
    const long long* __restrict__ shp, int nlev,
    double x0,
    double y0,
    double z0,
    double x1,
    double y1,
    double z1,
    long long (*stack)[3]
) {
    const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
    const long long rows = shp[0], cols = shp[1];  // the number of cells

    // Row and column offset of the child that the segment enters first.
    const int near_i = dy >= 0.0 ? 0 : 1;
    const int near_j = dx >= 0.0 ? 0 : 1;

    // Start at the root node, which covers the whole raster.
    stack[0][0] = nlev - 1; stack[0][1] = 0; stack[0][2] = 0;
    int sp = 1;
    while (sp > 0) {
        // Pop the next node: level k, row i, column j.
        --sp;
        const long long k = stack[sp][0], i = stack[sp][1], j = stack[sp][2];

        // The footprint of the node: 2^k x 2^k cells.
        const double xlo = (double)(j << k);
        const double xhi = (double)min((j + 1) << k, cols);
        const double ylo = (double)(i << k);
        const double yhi = (double)min((i + 1) << k, rows);

        // Clip the segment to the footprint.
        double ta = 0.0, tb = 1.0;
        clip_slab(x0, dx, xlo, xhi, &ta, &tb);
        clip_slab(y0, dy, ylo, yhi, &ta, &tb);
        if (ta > tb) continue;

        // Skip the node if the lowest end of the clipped segment is above the node maximum.
        const double za = z0 + dz * ta, zb = z0 + dz * tb;
        const double zmin = za < zb ? za : zb;
        if (zmin > (double)pyr[offs[k] + i * shp[2 * k + 1] + j]) continue;

        // A leaf is one cell: do the exact test on its bilinear patch.
        if (k == 0) {
            const double t = bilinear_hit(hgt, hcols, i, j, x0, y0, z0, dx, dy, dz, ta, tb);
            if (t >= 0.0) return t;
            continue;
        }

        // Push the children far-to-near, so the near child pops first.
        const long long child_rows = shp[2 * (k - 1)], child_cols = shp[2 * (k - 1) + 1];
        for (int n = 0; n < 4; ++n) {
            const long long ci = 2 * i + (n >= 2 ? near_i : 1 - near_i);
            const long long cj = 2 * j + (n % 2 == 1 ? near_j : 1 - near_j);
            if (ci < child_rows && cj < child_cols) {
                stack[sp][0] = k - 1; stack[sp][1] = ci; stack[sp][2] = cj; ++sp;
            }
        }
    }
    return -1.0;
}

__device__ void ecef_to_geodetic(
    double x,
    double y,
    double z,
    double a,
    double e2,
    double* lat,
    double* lon,
    double* h
) {
    // b: semi-minor axis. ep2: second eccentricity squared. p: distance from the polar axis.
    const double b = a * sqrt(1.0 - e2);
    const double ep2 = e2 / (1.0 - e2);
    const double p = sqrt(x * x + y * y);

    // The longitude is exact. beta is the first guess of the reduced latitude.
    *lon = atan2(y, x);
    double beta = atan2(z * a, p * b);
    double phi = 0.0;

    // Each iteration gets the latitude from beta, then beta from the latitude.
    for (int it = 0; it < 2; ++it) {
        const double sb = sin(beta), cb = cos(beta);
        phi = atan2(z + ep2 * b * sb * sb * sb, p - e2 * a * cb * cb * cb);
        beta = atan2(b * sin(phi), a * cos(phi));
    }

    // The height along the ellipsoid normal.
    const double sl = sin(phi);
    *h = p * cos(phi) + z * sl - a * sqrt(1.0 - e2 * sl * sl);
    *lat = phi;
}

__device__ void ecef_to_grid(double x,
    double y,
    double z,
    const double* __restrict__ params,
    const double* __restrict__ lut_x,
    const double* __restrict__ lut_y,
    long long lut_rows,
    long long lut_cols,
    double* gx,
    double* gy,
    double* gh
) {
    // Geodetic latitude and longitude in degrees.
    double lat, lon;
    ecef_to_geodetic(x, y, z, params[P_A], params[P_E2], &lat, &lon, gh);
    lat *= RAD_TO_DEG;
    lon *= RAD_TO_DEG;

    // Wrap the longitude to within 180 degrees of c. fmod keeps the sign, so add 360 if needed.
    const double c = params[P_WRAP];
    if (c == c) {
        double m = fmod(lon - c + 180.0, 360.0);
        if (m < 0.0) m += 360.0;
        lon = c + m - 180.0;
    }

    // The affine map from (lon, lat) to (u, v). In affine mode, (u, v) are the grid coordinates.
    const double* aff = params + P_AFFINE;
    const double u = aff[0] * lon + aff[1] * lat + aff[2];
    const double v = aff[3] * lon + aff[4] * lat + aff[5];
    if (params[P_MODE] == MODE_AFFINE) { *gx = u; *gy = v; return; }

    // In LUT mode, (u, v) are fractional table indices. Points outside the tables give NaN.
    if (!(u >= 0.0 && u <= (double)(lut_cols - 1) && v >= 0.0 && v <= (double)(lut_rows - 1))) {
        *gx = CUDART_NAN; *gy = CUDART_NAN; return;
    }

    // Bilinear interpolation in table cell (i, j): along u, then along v.
    long long j = (long long)u; if (j > lut_cols - 2) j = lut_cols - 2;
    long long i = (long long)v; if (i > lut_rows - 2) i = lut_rows - 2;
    const double fu = u - (double)j, fv = v - (double)i;
    const long long q00 = i * lut_cols + j, q01 = q00 + 1;
    const long long q10 = q00 + lut_cols, q11 = q10 + 1;

    *gx = (lut_x[q00] * (1.0 - fu) + lut_x[q01] * fu) * (1.0 - fv)
        + (lut_x[q10] * (1.0 - fu) + lut_x[q11] * fu) * fv;

    *gy = (lut_y[q00] * (1.0 - fu) + lut_y[q01] * fu) * (1.0 - fv)
        + (lut_y[q10] * (1.0 - fu) + lut_y[q11] * fu) * fv;
}

__device__ double trace_one(\
    const float* __restrict__ hgt,
    long long hcols,
    const float* __restrict__ pyr,
    const long long* __restrict__ offs,
    const long long* __restrict__ shp,
    int nlev,
    const double* __restrict__ params,
    const double* __restrict__ lut_x,
    const double* __restrict__ lut_y,
    long long lut_rows,
    long long lut_cols,
    double ox,
    double oy,
    double oz,
    double dx,
    double dy,
    double dz,
    double t_start,
    double t_end,
    long long nseg,
    long long (*stack)[3]
) {
    // Grid coordinates of the start of the first segment.
    const double step = (t_end - t_start) / (double)nseg;
    double ta = t_start;
    double ax, ay, az;
    ecef_to_grid(
        ox + ta * dx,
        oy + ta * dy,
        oz + ta * dz,
        params,
        lut_x,
        lut_y,
        lut_rows,
        lut_cols,
        &ax,
        &ay,
        &az
    );

    for (long long k = 0; k < nseg; ++k) {
        // The last segment ends exactly at t_end.
        const double tb = (k == nseg - 1) ? t_end : t_start + (double)(k + 1) * step;
        double bx, by, bz;
        ecef_to_grid(ox + tb * dx, oy + tb * dy, oz + tb * dz, params, lut_x, lut_y,
                     lut_rows, lut_cols, &bx, &by, &bz);

        // Skip a segment with a NaN end. Convert a hit fraction to the ray parameter.
        if (isfinite(ax) && isfinite(ay) && isfinite(bx) && isfinite(by)) {
            const double t = trace_segment(hgt, hcols, pyr, offs, shp, nlev,
                                           ax, ay, az, bx, by, bz, stack);
            if (t >= 0.0) return ta + t * (tb - ta);
        }

        // This segment end is the start of the next segment.
        ta = tb; ax = bx; ay = by; az = bz;
    }
    return CUDART_NAN;
}

extern "C" __global__
void trace_rays(
    const float* __restrict__ hgt,
    long long hcols,
    const float* __restrict__ pyr,
    const long long* __restrict__ offs,
    const long long* __restrict__ shp,
    int nlev,
    const double* __restrict__ params,
    const double* __restrict__ lut_x,
    const double* __restrict__ lut_y,
    long long lut_rows,
    long long lut_cols,
    const double* __restrict__ orig,
    const double* __restrict__ dirs,
    const double* __restrict__ t0,
    const double* __restrict__ t1,
    const long long* __restrict__ nseg,
    long long n,
    double* __restrict__ out_t
) {
    // One thread for each ray. Each thread has its own traversal stack.
    const long long r = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (r >= n) return;

    out_t[r] = CUDART_NAN;
    if (nseg[r] <= 0) return;

    long long stack[STACK][3];
    out_t[r] = trace_one(hgt, hcols, pyr, offs, shp, nlev, params, lut_x, lut_y,
                         lut_rows, lut_cols,
                         orig[3 * r], orig[3 * r + 1], orig[3 * r + 2],
                         dirs[3 * r], dirs[3 * r + 1], dirs[3 * r + 2],
                         t0[r], t1[r], nseg[r], stack);
}
"""

_module: cp.RawModule | None = None


def _kernel() -> cp.RawKernel:
    """The compiled ``trace_rays`` kernel. It compiles on the first call."""
    global _module
    if _module is None:
        header = "#include <math_constants.h>\n"
        header += "".join(f"#define {name} {value!r}\n" for name, value in _DEFINES.items())
        # --fmad=false keeps the floating-point results equal to the CPU backends.
        _module = cp.RawModule(code=header + _SOURCE, options=("--std=c++14", "--fmad=false"))
    return _module.get_function("trace_rays")


def check_usable() -> None:
    """Raise an exception when no CUDA device is present or the kernel does not compile.

    The kernel compiles here, so a missing CUDA compiler shows when the
    backend is selected, not on the first ray batch.
    """
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device is present")

    _kernel()


def _device_terrain(terrain: Terrain) -> dict[str, cp.ndarray]:
    """The terrain arrays on the GPU. They upload once for each terrain."""
    cached = terrain._device_cache.get("cupy")

    if cached is None:
        pyr = terrain.pyramid
        if K.stack_size(pyr.num_levels) > STACK:
            raise ValueError("the elevation model is too large for the CUDA stack")

        params, lut_x, lut_y = terrain.grid_mapping()
        cached = {
            "hgt": cp.asarray(np.ascontiguousarray(terrain.heights, dtype=np.float32)),
            "pyr": cp.asarray(pyr.data),
            "offs": cp.asarray(pyr.offsets),
            "shp": cp.asarray(np.ascontiguousarray(pyr.shapes.ravel())),
            "params": cp.asarray(params),
            "lut_x": cp.asarray(np.ascontiguousarray(lut_x)),
            "lut_y": cp.asarray(np.ascontiguousarray(lut_y)),
        }
        terrain._device_cache["cupy"] = cached

    return cached


def trace_rays(
    terrain: Terrain,
    orig: FloatArray,
    dirs: FloatArray,
    t0: FloatArray,
    t1: FloatArray,
    nseg: IntArray,
    block: int = 128,
) -> FloatArray:
    """Run the kernel. See :meth:`pix2geo.backends.Backend.trace_rays`."""
    n = len(orig)
    if n == 0:
        return np.empty(0)

    dev = _device_terrain(terrain)
    lut_rows, lut_cols = dev["lut_x"].shape
    out = cp.empty(n, dtype=cp.float64)

    _kernel()(
        ((n + block - 1) // block,),
        (block,),
        (
            dev["hgt"], np.int64(terrain.heights.shape[1]),
            dev["pyr"], dev["offs"], dev["shp"], np.int32(terrain.pyramid.num_levels),
            dev["params"], dev["lut_x"], dev["lut_y"], np.int64(lut_rows), np.int64(lut_cols),
            cp.asarray(orig), cp.asarray(dirs), cp.asarray(t0), cp.asarray(t1), cp.asarray(nseg),
            np.int64(n), out,
        ),
    )

    return cp.asnumpy(out)
