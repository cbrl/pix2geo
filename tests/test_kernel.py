from collections import Counter

import numpy as np
import pytest
from affine import Affine

from conftest import BACKEND_ATOL, LAT0, LON0, ecef_reference, utm_terrain
from pix2geo import (
    Backend,
    CameraIntrinsics,
    CameraPose,
    Geolocator,
    Terrain,
    available_backends,
    ray_ellipsoid_intersection,
)
from pix2geo._geodesy import (
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    WGS84_EP2,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from pix2geo.backends import _kernel
from pix2geo.backends.python_backend import kernels


def test_kernel_bowring_matches_true_coordinates():
    rng = np.random.default_rng(5)
    lat = rng.uniform(-89.9, 89.9, 2000)
    lon = rng.uniform(-180, 180, 2000)
    h = rng.uniform(-1000, 50000, 2000)
    x, y, z = ecef_reference(lat, lon, h).T
    for i in range(len(lat)):
        la, lo, hh = kernels.ecef_to_geodetic(
            x[i], y[i], z[i], WGS84_A, WGS84_B, WGS84_E2, WGS84_EP2
        )
        assert np.degrees(la) == pytest.approx(lat[i], abs=1e-10)
        assert np.degrees(lo) == pytest.approx(lon[i], abs=1e-10)
        assert hh == pytest.approx(h[i], abs=1e-6)


@pytest.mark.parametrize("make", ["wgs84", "utm"])
def test_kernel_grid_mapping_matches_pyproj(make, hills_terrain):
    ter = hills_terrain if make == "wgs84" else utm_terrain(300)
    params, lut_x, lut_y = ter.grid_mapping()
    mode = _kernel.MODE_AFFINE if make == "wgs84" else _kernel.MODE_LUT
    assert params[_kernel.P_MODE] == mode
    rng = np.random.default_rng(9)
    gx = rng.uniform(0, ter.shape[1] - 1, 500)
    gy = rng.uniform(0, ter.shape[0] - 1, 500)
    lon, lat = ter.grid_to_lonlat(gx, gy)
    x, y, z = ecef_reference(lat, lon, 1234.0).T
    for i in range(len(gx)):
        kx, ky, kh = kernels.ecef_to_grid(x[i], y[i], z[i], params, lut_x, lut_y)
        assert kx == pytest.approx(gx[i], abs=1e-3)
        assert ky == pytest.approx(gy[i], abs=1e-3)
        assert kh == pytest.approx(1234.0, abs=1e-6)


def test_backends_agree_on_projected_terrain(backends):
    ter = utm_terrain(300)
    origin = ecef_reference(LAT0, LON0, 2000.0)
    rng = np.random.default_rng(2)
    dirs = []
    for _ in range(200):
        pose = CameraPose.from_euler(
            LAT0, LON0, 2000.0, yaw=rng.uniform(0, 360), pitch=rng.uniform(-80, -10), roll=0
        )
        dirs.append(pose.rays_to_ecef(np.array([0.0, 0.0, 1.0])))
    dirs = np.array(dirs)
    ref = Geolocator(ter, backend="python").rays_to_geo(origin, dirs)
    assert ref.hit.mean() > 0.9
    for name in backends:
        res = Geolocator(ter, backend=name).rays_to_geo(origin, dirs)
        np.testing.assert_array_equal(res.hit, ref.hit)
        np.testing.assert_allclose(
            res.range, ref.range, rtol=0, atol=BACKEND_ATOL[name], equal_nan=True
        )


def test_quadtree_prunes_cell_tests():
    # Build the kernels with a decorator that counts the calls of each function.
    calls = Counter()

    def counting(func):
        def wrapper(*args, **kwargs):
            calls[func.__name__] += 1
            return func(*args, **kwargs)

        return wrapper

    counted = _kernel.build_kernels(counting, counting, range)

    class CountingBackend(Backend):
        name = "counting"

        def _trace_rays(self, terrain, orig, dirs, limits):
            return _kernel.run_cpu_kernel(counted.trace_rays, terrain, orig, dirs, limits)

    # Flat terrain at 100 m with one 2000 m post in a corner. The post makes the
    # height band 2 km thick, so the shallow ray crosses about 200 cells in the band.
    n, res = 1000, 1.0 / 1000
    h = np.full((n, n), 100.0, np.float32)
    h[0, 0] = 2000.0
    ter = Terrain.from_array(h, Affine(res, 0, LON0 - n * res / 2, 0, -res, LAT0 + n * res / 2))
    pose = CameraPose.from_euler(LAT0, LON0, 1500.0, yaw=45, pitch=-5, roll=0)
    origin = ecef_reference(LAT0, LON0, 1500.0)
    d = pose.rays_to_ecef(np.array([0.0, 0.0, 1.0]))
    out = Geolocator(ter, backend=CountingBackend()).rays_to_geo(origin, d)
    assert bool(out.hit)
    assert float(out.alt) == pytest.approx(100.0, abs=0.01)

    # The quadtree skips every cell above which the ray passes. Only the cells
    # near the hit get the exact test.
    assert calls["trace_segment"] > 25
    assert 0 < calls["bilinear_hit"] < 20


def test_host_bowring_matches_true_coordinates():
    # From 500 m below the surface up to 1000 km above it. NaN points stay NaN.
    rng = np.random.default_rng(6)
    lat = rng.uniform(-90, 90, 20000)
    lon = rng.uniform(-180, 180, 20000)
    h = rng.uniform(-500, 1e6, 20000)
    points = ecef_reference(lat, lon, h)
    points[0] = np.nan
    la, lo, hh = ecef_to_geodetic(points)
    assert np.isnan(la[0]) and np.isnan(lo[0]) and np.isnan(hh[0])
    np.testing.assert_allclose(la[1:], lat[1:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(lo[1:], lon[1:], rtol=0, atol=1e-9)
    np.testing.assert_allclose(hh[1:], h[1:], rtol=0, atol=1e-6)


def test_host_geodetic_to_ecef_matches_proj():
    # Random points from 500 m below the surface up to 1000 km above it, then
    # the poles, the antimeridian and longitudes outside [-180, 180].
    rng = np.random.default_rng(7)
    lat = np.concatenate([rng.uniform(-90, 90, 20000), [90, -90, 0, 89.9999999, -45]])
    lon = np.concatenate([rng.uniform(-180, 180, 20000), [0, 180, -180, 540, -200]])
    h = np.concatenate([rng.uniform(-500, 1e6, 20000), [0, 100, 1e6, -400, 5]])
    want = ecef_reference(lat, lon, h)
    np.testing.assert_allclose(geodetic_to_ecef(lat, lon, h), want, rtol=0, atol=1e-8)

    # One point uses Python floats, with the same steps. One point in 0-d
    # arrays uses the NumPy steps.
    for i in (0, 20000, 20001, 20003, 20004):
        for point in ((lat[i], lon[i], h[i]), (np.array(lat[i]), np.array(lon[i]), np.array(h[i]))):
            got = geodetic_to_ecef(*point)
            assert got.shape == (3,)
            np.testing.assert_allclose(got, want[i], rtol=0, atol=1e-8)

    # The inputs broadcast against each other.
    grid = (lat[:300, None], lon[None, :400], 250.0)
    want = ecef_reference(*grid)
    got = geodetic_to_ecef(*grid)
    assert got.shape == (300, 400, 3)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)

    # An infinite angle gives NaN, also for one point.
    with np.errstate(invalid="ignore"):
        assert np.isnan(geodetic_to_ecef(np.inf, 0.0, 0.0)).all()
        assert np.isnan(geodetic_to_ecef([10.0], [-np.inf], [0.0])[0, :2]).all()


def test_kernel_ellipsoid_hits_match_numpy():
    # Random rays from points between 100 m below the surface and 1000 km above it.
    rng = np.random.default_rng(3)
    n = 500
    lat = rng.uniform(-89, 89, n)
    lon = rng.uniform(-180, 180, n)
    alt = rng.uniform(-100, 1e6, n)
    o = ecef_reference(lat, lon, alt)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    for h in (-500.0, 0.0, 9000.0):
        near, far = ray_ellipsoid_intersection(o, d, h)
        assert 0 < np.isnan(near).mean() < 1
        for i in range(n):
            kn, kf = kernels.ellipsoid_hits(*o[i], *d[i], WGS84_A + h, WGS84_B + h)
            assert kn == pytest.approx(near[i], rel=1e-9, abs=1e-6, nan_ok=True)
            assert kf == pytest.approx(far[i], rel=1e-9, abs=1e-6, nan_ok=True)


@pytest.mark.skipif("cupy" not in available_backends(), reason="needs the CuPy backend")
def test_cupy_float32_walk_keeps_precision_on_wide_raster():
    # A raster 40000 columns wide, with the camera near column 39000. There,
    # float32 grid coordinates are 0.004 cells (9 cm) apart. The walk uses
    # coordinates relative to each segment, so the hits keep millimeter precision.
    rows, cols, res = 128, 40000, 1.0 / 3600
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    h = (1000 + 50 * np.sin(xx / 13.0) * np.cos(yy / 7.0)).astype(np.float32)
    transform = Affine(res, 0, LON0 - 39000 * res, 0, -res, LAT0 + rows * res / 2)
    ter = Terrain.from_array(h, transform, "EPSG:4326")
    cam = CameraIntrinsics.from_fov(640, 480, hfov=60)
    pose = CameraPose.from_euler(LAT0, LON0, 2000.0, yaw=270, pitch=-35, roll=0)
    uv = np.stack(np.meshgrid(np.arange(0, 640, 64.0), np.arange(240, 480, 24.0)), -1)

    ref = Geolocator(ter, backend="python").pixel_to_geo(cam, pose, uv)
    out = Geolocator(ter, backend="cupy").pixel_to_geo(cam, pose, uv)
    assert ref.hit.all()
    np.testing.assert_array_equal(out.hit, ref.hit)
    np.testing.assert_allclose(out.range, ref.range, rtol=0, atol=2e-3)
