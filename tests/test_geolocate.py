import numpy as np
import pytest
from affine import Affine

from conftest import (
    BACKEND_ATOL,
    LAT0,
    LON0,
    ecef_reference,
    geodetic_reference,
    hills,
    utm_terrain,
)
from pix2geo import (
    CameraIntrinsics,
    CameraPose,
    Geolocator,
    Terrain,
    ray_ellipsoid_intersection,
)

CAM = CameraIntrinsics.from_fov(640, 480, hfov=60)


def flat_terrain(height=500.0, n=600, res=1.0 / 1000):
    h = np.full((n, n), height, np.float32)
    tr = Affine(res, 0, LON0 - n * res / 2, 0, -res, LAT0 + n * res / 2)
    return Terrain.from_array(h, tr, "EPSG:4326")


def march_reference(terrain, origin, direction, t_max, step=0.05):
    """Brute-force first crossing by dense sampling along the exact ECEF ray."""
    t = np.arange(0.0, t_max, step)
    p = origin + t[:, None] * direction
    lat, lon, h = geodetic_reference(p)
    g = h - terrain.height_at(lat, lon)
    below = np.flatnonzero(g <= 0)
    if below.size == 0:
        return np.nan
    k = below[0]
    if k == 0:
        return 0.0
    return t[k - 1] + step * g[k - 1] / (g[k - 1] - g[k])


def test_flat_terrain_matches_ellipsoid(backend_name):
    ter = flat_terrain(500.0)
    geo = Geolocator(ter, backend=backend_name)
    pose = CameraPose.from_euler(LAT0, LON0, 2500.0, yaw=60, pitch=-35, roll=10)
    uv = np.stack(np.meshgrid(np.linspace(0, 639, 9), np.linspace(0, 479, 7)), -1)
    res = geo.pixel_to_geo(CAM, pose, uv)
    assert res.hit.all()
    np.testing.assert_allclose(res.alt, 500.0, atol=0.02)
    o = geo.camera_position_ecef(pose)
    d = pose.rays_to_ecef(CAM.pixel_to_ray(uv[..., 0], uv[..., 1]))
    t_ref, _ = ray_ellipsoid_intersection(o, d, 500.0)
    np.testing.assert_allclose(res.range, t_ref, atol=0.05)


def test_matches_brute_force(hills_terrain, backend_name):
    geo = Geolocator(hills_terrain, backend=backend_name, max_segment_length=100.0)
    rng = np.random.default_rng(7)
    origin = ecef_reference(LAT0, LON0, 2200.0)
    for _ in range(15):
        pose = CameraPose.from_euler(
            LAT0, LON0, 2200.0, yaw=rng.uniform(0, 360), pitch=rng.uniform(-40, -8), roll=0
        )
        d = pose.rays_to_ecef(np.array([0.0, 0.0, 1.0]))
        res = geo.rays_to_geo(origin, d)
        ref = march_reference(hills_terrain, origin, d, t_max=20000.0, step=0.5)
        assert np.isfinite(ref)
        assert float(res.range) == pytest.approx(ref, abs=0.05)


def test_roundtrip_ground_points(hills_terrain, backend_name):
    geo = Geolocator(hills_terrain, backend=backend_name)
    cam = CameraIntrinsics.from_fov(1920, 1080, hfov=70)
    pose = CameraPose.from_euler(LAT0 - 0.03, LON0 + 0.02, 6000.0, yaw=320, pitch=-55, roll=3)
    rng = np.random.default_rng(11)
    lat = LAT0 + rng.uniform(-0.06, 0.06, 4000)
    lon = LON0 + rng.uniform(-0.06, 0.06, 4000)
    alt = hills_terrain.height_at(lat, lon)
    u, v, visible = geo.geo_to_pixel(cam, pose, lat, lon, alt, check_occlusion=True)
    assert visible.sum() > 500
    res = geo.pixel_to_geo(cam, pose, np.c_[u[visible], v[visible]])
    assert res.hit.all()
    # The ENU distance is the same as the ECEF distance.
    found = ecef_reference(res.lat, res.lon, res.alt)
    err = np.linalg.norm(found - ecef_reference(lat[visible], lon[visible], alt[visible]), axis=-1)
    assert np.max(err) < 0.05


def test_occlusion_casts_only_points_in_view(hills_terrain, monkeypatch):
    geo = Geolocator(hills_terrain, backend="python")
    pose = CameraPose.from_euler(LAT0, LON0, 2200.0, yaw=20, pitch=-12, roll=0)
    rng = np.random.default_rng(5)
    lat = LAT0 + rng.uniform(-0.2, 0.2, 600)
    lon = LON0 + rng.uniform(-0.2, 0.2, 600)
    alt = hills_terrain.height_at(lat, lon)
    _, _, in_view = geo.geo_to_pixel(CAM, pose, lat, lon, alt)

    # A ray from the camera to each point in view, as the reference.
    cam = geo.camera_position_ecef(pose)
    vec = ecef_reference(lat, lon, alt)[in_view] - cam
    first = geo.rays_to_geo(cam, vec)
    expected = in_view.copy()
    expected[in_view] = ~(first.hit & (first.range < np.linalg.norm(vec, axis=-1) - 1.0))

    # Only the points in view need a ray.
    calls = []
    trace = geo.rays_to_geo
    monkeypatch.setattr(geo, "rays_to_geo", lambda o, d: calls.append(len(d)) or trace(o, d))
    _, _, visible = geo.geo_to_pixel(CAM, pose, lat, lon, alt, check_occlusion=True)
    assert calls == [in_view.sum()]
    assert 0 < visible.sum() < in_view.sum()
    np.testing.assert_array_equal(visible, expected)

    # One point gives one NumPy bool, as before.
    k = int(np.flatnonzero(in_view)[0])
    _, _, one = geo.geo_to_pixel(CAM, pose, lat[k], lon[k], alt[k], check_occlusion=True)
    assert isinstance(one, np.bool_) and one == expected[k]


def test_backends_agree(hills_terrain, backends):
    pose = CameraPose.from_euler(LAT0 + 0.05, LON0, 4000.0, yaw=200, pitch=-25, roll=-4)
    uv = np.stack(np.meshgrid(np.arange(0, 640, 16.0), np.arange(0, 480, 16.0)), -1)
    ref = Geolocator(hills_terrain, backend="python").pixel_to_geo(CAM, pose, uv)
    for name in backends:
        res = Geolocator(hills_terrain, backend=name).pixel_to_geo(CAM, pose, uv)
        np.testing.assert_array_equal(res.hit, ref.hit)
        np.testing.assert_allclose(
            res.range, ref.range, rtol=0, atol=BACKEND_ATOL[name], equal_nan=True
        )


def test_sky_misses_and_fallback(hills_terrain):
    geo = Geolocator(hills_terrain)
    pose = CameraPose.from_euler(LAT0, LON0, 3000.0, yaw=0, pitch=30, roll=0)
    res = geo.pixel_to_geo(CAM, pose, (CAM.cx, CAM.cy))
    assert not bool(res.hit) and np.isnan(res.lat)
    # Looking down but far outside the DEM: fallback to a constant height.
    pose = CameraPose.from_euler(LAT0 + 3.0, LON0, 3000.0, yaw=0, pitch=-60, roll=0)
    res = geo.pixel_to_geo(CAM, pose, (CAM.cx, CAM.cy))
    assert np.isnan(res.lat)
    res = geo.pixel_to_geo(CAM, pose, (CAM.cx, CAM.cy), fallback_height=100.0)
    assert not bool(res.hit)
    assert float(res.alt) == pytest.approx(100.0, abs=0.05)


def test_max_range(hills_terrain):
    pose = CameraPose.from_euler(LAT0, LON0, 3000.0, yaw=0, pitch=-20, roll=0)
    full = Geolocator(hills_terrain).pixel_to_geo(CAM, pose, (CAM.cx, CAM.cy))
    short = Geolocator(hills_terrain, max_range=float(full.range) - 10).pixel_to_geo(
        CAM, pose, (CAM.cx, CAM.cy)
    )
    assert bool(full.hit) and not bool(short.hit)


def test_nodata_hole_is_transparent(backend_name):
    h, tr = hills(rows=200, cols=200)
    h[90:110, 90:110] = np.nan
    ter = Terrain.from_array(h, tr, "EPSG:4326")
    lon, lat = ter.grid_to_lonlat(100.0, 100.0)
    pose = CameraPose.from_euler(float(lat), float(lon), 3000.0, yaw=0, pitch=-90, roll=0)
    res = Geolocator(ter, backend=backend_name).pixel_to_geo(CAM, pose, (CAM.cx, CAM.cy))
    assert not bool(res.hit)
    filled = Terrain.from_array(h, tr, "EPSG:4326", fill_nodata=1000.0)
    res = Geolocator(filled, backend=backend_name).pixel_to_geo(CAM, pose, (CAM.cx, CAM.cy))
    assert float(res.alt) == pytest.approx(1000.0, abs=0.01)


def test_geoid_and_altitude_references():
    h, tr = hills()
    ortho = Terrain.from_array(h, tr, "EPSG:4326", geoid=-20.0)
    ellip = Terrain.from_array(h - 20.0, tr, "EPSG:4326")
    pose = CameraPose.from_euler(LAT0, LON0, 3000.0, yaw=45, pitch=-50, roll=0)
    uv = np.array([[100.0, 100.0], [500.0, 400.0]])
    a = Geolocator(ortho).pixel_to_geo(CAM, pose, uv)
    b = Geolocator(ellip).pixel_to_geo(CAM, pose, uv)
    np.testing.assert_allclose(a.as_array(), b.as_array(), atol=1e-6)
    np.testing.assert_allclose(a.alt_msl, a.alt + 20.0, atol=1e-9)
    assert b.alt_msl is None
    # Camera altitude 3020 m above the geoid is 3000 m above the ellipsoid.
    msl = Geolocator(ortho).pixel_to_geo(CAM, pose.with_altitude(3020.0, "geoid"), uv)
    np.testing.assert_allclose(msl.range, a.range, atol=1e-6)
    # AGL altitude is relative to the terrain below the camera.
    ground = float(ortho.height_at(LAT0, LON0))
    agl = Geolocator(ortho).pixel_to_geo(CAM, pose.with_altitude(3000.0 - ground, "agl"), uv)
    np.testing.assert_allclose(agl.range, a.range, atol=1e-6)


def test_utm_terrain():
    ter = utm_terrain(400)
    geo = Geolocator(ter)
    pose = CameraPose.from_euler(LAT0, LON0, 2500.0, yaw=135, pitch=-40, roll=0)
    uv = np.stack(np.meshgrid(np.arange(0, 640, 32.0), np.arange(0, 480, 32.0)), -1)
    res = geo.pixel_to_geo(CAM, pose, uv)
    assert res.hit.all()
    np.testing.assert_allclose(res.alt, ter.height_at(res.lat, res.lon), atol=0.01)


def test_antimeridian():
    # A DEM from 179.5 E to 180.5 E (= 179.5 W), stored with longitudes > 180.
    n = 400
    res = 1.0 / n
    h = np.full((n, n), 50.0, np.float32)
    ter = Terrain.from_array(h, Affine(res, 0, 179.5, 0, -res, 10.5), "EPSG:4326")
    # The camera is east of the antimeridian. It looks west, across it.
    pose = CameraPose.from_euler(10.3, -179.99, 2000.0, yaw=270, pitch=-30, roll=0)
    out = Geolocator(ter).pixel_to_geo(CAM, pose, (CAM.cx, CAM.cy))
    assert bool(out.hit)
    assert 179.5 < float(out.lon) < 180.0
    assert float(out.alt) == pytest.approx(50.0, abs=0.01)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_range": 0.0}, {"max_segment_length": -1.0}, {"batch_size": 0}],
)
def test_geolocator_validation(hills_terrain, kwargs):
    with pytest.raises(ValueError):
        Geolocator(hills_terrain, **kwargs)
