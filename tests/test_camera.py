import numpy as np
import pytest

from pix2geo import CameraIntrinsics


def test_from_fov_square_pixels():
    cam = CameraIntrinsics.from_fov(1920, 1080, hfov=90)
    assert cam.fx == pytest.approx(960.0)
    assert cam.fy == pytest.approx(960.0)
    assert (cam.cx, cam.cy) == (959.5, 539.5)
    assert cam.hfov == pytest.approx(90.0)


def test_from_fov_vfov_and_dfov():
    cam = CameraIntrinsics.from_fov(640, 480, vfov=40)
    assert cam.vfov == pytest.approx(40.0)
    assert cam.fx == cam.fy
    cam = CameraIntrinsics.from_fov(640, 480, dfov=70)
    half_diag = np.hypot(640, 480) / 2
    assert np.degrees(2 * np.arctan(half_diag / cam.fx)) == pytest.approx(70.0)
    with pytest.raises(ValueError):
        CameraIntrinsics.from_fov(640, 480)


def test_center_ray_is_optical_axis():
    cam = CameraIntrinsics.from_fov(641, 481, hfov=60)
    ray = cam.pixel_to_ray(320, 240)
    np.testing.assert_allclose(ray, [0, 0, 1], atol=1e-12)


def test_edge_ray_matches_half_fov():
    cam = CameraIntrinsics.from_fov(800, 600, hfov=70)
    ray = cam.pixel_to_ray(-0.5, cam.cy)  # left image edge
    assert np.degrees(np.arctan2(-ray[0], ray[2])) == pytest.approx(35.0)
    ray = cam.pixel_to_ray(cam.cx, cam.height - 0.5)  # bottom edge: +y is down
    assert ray[1] > 0


def test_from_matrix_roundtrip():
    K = [[1000.0, 0.0, 400.0], [0.0, 1100.0, 300.0], [0.0, 0.0, 1.0]]
    cam = CameraIntrinsics.from_matrix(K, 800, 600)
    np.testing.assert_allclose(cam.K, K)


@pytest.mark.parametrize(
    "dist",
    [
        None,
        (-0.2, 0.05, 0.001, -0.002),
        (-0.1, 0.02, 0.0, 0.0, -0.005),
        (0.1, -0.05, 0.001, 0.001, 0.0, 0.05, -0.01, 0.0),
    ],
)
def test_pixel_ray_roundtrip(dist):
    cam = CameraIntrinsics.from_fov(1280, 720, hfov=80, dist=dist)
    rng = np.random.default_rng(0)
    uv = rng.uniform([0, 0], [1279, 719], size=(500, 2))
    rays = cam.pixel_to_ray(uv[:, 0], uv[:, 1])
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0)
    u, v = cam.ray_to_pixel(rays)
    np.testing.assert_allclose(np.c_[u, v], uv, atol=1e-6)


def test_behind_camera_projects_to_nan():
    cam = CameraIntrinsics.from_fov(100, 100, hfov=60)
    u, v = cam.ray_to_pixel(np.array([[0.0, 0.0, -1.0]]))
    assert np.isnan(u).all() and np.isnan(v).all()


def test_bad_distortion_length():
    with pytest.raises(ValueError):
        CameraIntrinsics.from_fov(100, 100, hfov=60, dist=(0.1, 0.2, 0.3))


@pytest.mark.parametrize("fov", [0.0, 180.0, -10.0, np.nan])
def test_bad_field_of_view(fov):
    with pytest.raises(ValueError, match="field of view"):
        CameraIntrinsics.from_fov(640, 480, hfov=fov)


def test_bad_intrinsics():
    K = np.array([[1000.0, 0.0, 400.0], [0.0, 1100.0, 300.0], [0.0, 0.0, 1.0]])
    # A transposed K would otherwise give a principal point of (0, 0).
    with pytest.raises(ValueError, match="form"):
        CameraIntrinsics.from_matrix(K.T, 800, 600)
    with pytest.raises(ValueError, match="finite"):
        CameraIntrinsics.from_matrix(np.where(K == 1000.0, np.nan, K), 800, 600)
    with pytest.raises(ValueError, match="finite"):
        CameraIntrinsics(800, 600, np.inf, 1000.0, 400.0, 300.0)
    with pytest.raises(ValueError, match="size"):
        CameraIntrinsics(0, 600, 1000.0, 1000.0, 400.0, 300.0)
