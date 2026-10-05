import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import ecef_reference, enu_axes_reference
from pix2geo import CameraIntrinsics, CameraPose
from pix2geo.pose import ned_to_ecef_matrix

LAT, LON = 35.0, 139.0
ENU_AXES = enu_axes_reference(LAT, LON)


def enu_dir(e, n, u):
    return np.array([e, n, u], float) @ ENU_AXES


def axis_ecef(pose, vec_cam):
    return pose.rays_to_ecef(np.asarray(vec_cam, float))


def test_level_north():
    pose = CameraPose.from_euler(LAT, LON, 100, yaw=0, pitch=0, roll=0)
    np.testing.assert_allclose(axis_ecef(pose, [0, 0, 1]), enu_dir(0, 1, 0), atol=1e-12)
    # Image right is east, image down is down.
    np.testing.assert_allclose(axis_ecef(pose, [1, 0, 0]), enu_dir(1, 0, 0), atol=1e-12)
    np.testing.assert_allclose(axis_ecef(pose, [0, 1, 0]), enu_dir(0, 0, -1), atol=1e-12)


def test_yaw_east_and_nadir():
    pose = CameraPose.from_euler(LAT, LON, 100, yaw=90, pitch=0, roll=0)
    np.testing.assert_allclose(axis_ecef(pose, [0, 0, 1]), enu_dir(1, 0, 0), atol=1e-12)
    pose = CameraPose.from_euler(LAT, LON, 100, yaw=0, pitch=-90, roll=0)
    np.testing.assert_allclose(axis_ecef(pose, [0, 0, 1]), enu_dir(0, 0, -1), atol=1e-12)
    # Nadir with yaw 0: the top of the image points north.
    np.testing.assert_allclose(axis_ecef(pose, [0, -1, 0]), enu_dir(0, 1, 0), atol=1e-12)


def test_positive_roll_turns_image_right_down():
    pose = CameraPose.from_euler(LAT, LON, 100, yaw=0, pitch=0, roll=30)
    right = axis_ecef(pose, [1, 0, 0])
    up = enu_dir(0, 0, 1)
    assert np.dot(right, up) == pytest.approx(-0.5)


def test_mount_composes_with_platform():
    mount = Rotation.from_euler("ZYX", [0, -90, 0], degrees=True)  # gimbal down
    pose = CameraPose.from_euler(LAT, LON, 100, yaw=45, pitch=0, roll=0, mount=mount)
    np.testing.assert_allclose(axis_ecef(pose, [0, 0, 1]), enu_dir(0, 0, -1), atol=1e-12)


def test_euler_roundtrip():
    pose = CameraPose.from_euler(LAT, LON, 100, yaw=123, pitch=-34, roll=7)
    np.testing.assert_allclose(pose.euler(), [123, -34, 7], atol=1e-9)


@pytest.mark.parametrize("frame", ["ENU", "ECEF"])
def test_from_rotation_frames_agree(frame):
    ref = CameraPose.from_euler(LAT, LON, 100, yaw=20, pitch=-40, roll=5)
    cam_to_ned = ref.rotation.as_matrix()
    if frame == "ENU":
        m = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) @ cam_to_ned
    else:
        m = ned_to_ecef_matrix(LAT, LON) @ cam_to_ned
    pose = CameraPose.from_rotation(LAT, LON, 100, m, frame=frame)
    np.testing.assert_allclose(pose.rotation.as_matrix(), cam_to_ned, atol=1e-12)
    pose_q = CameraPose.from_rotation(LAT, LON, 100, Rotation.from_matrix(m).as_quat(), frame=frame)
    np.testing.assert_allclose(pose_q.rotation.as_matrix(), cam_to_ned, atol=1e-12)


def test_look_at_projects_target_to_principal_point():
    cam = CameraIntrinsics.from_fov(1001, 801, hfov=50)
    pose = CameraPose.look_at(LAT, LON, 2000.0, LAT + 0.01, LON + 0.02, 50.0)
    vec = ecef_reference(LAT + 0.01, LON + 0.02, 50.0) - ecef_reference(LAT, LON, 2000.0)
    vc = pose.ecef_to_camera(vec)
    u, v = cam.ray_to_pixel(vc)
    assert u == pytest.approx(cam.cx, abs=1e-6)
    assert v == pytest.approx(cam.cy, abs=1e-6)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(LAT, LON), (90.0, 0.0), (-90.0, 45.0), (0.0, 180.0), (-33.3, -70.6), (89.99, -179.9)],
)
def test_ned_to_ecef_matrix_matches_proj(lat, lon):
    # The columns are north, east and down, also at the poles and the antimeridian.
    e, n, u = enu_axes_reference(lat, lon)
    np.testing.assert_allclose(ned_to_ecef_matrix(lat, lon), np.c_[n, e, -u], atol=1e-15)


def test_ecef_camera_roundtrip():
    pose = CameraPose.from_euler(LAT, LON, 100, yaw=-70, pitch=12, roll=-3)
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, 3))
    np.testing.assert_allclose(pose.ecef_to_camera(pose.rays_to_ecef(v)), v, atol=1e-12)
    np.testing.assert_allclose(pose.camera_to_ecef() @ v[0], pose.rays_to_ecef(v[0]), atol=1e-12)


def test_alt_ref_validation():
    with pytest.raises(ValueError):
        CameraPose.from_euler(LAT, LON, 100, 0, 0, 0, alt_ref="msl")
    pose = CameraPose.from_euler(LAT, LON, 100, 0, 0, 0, alt_ref="geoid")
    with pytest.raises(ValueError):
        pose.ellipsoidal_altitude()


def test_position_and_rotation_validation():
    # A longitude in the latitude argument is a common mistake.
    with pytest.raises(ValueError, match="lat"):
        CameraPose.from_euler(-105.0, 40.0, 100, 0, 0, 0)
    with pytest.raises(ValueError, match="finite"):
        CameraPose.from_euler(LAT, LON, np.nan, 0, 0, 0)
    stack = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="one scipy Rotation"):
        CameraPose(LAT, LON, 100.0, stack)
