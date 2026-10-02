"""Camera extrinsics: geographic position and orientation.

Frames and conventions
----------------------
* **Camera optical frame** (OpenCV): +X right, +Y down, +Z forward.
* **Camera body frame** (aerospace FRD): +X forward along the optical axis,
  +Y right, +Z down. The fixed rotation :data:`CAMERA_TO_BODY` maps optical
  frame vectors into this frame.
* **Local NED frame**: North, East, Down at the camera position. "Down" is
  along the WGS84 ellipsoid normal (geodetic, not geocentric).
* **Euler angles**: yaw, pitch, roll as intrinsic Z-Y'-X'' Tait-Bryan
  rotations from NED to the body frame. Yaw is the heading clockwise from
  north. Positive pitch raises the nose. Positive roll lowers the right
  wing. With ``yaw=0, pitch=0, roll=0`` the camera looks north at the
  horizon. With ``pitch=-90`` the camera looks straight down (nadir).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Union, get_args

import numpy as np
from scipy.spatial.transform import Rotation

from ._geodesy import geodetic_to_ecef
from ._typing import ArrayLike, FloatArray

if TYPE_CHECKING:
    from .geoid import SupportsUndulation
    from .terrain import Terrain

__all__ = [
    "CAMERA_TO_BODY",
    "NED_TO_ENU",
    "AltRef",
    "CameraPose",
    "RotationLike",
    "euler_rotation",
    "ned_to_ecef_matrix",
]

#: Rotation from the camera optical frame (x right, y down, z forward) to the
#: camera body frame (x forward, y right, z down).
CAMERA_TO_BODY = Rotation.from_matrix(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

#: Rotation from local NED to local ENU coordinates.
NED_TO_ENU = Rotation.from_matrix(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

#: The reference surface of a camera altitude.
AltRef = Literal["ellipsoid", "geoid", "agl"]

#: A scipy ``Rotation``, a 3x3 rotation matrix or an ``(x, y, z, w)`` quaternion.
RotationLike = Union[Rotation, ArrayLike]


def euler_rotation(yaw: float, pitch: float, roll: float, degrees: bool = True) -> Rotation:
    """Aerospace yaw/pitch/roll (intrinsic ZYX) as a body-to-NED rotation."""
    return Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=degrees)


def ned_to_ecef_matrix(lat: float, lon: float) -> FloatArray:
    """3x3 matrix whose columns are the N, E, D unit vectors in ECEF."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    sp, cp = math.sin(phi), math.cos(phi)
    sl, cl = math.sin(lam), math.cos(lam)

    # North is the latitude direction and east is the longitude direction.
    # Down is the inward ellipsoid normal.
    return np.array(
        [
            [-sp * cl, -sl, -cp * cl],
            [-sp * sl, cl, -cp * sl],
            [cp, 0.0, -sp],
        ]
    )


def _as_rotation(rotation: RotationLike) -> Rotation:
    if isinstance(rotation, Rotation):
        return rotation

    arr = np.asarray(rotation, dtype=float)
    if arr.shape == (3, 3):
        return Rotation.from_matrix(arr)
    if arr.shape == (4,):
        return Rotation.from_quat(arr)  # scalar-last (x, y, z, w)

    raise ValueError(
        "rotation must be a scipy Rotation, a 3x3 matrix or an (x, y, z, w) quaternion"
    )


@dataclass(frozen=True)
class CameraPose:
    """Camera position and orientation.

    Parameters
    ----------
    lat, lon:
        Geodetic WGS84 latitude and longitude in degrees.
    alt:
        Camera altitude in meters. The meaning depends on ``alt_ref``.
    rotation:
        Rotation from the camera *optical* frame to local NED.
    alt_ref:
        ``"ellipsoid"`` (height above the WGS84 ellipsoid, the default),
        ``"geoid"`` (height above mean sea level, needs a geoid model) or
        ``"agl"`` (height above the terrain model).
    """

    lat: float
    lon: float
    alt: float
    rotation: Rotation
    alt_ref: AltRef = "ellipsoid"

    def __post_init__(self) -> None:
        if self.alt_ref not in get_args(AltRef):
            raise ValueError(f"alt_ref must be one of {get_args(AltRef)}")
        if not np.all(np.isfinite([self.lat, self.lon, self.alt])):
            raise ValueError("lat, lon and alt must be finite")

        # A latitude outside [-90, 90] is usually a longitude in the wrong argument.
        if abs(self.lat) > 90.0:
            raise ValueError(f"lat must be in [-90, 90] degrees, not {self.lat}")
        if not isinstance(self.rotation, Rotation) or not self.rotation.single:
            raise ValueError("rotation must be one scipy Rotation, not a stack of rotations")

    # ---- Constructors -------------------------------------------------------

    @classmethod
    def from_euler(
        cls,
        lat: float,
        lon: float,
        alt: float,
        yaw: float,
        pitch: float,
        roll: float,
        *,
        degrees: bool = True,
        mount: RotationLike | None = None,
        alt_ref: AltRef = "ellipsoid",
    ) -> CameraPose:
        """Pose from aerospace yaw/pitch/roll.

        The angles give the attitude of the platform body frame (FRD) in NED.
        ``mount`` is an optional rotation from the camera body frame to the
        platform body frame, for example the gimbal angles relative to the
        aircraft. Use :func:`euler_rotation` to make it. Without ``mount``,
        the camera looks along the platform forward axis, so the angles
        describe the camera directly.
        """
        # The rotations apply right to left: optical frame to camera body,
        # camera body to platform body, then platform body to NED.
        body_to_ned = euler_rotation(yaw, pitch, roll, degrees)
        mount_rot = Rotation.identity() if mount is None else _as_rotation(mount)
        cam_to_ned = body_to_ned * mount_rot * CAMERA_TO_BODY

        return cls(float(lat), float(lon), float(alt), cam_to_ned, alt_ref)

    @classmethod
    def from_rotation(
        cls,
        lat: float,
        lon: float,
        alt: float,
        rotation: RotationLike,
        *,
        frame: str = "NED",
        alt_ref: AltRef = "ellipsoid",
    ) -> CameraPose:
        """Pose from a rotation that maps camera optical-frame vectors to ``frame``.

        ``rotation`` is a scipy ``Rotation``, a 3x3 matrix or an
        ``(x, y, z, w)`` quaternion. ``frame`` is ``"NED"``, ``"ENU"`` or
        ``"ECEF"``.
        """
        rot = _as_rotation(rotation)
        frame = frame.upper()

        if frame == "NED":
            cam_to_ned = rot
        elif frame == "ENU":
            cam_to_ned = NED_TO_ENU.inv() * rot
        elif frame == "ECEF":
            # The inverse of a rotation matrix is its transpose.
            ecef_to_ned = Rotation.from_matrix(ned_to_ecef_matrix(lat, lon).T)
            cam_to_ned = ecef_to_ned * rot
        else:
            raise ValueError("frame must be 'NED', 'ENU' or 'ECEF'")

        return cls(float(lat), float(lon), float(alt), cam_to_ned, alt_ref)

    @classmethod
    def look_at(
        cls,
        lat: float,
        lon: float,
        alt: float,
        target_lat: float,
        target_lon: float,
        target_alt: float,
        *,
        roll: float = 0.0,
        degrees: bool = True,
    ) -> CameraPose:
        """Pose at a position with the optical axis pointing at a target.

        Both altitudes are heights above the WGS84 ellipsoid.
        """
        # The target in local NED at the camera. Row vectors: v @ R is the same
        # as R.T @ v, the rotation from ECEF to NED.
        vec = geodetic_to_ecef(target_lat, target_lon, target_alt) - geodetic_to_ecef(lat, lon, alt)
        n, e, d = vec @ ned_to_ecef_matrix(lat, lon)

        # Yaw is the bearing to the target. Pitch is the elevation angle,
        # positive up, so it uses -d.
        yaw = math.degrees(math.atan2(e, n))
        pitch = math.degrees(math.atan2(-d, math.hypot(n, e)))
        r = roll if degrees else math.degrees(roll)

        return cls.from_euler(lat, lon, alt, yaw, pitch, r)

    # ---- Queries ------------------------------------------------------------

    def euler(self, degrees: bool = True) -> FloatArray:
        """Camera yaw, pitch, roll (the optical axis is the body forward axis)."""
        # Remove the fixed optical-to-body rotation. The rest is the body attitude.
        body = self.rotation * CAMERA_TO_BODY.inv()

        return body.as_euler("ZYX", degrees=degrees)

    def camera_to_ecef(self) -> FloatArray:
        """3x3 matrix that maps camera optical-frame vectors to ECEF."""
        return ned_to_ecef_matrix(self.lat, self.lon) @ self.rotation.as_matrix()

    def rays_to_ecef(self, rays_cam: ArrayLike) -> FloatArray:
        """Rotate camera optical-frame directions ``(..., 3)`` into ECEF."""
        # Row vectors: v @ R.T is the same as R @ v for each vector.
        return np.asarray(rays_cam, dtype=float) @ self.camera_to_ecef().T

    def ecef_to_camera(self, vec_ecef: ArrayLike) -> FloatArray:
        """Rotate ECEF vectors ``(..., 3)`` into the camera optical frame."""
        # Row vectors: v @ R is the same as R.T @ v, the inverse rotation.
        return np.asarray(vec_ecef, dtype=float) @ self.camera_to_ecef()

    def with_altitude(self, alt: float, alt_ref: AltRef = "ellipsoid") -> CameraPose:
        """A copy of this pose with a new altitude."""
        return replace(self, alt=float(alt), alt_ref=alt_ref)

    def ellipsoidal_altitude(
        self,
        geoid: SupportsUndulation | None = None,
        terrain: Terrain | None = None,
    ) -> float:
        """The camera height above the WGS84 ellipsoid.

        ``alt_ref == "geoid"`` needs ``geoid``. ``alt_ref == "agl"`` needs
        ``terrain``.
        """
        if self.alt_ref == "ellipsoid":
            return self.alt

        if self.alt_ref == "geoid":
            if geoid is None:
                raise ValueError("alt_ref='geoid' needs a geoid model (give the Terrain a geoid)")
            return self.alt + float(np.asarray(geoid.undulation(self.lat, self.lon)))

        if terrain is None:
            raise ValueError("alt_ref='agl' needs a terrain model")

        ground = float(np.asarray(terrain.height_at(self.lat, self.lon)))
        if not np.isfinite(ground):
            raise ValueError("alt_ref='agl': no terrain data below the camera")

        return self.alt + ground
