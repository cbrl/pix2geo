"""pix2geo: camera pixel to geographic coordinate on a terrain model.

Example
-------
>>> from pix2geo import CameraIntrinsics, CameraPose, Terrain, Geolocator
>>> cam = CameraIntrinsics.from_fov(1920, 1080, hfov=60)
>>> pose = CameraPose.from_euler(46.85, -121.76, 4500.0, yaw=45, pitch=-30, roll=0)
>>> terrain = Terrain.from_file("dem.tif", geoid="egm96")       # doctest: +SKIP
>>> geo = Geolocator(terrain, backend="auto")                    # doctest: +SKIP
>>> res = geo.pixel_to_geo(cam, pose, (960, 540))                # doctest: +SKIP
>>> float(res.lat), float(res.lon), float(res.alt)               # doctest: +SKIP
"""

from ._geodesy import ray_ellipsoid_intersection
from .backends import Backend, BackendUnavailableError, available_backends, get_backend
from .camera import CameraIntrinsics
from .geoid import ConstantGeoid, GeoidModel, PyprojGeoid, RasterGeoid, resolve_geoid
from .geolocator import Geolocator, GeoResult, pixel_to_geo
from .pose import CAMERA_TO_BODY, CameraPose, euler_rotation
from .quadtree import HeightPyramid
from .terrain import Terrain

__version__ = "0.1.0"

__all__ = [
    # Core pipeline
    "CameraIntrinsics",
    "CameraPose",
    "GeoResult",
    "Geolocator",
    "Terrain",
    "pixel_to_geo",

    # Orientation helpers
    "CAMERA_TO_BODY",
    "euler_rotation",

    # Geoid models
    "ConstantGeoid",
    "GeoidModel",
    "PyprojGeoid",
    "RasterGeoid",
    "resolve_geoid",

    # Backends
    "Backend",
    "BackendUnavailableError",
    "available_backends",
    "get_backend",

    # Building blocks
    "HeightPyramid",
    "ray_ellipsoid_intersection",
]
