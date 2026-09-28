"""Pixel to geographic coordinate: the main ray casting pipeline.

Pipeline
--------
1. The camera intrinsics turn each pixel into a unit ray in the camera frame.
2. The camera pose rotates the ray into ECEF. The ray starts at the camera
   ECEF position.
3. Two ray / ellipsoid intersections clip the ray to the height band of the
   terrain (``h_min`` to ``h_max``). This removes the empty space above the
   terrain, which can be hundreds of kilometers for a satellite.
4. The backend kernel cuts the clipped ray into segments of at most
   ``max_segment_length`` meters and finds the first intersection with the
   bilinear terrain surface. See :mod:`pix2geo.backends._kernel`.
5. The hit parameter maps back onto the exact ECEF ray, and pymap3d
   converts the hit point to latitude, longitude, and height.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._geodesy import ecef_to_geodetic, geodetic_to_ecef, ray_ellipsoid_intersection
from ._typing import ArrayLike, BoolArray, FloatArray
from .backends import Backend, get_backend
from .camera import CameraIntrinsics
from .pose import CameraPose
from .terrain import Terrain

__all__ = ["GeoResult", "Geolocator", "pixel_to_geo"]


@dataclass
class GeoResult:
    """The geolocation result. All arrays have the shape of the input batch.

    Attributes
    ----------
    lat, lon:
        WGS84 geodetic latitude and longitude in degrees. NaN on a miss.
    alt:
        Height above the WGS84 ellipsoid in meters.
    ecef:
        ECEF position in meters, shape ``(..., 3)``.
    range:
        Slant range from the camera in meters.
    hit:
        True where the ray hit the terrain model. False where the ray missed
        or where the fallback surface gave the point.
    alt_msl:
        Orthometric height (above the geoid of the terrain), or ``None``
        when the terrain has no geoid model.
    """

    lat: FloatArray
    lon: FloatArray
    alt: FloatArray
    ecef: FloatArray
    range: FloatArray
    hit: BoolArray
    alt_msl: FloatArray | None = None

    def as_array(self) -> FloatArray:
        """``(..., 3)`` array of ``(lat, lon, alt)``."""
        return np.stack([self.lat, self.lon, self.alt], axis=-1)

    def __getitem__(self, idx: Any) -> GeoResult:
        """Select rays with a NumPy index over the batch axes."""
        return GeoResult(
            lat=self.lat[idx],
            lon=self.lon[idx],
            alt=self.alt[idx],
            ecef=self.ecef[idx],
            range=self.range[idx],
            hit=self.hit[idx],
            alt_msl=None if self.alt_msl is None else self.alt_msl[idx],
        )


class Geolocator:
    """Ray caster that maps camera pixels onto a terrain model.

    Parameters
    ----------
    terrain:
        The elevation model.
    backend:
        ``"auto"``, ``"numba"``, ``"cupy"``, ``"python"`` or a
        :class:`~pix2geo.backends.Backend` instance.
    max_segment_length:
        The longest straight grid-space segment, in meters.
    max_range:
        Optional largest slant range in meters. Rays that do not hit
        within this range miss.
    batch_size:
        The largest number of rays in one kernel call. This limits the
        memory use for very large batches.
    """

    def __init__(
        self,
        terrain: Terrain,
        backend: str | Backend = "auto",
        *,
        max_segment_length: float = 500.0,
        max_range: float | None = None,
        batch_size: int = 1 << 22,
    ) -> None:
        if not max_segment_length > 0:
            raise ValueError("max_segment_length must be positive")
        if max_range is not None and not max_range > 0:
            raise ValueError("max_range must be positive, or None for no limit")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.terrain = terrain
        self.backend = get_backend(backend)
        self.max_segment_length = float(max_segment_length)
        self.max_range = None if max_range is None else float(max_range)
        self.batch_size = int(batch_size)

    # ---- Public API ---------------------------------------------------------

    def camera_position_ecef(self, pose: CameraPose) -> FloatArray:
        """ECEF position of the camera, with the altitude reference resolved."""
        h = pose.ellipsoidal_altitude(self.terrain.geoid, self.terrain)
        return geodetic_to_ecef(pose.lat, pose.lon, h)

    def pixel_to_geo(
        self,
        camera: CameraIntrinsics,
        pose: CameraPose,
        pixels: ArrayLike,
        *,
        fallback_height: float | None = None,
    ) -> GeoResult:
        """Geolocate one pixel ``(u, v)`` or an array of pixels ``(..., 2)``.

        ``fallback_height`` is an optional WGS84 ellipsoidal height. Rays
        that miss the terrain model then hit the ellipsoid grown by that
        height (``hit`` stays False for them).
        """
        px = np.asarray(pixels, dtype=float)
        if px.ndim == 0 or px.shape[-1] != 2:
            raise ValueError("pixels must have shape (2,) or (..., 2)")

        dirs = pose.rays_to_ecef(camera.pixel_to_ray(px[..., 0], px[..., 1]))
        origin = self.camera_position_ecef(pose)

        return self.rays_to_geo(origin, dirs, fallback_height=fallback_height)

    def image_to_geo(
        self,
        camera: CameraIntrinsics,
        pose: CameraPose,
        step: int = 1,
        *,
        fallback_height: float | None = None,
    ) -> GeoResult:
        """Geolocate a regular grid of pixels over the whole image.

        The result has shape ``(ceil(height/step), ceil(width/step))``.
        """
        u = np.arange(0, camera.width, step, dtype=float)
        v = np.arange(0, camera.height, step, dtype=float)
        pixels = np.stack(np.meshgrid(u, v), axis=-1)

        return self.pixel_to_geo(camera, pose, pixels, fallback_height=fallback_height)

    def rays_to_geo(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        *,
        fallback_height: float | None = None,
    ) -> GeoResult:
        """Intersect ECEF rays with the terrain.

        ``origins`` and ``directions`` have shape ``(3,)`` or ``(..., 3)``, and
        they broadcast against each other. Thus many rays can share one
        origin (a camera), and many origins can share one direction (parallel
        rays). ``directions`` does not need unit length.
        """
        o, d = np.broadcast_arrays(
            np.asarray(origins, dtype=float),
            np.asarray(directions, dtype=float),
        )
        if d.shape[-1] != 3:
            raise ValueError("origins and directions must have shape (3,) or (..., 3)")

        # Flatten the batch. Unit directions make the ray parameter the range in meters.
        shape = d.shape[:-1]
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)

        # Rays that miss the terrain can hit the fallback ellipsoid. A hit
        # behind the camera (t <= 0) does not count.
        t_hit = self._trace(o, d)
        hit = np.isfinite(t_hit)
        if fallback_height is not None and not hit.all():
            miss = ~hit
            t_near, _ = ray_ellipsoid_intersection(o[miss], d[miss], float(fallback_height))
            t_hit[miss] = np.where(t_near > 0, t_near, np.nan)

        points = o + t_hit[:, None] * d
        lat, lon, alt = ecef_to_geodetic(points)
        alt_msl = self._orthometric(lat, lon, alt)

        return GeoResult(
            lat=lat.reshape(shape),
            lon=lon.reshape(shape),
            alt=alt.reshape(shape),
            ecef=points.reshape((*shape, 3)),
            range=t_hit.reshape(shape),
            hit=hit.reshape(shape),
            alt_msl=None if alt_msl is None else alt_msl.reshape(shape),
        )

    def geo_to_pixel(
        self,
        camera: CameraIntrinsics,
        pose: CameraPose,
        lat: ArrayLike,
        lon: ArrayLike,
        alt: ArrayLike,
        *,
        check_occlusion: bool = False,
        tolerance: float = 1.0,
    ) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Project geographic points into the image.

        ``alt`` is the WGS84 ellipsoidal height. Returns ``(u, v, visible)``.
        ``visible`` is True for points in front of the camera and inside the
        image. With ``check_occlusion=True``, ``visible`` is also False when
        the terrain blocks the line of sight by more than ``tolerance`` meters.
        """
        # The line of sight from the camera to each point, in the camera frame.
        cam = self.camera_position_ecef(pose)
        vec = geodetic_to_ecef(lat, lon, alt) - cam
        vec_cam = pose.ecef_to_camera(vec)
        u, v = camera.ray_to_pixel(vec_cam)
        visible = (vec_cam[..., 2] > 0) & camera.contains(u, v)

        # Cast a ray to each point. A terrain hit before the point blocks it.
        if check_occlusion and np.any(visible):
            first = self.rays_to_geo(cam, vec)
            blocked = first.hit & (first.range < np.linalg.norm(vec, axis=-1) - tolerance)
            visible &= ~blocked

        return u, v, visible

    # ---- Internals ----------------------------------------------------------

    def _clip_to_height_band(self, o: FloatArray, d: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The ray parameter interval ``[t0, t1]`` inside the height band of the terrain.

        The band is the shell between the ellipsoids grown by ``h_min - 1``
        and ``h_max + 1``. The interval ends where the ray first enters the
        lower ellipsoid. A ray that misses the band gets NaN.
        """
        top_in, top_out = ray_ellipsoid_intersection(o, d, self.terrain.h_max + 1.0)
        bottom_in, _ = ray_ellipsoid_intersection(o, d, self.terrain.h_min - 1.0)

        # Start where the ray enters the top ellipsoid, or at the camera when the
        # camera is already inside it.
        t0 = np.fmax(top_in, 0.0)

        # End where the ray leaves the top ellipsoid, or earlier where it enters
        # the bottom ellipsoid in front of the camera.
        t1 = np.where(bottom_in > 0, np.fmin(top_out, bottom_in), top_out)

        if self.max_range is not None:
            t1 = np.fmin(t1, self.max_range)

        return t0, t1

    def _trace(self, o: FloatArray, d: FloatArray) -> FloatArray:
        """Ray parameter of the first terrain hit (NaN for a miss)."""
        t_hit = np.full(len(o), np.nan)
        t0, t1 = self._clip_to_height_band(o, d)

        # Only rays with a non-empty interval go to the backend.
        rays = np.flatnonzero(t1 > t0)  # NaN compares False

        # The fewest equal segments that are each at most max_segment_length long.
        n_seg = np.maximum(np.ceil((t1[rays] - t0[rays]) / self.max_segment_length), 1)
        n_seg = n_seg.astype(np.int64)

        # Batches limit the memory of each backend call.
        for start in range(0, rays.size, self.batch_size):
            part = slice(start, start + self.batch_size)
            batch = rays[part]
            t_hit[batch] = self.backend.trace_rays(
                self.terrain,
                o[batch],
                d[batch],
                t0[batch],
                t1[batch],
                n_seg[part],
            )

        return t_hit

    def _orthometric(self, lat: FloatArray, lon: FloatArray, alt: FloatArray) -> FloatArray | None:
        """Heights above the terrain geoid, or None when the terrain has no geoid."""
        if self.terrain.geoid is None:
            return None

        alt_msl = np.full_like(alt, np.nan)
        ok = np.isfinite(lat)
        if ok.any():
            alt_msl[ok] = alt[ok] - self.terrain.geoid.undulation(lat[ok], lon[ok])

        return alt_msl


def pixel_to_geo(
    pixels: ArrayLike,
    camera: CameraIntrinsics,
    pose: CameraPose,
    terrain: Terrain,
    *,
    backend: str | Backend = "auto",
    fallback_height: float | None = None,
    **kwargs: Any,
) -> GeoResult:
    """One-call form of :meth:`Geolocator.pixel_to_geo`.

    A single pixel ``(u, v)`` gives 0-d result arrays. Use ``float()`` on them.
    ``kwargs`` go to :class:`Geolocator`.
    """
    geo = Geolocator(terrain, backend=backend, **kwargs)
    return geo.pixel_to_geo(camera, pose, pixels, fallback_height=fallback_height)
