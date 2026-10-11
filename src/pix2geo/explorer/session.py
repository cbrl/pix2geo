"""The computations of an explorer session.

Each public method of :class:`Session` takes the decoded JSON body of one
request and returns a JSON-ready dict. The page owns the camera and the pose,
and it sends them with each request. Thus the session keeps only the terrain,
the geolocator and the fixed scene frame.

The scene frame is the local ENU frame of :mod:`._geometry`, with its origin
on the ellipsoid below the first camera position. The origin does not move
with the camera, so the scene does not jump when the pose changes.

Large arrays travel as base64 text of little-endian float32 or uint8 data.
Small arrays travel as JSON lists, with null for NaN.
"""

from __future__ import annotations

import base64
import math
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyproj import Geod, Transformer

from .._geodesy import ecef_to_geodetic, geodetic_to_ecef, ray_ellipsoid_intersection
from .._typing import ArrayLike, FloatArray, IntArray
from ..backends import AutoBackend, available_backends
from ..camera import CameraIntrinsics
from ..geolocator import Geolocator, GeoResult
from ..pose import CameraPose
from ..terrain import Terrain
from ._geometry import (
    Window,
    bounds_window,
    enu_frame,
    image_border,
    pixel_grid,
    post_window,
)
from ._geometry import to_enu as ecef_to_enu

__all__ = ["JSON", "Session"]

#: A decoded JSON object.
JSON = dict[str, Any]

#: The mean earth radius in meters. It converts small distances to angles.
_EARTH_RADIUS = 6_371_008.8

#: The number of terrain meshes that the session keeps for the viewshed.
_MESH_CACHE = 3

#: The pixel offsets to the right and the lower neighbor.
_RIGHT = np.array([1.0, 0.0])
_DOWN = np.array([0.0, 1.0])

#: The camera-view layers of :meth:`Session.image`.
_LAYERS = ("range", "height", "status", "gsd", "shade", "sensitivity")


@dataclass
class _Mesh:
    """The posts of one drawn terrain mesh, shape ``(rows, cols)``."""

    lat: FloatArray
    lon: FloatArray
    height: FloatArray
    enu: FloatArray


@dataclass
class _HitAngles:
    """The geometry of rays at their hit points. All angles are in degrees."""

    slope: FloatArray
    grazing: FloatArray
    sensitivity: FloatArray


class Session:
    """The state of one explorer: a terrain, a geolocator and the start view.

    Parameters
    ----------
    geolocator:
        The geolocator, or a terrain for a default geolocator. The session
        makes its own geolocator with the same settings. Thus a change of
        backend in the page does not change the given geolocator.
    camera, pose:
        The camera and the pose that the page starts with. The pose also
        sets the origin of the scene frame.
    fallback_height:
        The fallback height that the page starts with. None turns it off.
    max_rays:
        The number of rays in the drawn fan that the page starts with.
    terrain_size:
        The number of drawn terrain posts on each axis that the page starts with.
    sources:
        The DEM paths, for the command line that the page makes.
    geoid:
        The ``--geoid`` value of the DEM, for the same command line.
    """

    def __init__(
        self,
        geolocator: Geolocator | Terrain,
        camera: CameraIntrinsics,
        pose: CameraPose,
        *,
        fallback_height: float | None = None,
        max_rays: int = 256,
        terrain_size: int = 512,
        sources: list[str] | None = None,
        geoid: str | float | None = None,
    ) -> None:
        if isinstance(geolocator, Terrain):
            geolocator = Geolocator(geolocator)

        self.terrain = geolocator.terrain
        self.geolocator = Geolocator(
            self.terrain,
            geolocator.backend,
            max_segment_length=geolocator.max_segment_length,
            max_range=geolocator.max_range,
            batch_size=geolocator.batch_size,
        )
        self.camera = camera
        self.pose = pose
        self.fallback_height = fallback_height
        self.max_rays = int(max_rays)
        self.terrain_size = int(terrain_size)
        self.sources = list(sources) if sources else []
        self.geoid_arg = geoid

        # The fixed scene frame: its ECEF origin and its axes as columns.
        self.origin = (pose.lat, pose.lon)
        self._origin_ecef, self._enu = enu_frame(pose.lat, pose.lon)

        self._geod = Geod(ellps="WGS84")
        self._utm: dict[int, Transformer] = {}
        self._meshes: OrderedDict[str, _Mesh] = OrderedDict()
        self._post_spacing = _post_spacing(self.terrain)

    # ---- Requests -----------------------------------------------------------

    def info(self) -> JSON:
        """The terrain, the start view and the settings."""
        terrain = self.terrain
        rows, cols = terrain.shape
        geoid = terrain.geoid

        return {
            "terrain": {
                "summary": repr(terrain),
                "shape": [rows, cols],
                "crs": terrain.crs.to_string(),
                "h_min": terrain.h_min,
                "h_max": terrain.h_max,
                "geoid": None if geoid is None else repr(geoid),
            },
            "origin": {"lat": self.origin[0], "lon": self.origin[1]},
            "camera": _camera_json(self.camera),
            "pose": _pose_json(self.pose),
            "fallback_height": self.fallback_height,
            "max_rays": self.max_rays,
            "terrain_size": self.terrain_size,
            "settings": self._settings_json(),
            "backends": ["auto", *available_backends()],
            "cli": {"dem": self.sources, "geoid": self.geoid_arg},
        }

    def terrain_mesh(self, req: JSON) -> JSON:
        """A terrain mesh: the whole DEM, the area around the view, or WGS84 bounds.

        The mesh uses every k-th post, so it has at most ``size`` posts on
        each axis. The page keeps the mesh id for :meth:`viewshed`.
        """
        terrain = self.terrain
        rows, cols = terrain.shape
        size = int(np.clip(int(req.get("size", self.terrain_size)), 2, 2048))
        extent = req.get("extent", "all")
        full: Window = (0, cols - 1, 0, rows - 1)

        # The area around the view holds the camera and the ray ends.
        if extent == "view":
            camera, pose, fallback = self._view(req)
            border, _ = image_border(camera, 16)
            pixels = np.concatenate([border, pixel_grid(camera, 256)])
            res = self.geolocator.pixel_to_geo(camera, pose, pixels, fallback_height=fallback)
            lat = np.r_[pose.lat, res.lat]
            lon = np.r_[pose.lon, res.lon]
            window = post_window(terrain, lon, lat, margin=0.3) or full
        elif extent == "bounds":
            w, s, e, n = (float(x) for x in req["bounds"])
            window = bounds_window(terrain, (w, s, e, n))
        else:
            window = full

        # Every k-th post of the window, with its last row and column.
        c0, c1, r0, r1 = window
        step = max(1, math.ceil(max(c1 - c0, r1 - r0) / (size - 1)))
        col_idx = np.unique(np.r_[np.arange(c0, c1 + 1, step), c1])
        row_idx = np.unique(np.r_[np.arange(r0, r1 + 1, step), r1])

        # The posts in the scene frame. A nodata post gives a NaN point.
        h = terrain.heights[np.ix_(row_idx, col_idx)].astype(float)
        gx, gy = np.meshgrid(col_idx.astype(float), row_idx.astype(float))
        lon, lat = terrain.grid_to_lonlat(gx, gy)
        enu = self.to_enu(geodetic_to_ecef(lat, lon, h))

        # The colors use heights above the geoid when the terrain has one.
        shown = h
        label = "Ellipsoidal height"
        if terrain.geoid is not None:
            shown = h - terrain.geoid.undulation(lat, lon)
            label = "Height above geoid"

        key = f"{c0}-{c1}-{r0}-{r1}-{step}"
        self._meshes[key] = _Mesh(lat=lat, lon=lon, height=h, enu=enu)
        self._meshes.move_to_end(key)
        while len(self._meshes) > _MESH_CACHE:
            self._meshes.popitem(last=False)

        ok = np.isfinite(h)
        return {
            "id": key,
            "rows": len(row_idx),
            "cols": len(col_idx),
            "step": step,
            "spacing": self._post_spacing * step,
            "positions": _b64(enu),
            "height": _b64(shown),
            "height_label": label,
            "height_range": [float(np.nanmin(shown)), float(np.nanmax(shown))],
            "lonlat": _b64(np.stack([lon - self.origin[1], lat - self.origin[0]], axis=-1)),
            "bounds": [
                float(lon[ok].min()),
                float(lat[ok].min()),
                float(lon[ok].max()),
                float(lat[ok].max()),
            ],
        }

    def cast(self, req: JSON) -> JSON:
        """Cast the picked pixels, a fan of rays and the image border.

        ``pixels`` and ``ids`` are the picks. ``grid`` is the number of fan
        rays, and ``edge_points`` the number of border rays on each image
        edge. With ``detail``, each pick also gets its ground sample
        distance, its angles and its UTM position, and the response has the
        same values for the image center.
        """
        camera, pose, fallback = self._view(req)
        picks = _pixels(req.get("pixels"))
        ids = list(req.get("ids") or range(len(picks)))
        if len(ids) != len(picks):
            raise ValueError("ids must have one entry for each pixel")
        detail = bool(req.get("detail", True))
        n_grid = int(req.get("grid", 0))
        edge = int(req.get("edge_points", 0))

        # The probes are the picks and the image center. Their right and lower
        # neighbors give the ground sample distance.
        probes = np.concatenate([picks, [[camera.cx, camera.cy]]]) if detail else picks
        grid = pixel_grid(camera, n_grid) if n_grid > 0 else np.empty((0, 2))
        border, corners = (
            image_border(camera, edge)
            if edge > 0
            else (np.empty((0, 2)), np.empty(0, dtype=np.int64))
        )
        neighbors = [probes + _RIGHT, probes + _DOWN] if detail else []
        pixels = np.concatenate([probes, *neighbors, grid, border])

        # One call for all the rays.
        start = time.perf_counter()
        res = self.geolocator.pixel_to_geo(camera, pose, pixels, fallback_height=fallback)
        ms = 1e3 * (time.perf_counter() - start)

        # Split the result into its parts.
        n_probe, n_grid = len(probes), len(grid)
        parts = np.cumsum([n_probe, n_probe * len(neighbors), n_grid])
        probe_res = res[: parts[0]]
        border_res = res[parts[2] :]

        # The ray directions in the scene frame, and the ray ends. A ray that
        # misses ends at the median range of the rays that land.
        cam = self._camera_state(pose)
        dirs = pose.rays_to_ecef(camera.pixel_to_ray(pixels[:, 0], pixels[:, 1])) @ self._enu
        landed = np.isfinite(res.range)
        if landed.any():
            miss_length = float(np.median(res.range[landed]))
        else:
            miss_length = max(1000.0, 2.0 * (cam["agl"] or 0.0))
        length = np.where(landed, res.range, miss_length)
        ends = np.asarray(cam["position"]) + dirs * length[:, None]
        status = _status(res)

        out: JSON = {
            "ms": ms,
            "rays": len(pixels),
            "backend": self._backend_name(len(pixels)),
            "camera": cam,
            "miss_length": miss_length,
            "grid": {
                "pixels": _nums(grid, 2),
                "ends": _nums(ends[parts[1] : parts[2]], 3),
                "status": status[parts[1] : parts[2]].tolist(),
            },
            "border": {
                "ends": _nums(ends[parts[2] :], 3),
                "dirs": _nums(dirs[parts[2] :], 7),
                "status": status[parts[2] :].tolist(),
                "corners": [int(c) for c in corners],
                "lonlat": _nums(np.stack([border_res.lon, border_res.lat], axis=-1), 9),
                "area": self._footprint_area(border_res),
            },
        }

        # The picks, then the image center.
        rows = [
            {
                "id": ids[k] if k < len(ids) else None,
                "u": float(probes[k, 0]),
                "v": float(probes[k, 1]),
                "status": _STATUS_NAMES[status[k]],
                "lat": _num(probe_res.lat[k]),
                "lon": _num(probe_res.lon[k]),
                "alt": _num(probe_res.alt[k], 3),
                "alt_msl": None if probe_res.alt_msl is None else _num(probe_res.alt_msl[k], 3),
                "range": _num(probe_res.range[k], 3),
                "end": _nums(ends[k], 3),
                "ecef": _nums(probe_res.ecef[k], 3),
            }
            for k in range(n_probe)
        ]
        if detail:
            self._add_detail(rows, pose, res, n_probe)
            out["center"] = rows.pop()
            out["measure"] = self._measure(rows)
        out["picks"] = rows

        return out

    def locate(self, req: JSON) -> JSON:
        """Cast a ray of the 3D view, and project its terrain hit into the camera.

        ``origin`` and ``direction`` are in the scene frame. The response
        says if the point is in front of the camera, inside the image, and
        not hidden by the terrain.
        """
        camera, pose, _ = self._view(req)
        origin = self.to_ecef(np.asarray(req["origin"], dtype=float))
        direction = np.asarray(req["direction"], dtype=float) @ self._enu.T
        hit = self.geolocator.rays_to_geo(origin, direction)
        if not bool(hit.hit):
            return {"hit": False}

        lat, lon, alt = float(hit.lat), float(hit.lon), float(hit.alt)
        geo = self.geolocator
        u, v, in_view = geo.geo_to_pixel(camera, pose, lat, lon, alt)
        _, _, visible = geo.geo_to_pixel(camera, pose, lat, lon, alt, check_occlusion=True)

        # In front of the camera: a positive depth along the optical axis.
        vec = geodetic_to_ecef(lat, lon, alt) - geo.camera_position_ecef(pose)
        depth = float(pose.ecef_to_camera(vec)[2])

        return {
            "hit": True,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "alt_msl": None if hit.alt_msl is None else _num(hit.alt_msl, 3),
            "enu": _nums(self.to_enu(hit.ecef), 3),
            "pixel": None if depth <= 0 else [float(u), float(v)],
            "in_front": depth > 0,
            "in_image": bool(in_view),
            "visible": bool(visible),
            "distance": float(np.linalg.norm(vec)),
        }

    def profile(self, req: JSON) -> JSON:
        """The ray height and the terrain height along the ray of one pixel.

        The samples run from the camera to a little past the hit. A ray that
        misses runs until it rises above the highest terrain, or ``length``
        meters when it never does. The response also gives the smallest
        clearance of the ray above the terrain before its final approach to
        the hit. A small clearance marks a ray that almost hit a ridge, where
        a small error moves the hit far.
        """
        camera, pose, fallback = self._view(req)
        u, v = (float(x) for x in req["pixel"])
        geo = self.geolocator
        origin = geo.camera_position_ecef(pose)
        direction = pose.rays_to_ecef(camera.pixel_to_ray(u, v))
        hit = geo.rays_to_geo(origin, direction, fallback_height=fallback)
        t_hit = float(hit.range)

        # A ray that misses ends where it leaves the terrain band at the top.
        if math.isfinite(t_hit):
            length = 1.15 * t_hit
        else:
            _, t_top = ray_ellipsoid_intersection(origin, direction, self.terrain.h_max + 20.0)
            length = float(t_top) if t_top > 0 else float(req.get("length", 5000.0))
            length = min(max(length, 100.0), 200_000.0)

        # Samples at about half the post spacing, between 200 and 4000.
        n = int(np.clip(length / (0.5 * self._post_spacing), 200, 4000))
        t = np.linspace(0.0, length, n)

        # The heights of the ray and of the terrain below each sample.
        lat, lon, h_ray = ecef_to_geodetic(origin + t[:, None] * direction)
        h_ground = self.terrain.height_at(lat, lon)
        label = "Ellipsoidal height"
        if self.terrain.geoid is not None:
            undulation = self.terrain.geoid.undulation(lat, lon)
            h_ray = h_ray - undulation
            h_ground = h_ground - undulation
            label = "Height above geoid"

        # The horizontal distance from the camera to each sample.
        _, _, ground = self._geod.inv(np.full(n, pose.lon), np.full(n, pose.lat), lon, lat)

        return {
            "t": _nums(t, 2),
            "ground": _nums(ground, 2),
            "ray": _nums(h_ray, 2),
            "terrain": _nums(h_ground, 2),
            "label": label,
            "hit": {
                "range": _num(t_hit, 3),
                "terrain": bool(hit.hit),
                "height": _num(hit.alt if hit.alt_msl is None else hit.alt_msl, 2),
            },
            "clearance": _near_miss(t, h_ray - h_ground, t_hit),
        }

    def image(self, req: JSON) -> JSON:
        """Geolocate a grid of pixels over the image, and return some layers of it.

        ``step`` is the pixel step of the grid. ``layers`` names the layers:

        - ``range``: the slant range in meters.
        - ``height``: the hit height, above the geoid when the terrain has one.
        - ``status``: 0 for a miss, 1 for a terrain hit, 2 for a fallback hit.
        - ``gsd``: the ground sample distance in meters per pixel.
        - ``shade``: the terrain lighting at the hit, from 0 to 1, for a sun
          at ``sun`` (azimuth and elevation in degrees).
        - ``sensitivity``: the horizontal move of the hit, in meters, for
          1 m of DEM height error.
        """
        camera, pose, fallback = self._view(req)
        step = max(1, int(req.get("step", 8)))
        names = list(req.get("layers") or ["range"])
        unknown = set(names) - set(_LAYERS)
        if unknown:
            raise ValueError(f"unknown layers {sorted(unknown)}; use {list(_LAYERS)}")

        start = time.perf_counter()
        res = self.geolocator.image_to_geo(camera, pose, step, fallback_height=fallback)
        ms = 1e3 * (time.perf_counter() - start)
        rows, cols = res.lat.shape

        layers: JSON = {}
        angles = None
        for name in names:
            if name == "status":
                layers[name] = {"data": _b64(_status(res), "<u1")}
                continue

            if name == "range":
                values = res.range
            elif name == "height":
                values = res.alt if res.alt_msl is None else res.alt_msl
            elif name == "gsd":
                values = _grid_gsd(res.ecef, step)
            elif name == "shade":
                values = self._shade(res, req.get("sun", (315.0, 45.0)))
            else:
                if angles is None:
                    origin = self.geolocator.camera_position_ecef(pose)
                    angles = self._hit_angles(res, origin)
                values = angles.sensitivity
            layers[name] = {"data": _b64(values), **_spread(values)}

        return {
            "rows": rows,
            "cols": cols,
            "step": step,
            "ms": ms,
            "rays": rows * cols,
            "backend": self._backend_name(rows * cols),
            "layers": layers,
        }

    def viewshed(self, req: JSON) -> JSON:
        """The camera pixel of each post of a terrain mesh, and if the camera sees it.

        A post is visible when it is in front of the camera, inside the
        image, and not hidden by the terrain. The response also gives the
        surface area of the visible terrain.
        """
        camera, pose, _ = self._view(req)
        mesh = self._meshes.get(str(req.get("mesh")))
        if mesh is None:
            raise ValueError("unknown terrain mesh; reload the terrain")

        start = time.perf_counter()
        u, v, visible = self.geolocator.geo_to_pixel(
            camera,
            pose,
            mesh.lat,
            mesh.lon,
            mesh.height,
            check_occlusion=True,
        )
        ms = 1e3 * (time.perf_counter() - start)

        # A cell counts when its four posts are visible. Its area is half the
        # cross product of its diagonals.
        p = mesh.enu
        d1 = p[1:, 1:] - p[:-1, :-1]
        d2 = p[1:, :-1] - p[:-1, 1:]
        area = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=-1)
        cells = visible[1:, 1:] & visible[:-1, :-1] & visible[1:, :-1] & visible[:-1, 1:]

        return {
            "mesh": req.get("mesh"),
            "ms": ms,
            "uv": _b64(np.stack([u, v], axis=-1)),
            "visible": _b64(visible, "<u1"),
            "count": int(np.count_nonzero(visible)),
            "area": float(np.nansum(area[cells])),
        }

    def look_at(self, req: JSON) -> JSON:
        """The yaw, pitch and roll that point the camera at ``target`` (lat, lon, alt)."""
        _, pose, _ = self._view(req)
        lat, lon, alt = (float(x) for x in req["target"])
        h = pose.ellipsoidal_altitude(self.terrain.geoid, self.terrain)
        roll = float(_pose_json(pose)["roll"])
        aimed = CameraPose.look_at(pose.lat, pose.lon, h, lat, lon, alt, roll=roll)

        return {k: v for k, v in _pose_json(aimed).items() if k in ("yaw", "pitch", "roll")}

    def settings(self, req: JSON) -> JSON:
        """Change the backend, the segment length or the largest range."""
        geo = self.geolocator
        backend = req.get("backend", geo.backend.name)
        max_range = req.get("max_range", geo.max_range)
        segment = float(req.get("max_segment_length", geo.max_segment_length))

        self.geolocator = Geolocator(
            self.terrain,
            backend,
            max_segment_length=segment,
            max_range=None if max_range in (None, "", 0) else float(max_range),
            batch_size=geo.batch_size,
        )

        return self._settings_json()

    def benchmark(self, req: JSON) -> JSON:
        """Time a full frame of the camera on each backend.

        The Python backend is slow, so it uses a pixel step that gives about
        5000 rays. Each backend first traces a few rays, so the compile time
        and the DEM upload do not count. Each time is the best of two runs.
        """
        camera, pose, fallback = self._view(req)
        geo = self.geolocator
        rows = []
        for name in available_backends():
            bench = Geolocator(
                self.terrain,
                name,
                max_segment_length=geo.max_segment_length,
                max_range=geo.max_range,
            )
            step = 1
            if name == "python":
                step = max(1, math.ceil(math.sqrt(camera.width * camera.height / 5000)))

            bench.pixel_to_geo(camera, pose, [[camera.cx, camera.cy]] * 64)
            best = math.inf
            for _ in range(2):
                start = time.perf_counter()
                res = bench.image_to_geo(camera, pose, step, fallback_height=fallback)
                best = min(best, time.perf_counter() - start)

            rays = int(res.lat.size)
            rows.append(
                {
                    "backend": name,
                    "step": step,
                    "rays": rays,
                    "seconds": best,
                    "rays_per_second": rays / best,
                    "hits": float(np.count_nonzero(res.hit)) / rays,
                }
            )

        return {"results": rows}

    # ---- Frames -------------------------------------------------------------

    def to_enu(self, points: ArrayLike) -> FloatArray:
        """ECEF points ``(..., 3)`` in the scene frame."""
        return ecef_to_enu(points, self._origin_ecef, self._enu)

    def to_ecef(self, points: ArrayLike) -> FloatArray:
        """Scene-frame points ``(..., 3)`` in ECEF."""
        return self._origin_ecef + np.asarray(points, dtype=float) @ self._enu.T

    # ---- Internals ----------------------------------------------------------

    def _view(self, req: JSON) -> tuple[CameraIntrinsics, CameraPose, float | None]:
        """The camera, the pose and the fallback height of a request."""
        camera = _camera_from(req["camera"]) if "camera" in req else self.camera
        pose = _pose_from(req["pose"]) if "pose" in req else self.pose
        fallback = req.get("fallback_height")

        return camera, pose, None if fallback is None else float(fallback)

    def _camera_state(self, pose: CameraPose) -> JSON:
        """The camera position, axes, heights and horizon in the scene frame."""
        terrain = self.terrain
        position = self.to_enu(self.geolocator.camera_position_ecef(pose))

        # The rows of the transposed camera-to-ECEF matrix are the camera axes.
        # The local frame at the camera differs a little from the scene frame.
        axes = pose.camera_to_ecef().T @ self._enu
        local = _local_axes(pose.lat, pose.lon) @ self._enu

        # The heights of the camera, and the terrain point below it.
        h = pose.ellipsoidal_altitude(terrain.geoid, terrain)
        h_ground = float(terrain.height_at(pose.lat, pose.lon))
        ground = None
        if math.isfinite(h_ground):
            ground = _nums(self.to_enu(geodetic_to_ecef(pose.lat, pose.lon, h_ground)), 3)
        h_msl = None
        if terrain.geoid is not None:
            h_msl = h - float(np.asarray(terrain.geoid.undulation(pose.lat, pose.lon)))

        # The dip of the horizon of the ellipsoid below the local horizontal.
        radius = _mean_radius(pose.lat)
        dip = math.degrees(math.acos(radius / (radius + h))) if h > 0 else 0.0

        return {
            **_pose_json(pose),
            "position": _nums(position, 3),
            "axes": _nums(axes, 9),
            "local": _nums(local, 9),
            "ground": ground,
            "alt_ellipsoid": h,
            "alt_msl": h_msl,
            "agl": _num(h - h_ground, 3),
            "dip": dip,
        }

    def _add_detail(
        self,
        rows: list[JSON],
        pose: CameraPose,
        res: GeoResult,
        n: int,
    ) -> None:
        """Add the GSD, the angles and the UTM position to the probe rows.

        ``res`` starts with the ``n`` probes, then their right neighbors, then
        their lower neighbors.
        """
        probe = res[:n]
        right = res[n : 2 * n]
        below = res[2 * n : 3 * n]

        # The ground sample distance: the hit spacing of neighbor pixels.
        gsd_u = np.linalg.norm(right.ecef - probe.ecef, axis=-1)
        gsd_v = np.linalg.norm(below.ecef - probe.ecef, axis=-1)

        # The horizontal distance and the bearing from the camera.
        ok = np.isfinite(probe.lat)
        lon0 = np.full(n, pose.lon)
        lat0 = np.full(n, pose.lat)
        azimuth, _, ground = self._geod.inv(
            lon0,
            lat0,
            np.where(ok, probe.lon, pose.lon),
            np.where(ok, probe.lat, pose.lat),
        )

        # The angle of each ray below the local horizontal at the camera.
        origin = self.geolocator.camera_position_ecef(pose)
        up = _local_axes(pose.lat, pose.lon)[2]
        dirs = probe.ecef - origin
        with np.errstate(invalid="ignore"):
            depression = np.degrees(np.arcsin(-(dirs @ up) / np.linalg.norm(dirs, axis=-1)))
        angles = self._hit_angles(probe, origin)

        for k, row in enumerate(rows):
            if not ok[k]:
                continue
            row.update(
                gsd=[_num(gsd_u[k], 4), _num(gsd_v[k], 4)],
                ground_distance=_num(ground[k], 3),
                azimuth=_num(azimuth[k] % 360.0, 3),
                depression=_num(depression[k], 4),
                slope=_num(angles.slope[k], 3),
                grazing=_num(angles.grazing[k], 3),
                sensitivity=_num(angles.sensitivity[k], 4),
                utm=self._utm_json(float(probe.lat[k]), float(probe.lon[k])),
            )

    def _hit_angles(self, res: GeoResult, origin: FloatArray) -> _HitAngles:
        """The terrain slope, the grazing angle and the DEM sensitivity at each hit.

        The grazing angle is the angle between the ray and the terrain
        surface. A height error ``dh`` of the DEM moves the surface by
        ``dh cos(slope)`` along its normal, so the hit moves along the ray by
        that amount over ``sin(grazing)``. The horizontal part of that move,
        for ``dh = 1 m``, is the sensitivity.
        """
        lat = np.ravel(res.lat)
        lon = np.ravel(res.lon)
        shape = np.shape(res.lat)
        ok = np.isfinite(lat)

        slope = np.full(lat.shape, np.nan)
        grazing = np.full(lat.shape, np.nan)
        sensitivity = np.full(lat.shape, np.nan)
        if ok.any():
            # The unit ray directions in the local ENU frame at each hit.
            d = np.reshape(res.ecef, (-1, 3))[ok] - origin
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            d_local = np.einsum("nij,nj->ni", _local_axes(lat[ok], lon[ok]), d)

            # The angles between the ray, the surface and the horizontal.
            normal = self._surface_normals(lat[ok], lon[ok])
            sin_grazing = -np.einsum("ni,ni->n", d_local, normal)
            cos_depression = np.hypot(d_local[:, 0], d_local[:, 1])
            slope[ok] = np.degrees(np.arccos(np.clip(normal[:, 2], -1.0, 1.0)))
            grazing[ok] = np.degrees(np.arcsin(np.clip(sin_grazing, -1.0, 1.0)))
            with np.errstate(divide="ignore", invalid="ignore"):
                move = normal[:, 2] * cos_depression / sin_grazing
            sensitivity[ok] = np.where(sin_grazing > 1e-6, move, np.nan)

        return _HitAngles(
            slope=slope.reshape(shape),
            grazing=grazing.reshape(shape),
            sensitivity=sensitivity.reshape(shape),
        )

    def _surface_normals(self, lat: FloatArray, lon: FloatArray) -> FloatArray:
        """The unit terrain normals ``(n, 3)`` in the local ENU frame at each point.

        Central differences of the bilinear surface, one post spacing apart.
        """
        d = self._post_spacing
        dlat = math.degrees(d / _EARTH_RADIUS)
        dlon = dlat / np.cos(np.radians(lat))
        height = self.terrain.height_at

        de = (height(lat, lon + dlon) - height(lat, lon - dlon)) / (2.0 * d)
        dn = (height(lat + dlat, lon) - height(lat - dlat, lon)) / (2.0 * d)
        normal = np.stack([-de, -dn, np.ones_like(de)], axis=-1)

        return normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    def _shade(self, res: GeoResult, sun: Any) -> FloatArray:
        """Lambert lighting of the terrain at each hit, for a sun at (azimuth, elevation)."""
        az, el = (math.radians(float(x)) for x in sun)
        light = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])

        shade = np.full(np.shape(res.lat), np.nan)
        ok = np.isfinite(res.lat) & res.hit
        if ok.any():
            normal = self._surface_normals(res.lat[ok], res.lon[ok])
            shade[ok] = np.clip(normal @ light, 0.0, 1.0)

        return shade

    def _measure(self, rows: list[JSON]) -> JSON:
        """The geodesic path through the picks that land, and the area inside it."""
        points = [r for r in rows if r["lat"] is not None]
        out: JSON = {"segments": [], "length": 0.0, "area": None, "perimeter": None}
        if len(points) < 2:
            return out

        lat = np.array([r["lat"] for r in points])
        lon = np.array([r["lon"] for r in points])
        azimuth, _, dist = self._geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])

        out["segments"] = [
            {
                "from": points[k]["id"],
                "to": points[k + 1]["id"],
                "distance": float(dist[k]),
                "azimuth": float(azimuth[k] % 360.0),
            }
            for k in range(len(points) - 1)
        ]
        out["length"] = float(np.sum(dist))
        if len(points) >= 3:
            area, perimeter = self._geod.polygon_area_perimeter(lon, lat)
            out["area"] = abs(float(area))
            out["perimeter"] = float(perimeter)

        return out

    def _footprint_area(self, border: GeoResult) -> float | None:
        """The area of the image footprint, or None when a border ray misses."""
        if not len(border.lat) or not np.isfinite(border.lat).all():
            return None

        # The last border pixel repeats the first.
        area, _ = self._geod.polygon_area_perimeter(border.lon[:-1], border.lat[:-1])
        return abs(float(area))

    def _utm_json(self, lat: float, lon: float) -> JSON:
        """The UTM zone, easting and northing of a point."""
        zone = int((lon + 180.0) // 6.0) % 60 + 1
        epsg = (32600 if lat >= 0 else 32700) + zone
        to_utm = self._utm.get(epsg)
        if to_utm is None:
            to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            self._utm[epsg] = to_utm
        easting, northing = to_utm.transform(lon, lat)

        return {
            "zone": f"{zone}{'N' if lat >= 0 else 'S'}",
            "easting": float(easting),
            "northing": float(northing),
        }

    def _backend_name(self, n_rays: int) -> str:
        """The backend that a batch of ``n_rays`` rays uses."""
        backend = self.geolocator.backend
        if isinstance(backend, AutoBackend):
            return f"auto: {backend.select(n_rays).name}"

        return backend.name

    def _settings_json(self) -> JSON:
        geo = self.geolocator
        return {
            "backend": geo.backend.name,
            "max_range": geo.max_range,
            "max_segment_length": geo.max_segment_length,
        }


# ---- Conversions ------------------------------------------------------------

#: The names of the ray states of :func:`_status`.
_STATUS_NAMES = ("miss", "hit", "fallback")


def _status(res: GeoResult) -> IntArray:
    """0 for a miss, 1 for a terrain hit, 2 for a fallback hit."""
    landed = np.isfinite(res.range)
    return np.where(res.hit, 1, np.where(landed, 2, 0)).astype(np.uint8)


def _camera_from(spec: JSON) -> CameraIntrinsics:
    """Camera intrinsics from their JSON form."""
    return CameraIntrinsics(
        width=int(spec["width"]),
        height=int(spec["height"]),
        fx=float(spec["fx"]),
        fy=float(spec["fy"]),
        cx=float(spec["cx"]),
        cy=float(spec["cy"]),
        skew=float(spec.get("skew", 0.0)),
        dist=tuple(float(x) for x in spec.get("dist") or ()),
    )


def _camera_json(camera: CameraIntrinsics) -> JSON:
    """The JSON form of camera intrinsics. The distortion always has 8 terms."""
    dist = list(camera.dist) + [0.0] * (8 - len(camera.dist))
    return {
        "width": camera.width,
        "height": camera.height,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "skew": camera.skew,
        "dist": dist,
    }


def _pose_from(spec: JSON) -> CameraPose:
    """A camera pose from its JSON form: a position and yaw, pitch and roll."""
    return CameraPose.from_euler(
        float(spec["lat"]),
        float(spec["lon"]),
        float(spec["alt"]),
        float(spec["yaw"]),
        float(spec["pitch"]),
        float(spec["roll"]),
        alt_ref=spec.get("alt_ref", "ellipsoid"),
    )


def _pose_json(pose: CameraPose) -> JSON:
    """The JSON form of a camera pose.

    A nadir or zenith camera has no unique yaw and roll. SciPy then warns
    and sets the roll to zero, which is a valid answer.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yaw, pitch, roll = (float(x) for x in pose.euler())

    return {
        "lat": pose.lat,
        "lon": pose.lon,
        "alt": pose.alt,
        "alt_ref": pose.alt_ref,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


def _pixels(values: Any) -> FloatArray:
    """Pixels ``(n, 2)`` from a JSON list. None gives no pixels."""
    if values is None:
        return np.empty((0, 2))

    px = np.asarray(values, dtype=float)
    if px.size == 0:
        return np.empty((0, 2))
    if px.ndim != 2 or px.shape[1] != 2:
        raise ValueError("pixels must be a list of [u, v] pairs")

    return px


def _b64(values: ArrayLike, dtype: str = "<f4") -> str:
    """The little-endian bytes of an array as base64 text."""
    data = np.ascontiguousarray(values, dtype=dtype).tobytes()
    return base64.b64encode(data).decode("ascii")


def _nums(values: ArrayLike, digits: int | None = None) -> Any:
    """An array as nested lists for JSON, with None for NaN."""
    a = np.asarray(values, dtype=float)
    if digits is not None:
        a = np.round(a, digits)
    out = a.astype(object)
    out[~np.isfinite(a)] = None

    return out.tolist()


def _num(value: Any, digits: int | None = None) -> float | None:
    """A number for JSON: None for NaN, else rounded to ``digits``."""
    x = float(value)
    if not math.isfinite(x):
        return None

    return round(x, digits) if digits is not None else x


def _spread(values: FloatArray) -> JSON:
    """The range of the finite values, and their 2nd and 98th percentiles."""
    ok = values[np.isfinite(values)]
    if not ok.size:
        return {"min": None, "max": None, "low": None, "high": None}

    low, high = np.percentile(ok, [2.0, 98.0])
    return {"min": float(ok.min()), "max": float(ok.max()), "low": float(low), "high": float(high)}


# ---- Geometry ---------------------------------------------------------------


def _local_axes(lat: ArrayLike, lon: ArrayLike) -> FloatArray:
    """The east, north and up unit vectors in ECEF, as rows, shape ``(..., 3, 3)``."""
    phi = np.radians(lat)
    lam = np.radians(lon)
    sp, cp = np.sin(phi), np.cos(phi)
    sl, cl = np.sin(lam), np.cos(lam)

    east = np.stack([-sl, cl, np.zeros_like(sl)], axis=-1)
    north = np.stack([-sp * cl, -sp * sl, cp], axis=-1)
    up = np.stack([cp * cl, cp * sl, sp], axis=-1)

    return np.stack([east, north, up], axis=-2)


def _mean_radius(lat: float) -> float:
    """The Gaussian mean radius of curvature of WGS84 at a latitude, in meters."""
    a = 6378137.0
    e2 = 6.69437999014e-3
    s2 = math.sin(math.radians(lat)) ** 2

    return a * math.sqrt(1.0 - e2) / (1.0 - e2 * s2)


def _post_spacing(terrain: Terrain) -> float:
    """The approximate distance between neighbor terrain posts, in meters."""
    w, s, e, n = terrain.bounds_wgs84()
    rows, cols = terrain.shape
    k = math.radians(1.0) * _EARTH_RADIUS
    dx = (e - w) * k * math.cos(math.radians(0.5 * (s + n))) / max(cols - 1, 1)
    dy = (n - s) * k / max(rows - 1, 1)

    return max(0.01, min(dx, dy))


def _grid_gsd(ecef: FloatArray, step: int) -> FloatArray:
    """The ground sample distance of a pixel grid: the geometric mean of the
    hit spacing along the rows and along the columns, for each pixel."""
    rows, cols = ecef.shape[:2]
    if rows < 2 or cols < 2:
        return np.full((rows, cols), np.nan)

    du = np.linalg.norm(np.diff(ecef, axis=1), axis=-1) / step
    dv = np.linalg.norm(np.diff(ecef, axis=0), axis=-1) / step
    du = np.concatenate([du, du[:, -1:]], axis=1)
    dv = np.concatenate([dv, dv[-1:]], axis=0)

    return np.sqrt(du * dv)


def _near_miss(t: FloatArray, clearance: FloatArray, t_hit: float) -> JSON | None:
    """The smallest clearance of a ray at a low point before its final approach.

    The final approach is the part where the clearance falls to zero at the
    hit. A low point is a local minimum of the clearance, so the start at
    the camera does not count. None when the ray has no such low point.
    """
    end = len(t) if not math.isfinite(t_hit) else int(np.searchsorted(t, t_hit))
    c = np.where(np.isfinite(clearance[:end]), clearance[:end], np.inf)
    if end < 3:
        return None

    # Walk back from the hit while the clearance grows. The rest is before
    # the final approach.
    k = end - 1
    if math.isfinite(t_hit):
        while k > 0 and c[k - 1] >= c[k]:
            k -= 1

    # The interior local minima before the final approach.
    if k < 2:
        return None
    mid = c[1:k]
    low = np.flatnonzero((mid <= c[: k - 1]) & (mid < c[2 : k + 1]) & np.isfinite(mid)) + 1
    if not low.size:
        return None

    i = int(low[np.argmin(c[low])])
    return {"t": float(t[i]), "clearance": float(c[i])}
