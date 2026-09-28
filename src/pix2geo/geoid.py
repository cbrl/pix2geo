"""Geoid models: the offset between orthometric (MSL) and ellipsoidal heights.

The undulation ``N`` relates the two height systems:
``h_ellipsoid = H_orthometric + N``.

Most global elevation products (DTED, SRTM, Copernicus GLO-30) give
orthometric heights above the EGM96 or EGM2008 geoid. The library works
with WGS84 ellipsoidal heights, so a :class:`~pix2geo.terrain.Terrain` made
from such a product needs a geoid model.
"""

from __future__ import annotations

import os
from typing import Protocol, Union, runtime_checkable

import numpy as np

from . import _raster
from ._interp import bilinear_sample
from ._typing import ArrayLike, Bounds, FloatArray, PathLike

__all__ = [
    "ConstantGeoid",
    "GeoidModel",
    "GeoidSpec",
    "PyprojGeoid",
    "RasterGeoid",
    "SupportsUndulation",
    "resolve_geoid",
]


_NAMED_GEOIDS = {"egm96": "EPSG:5773", "egm2008": "EPSG:3855", "egm08": "EPSG:3855"}


@runtime_checkable
class SupportsUndulation(Protocol):
    """Any object with an ``undulation(lat, lon)`` method can be a geoid model."""

    def undulation(self, lat: ArrayLike, lon: ArrayLike) -> FloatArray: ...


#: The geoid values that :func:`resolve_geoid` accepts.
GeoidSpec = Union[None, float, PathLike, SupportsUndulation]


class GeoidModel:
    """Base class. Subclasses give the undulation ``N`` in meters."""

    def undulation(self, lat: ArrayLike, lon: ArrayLike) -> FloatArray:  # pragma: no cover
        raise NotImplementedError

    def to_ellipsoidal(
        self, lat: ArrayLike, lon: ArrayLike, h_orthometric: ArrayLike
    ) -> FloatArray:
        """Orthometric height to WGS84 ellipsoidal height."""
        return np.asarray(h_orthometric, float) + self.undulation(lat, lon)

    def to_orthometric(
        self, lat: ArrayLike, lon: ArrayLike, h_ellipsoidal: ArrayLike
    ) -> FloatArray:
        """WGS84 ellipsoidal height to orthometric height."""
        return np.asarray(h_ellipsoidal, float) - self.undulation(lat, lon)


class ConstantGeoid(GeoidModel):
    """The same undulation everywhere. Useful for small areas and tests."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def undulation(self, lat: ArrayLike, lon: ArrayLike) -> FloatArray:
        shape = np.broadcast(np.asarray(lat), np.asarray(lon)).shape
        return np.full(shape, self.value)

    def __repr__(self) -> str:
        return f"ConstantGeoid({self.value})"


class RasterGeoid(GeoidModel):
    """Undulations from a geographic raster, for example a PROJ geoid grid.

    PROJ distributes the grids at https://cdn.proj.org, for example
    ``us_nga_egm96_15.tif`` (EGM96) and ``us_nga_egm08_25.tif`` (EGM2008).

    Parameters
    ----------
    path:
        Any raster that rasterio can read, in a geographic CRS.
    bounds:
        Optional ``(west, south, east, north)`` in degrees. Only this area
        is read. This saves memory for high-resolution global grids.
    """

    def __init__(self, path: PathLike, bounds: Bounds | None = None) -> None:
        import rasterio

        with rasterio.open(path) as ds:
            if ds.crs is not None and not ds.crs.is_geographic:
                raise ValueError("a geoid raster must use a geographic CRS")

            window = None if bounds is None else _raster.padded_window(ds, bounds, pad=2)
            self.grid = _raster.read_band(ds, 1, window, dtype=np.float64)
            self.transform = ds.window_transform(window) if window is not None else ds.transform
            full_width_deg = abs(ds.transform.a) * ds.width

        self.path = os.fspath(path)
        self._inv = ~self.transform

        # A grid that spans the full globe gets longitude wrap-around. A copy of
        # the first column after the last column closes the gap at the seam.
        self._global = bounds is None and full_width_deg >= 359.9
        if self._global:
            self.grid = np.concatenate([self.grid, self.grid[:, :1]], axis=1)
            self._period = 360.0 / abs(self.transform.a)

    def undulation(self, lat: ArrayLike, lon: ArrayLike) -> FloatArray:
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)

        # (lon, lat) to grid coordinates with the origin at the pixel center.
        inv = self._inv
        col = inv.a * lon + inv.b * lat + inv.c - 0.5
        row = inv.d * lon + inv.e * lat + inv.f - 0.5

        # A global grid repeats every 360 degrees, which is _period columns.
        if self._global:
            col = np.mod(col, self._period)

        return bilinear_sample(self.grid, col, row)

    def __repr__(self) -> str:
        return f"RasterGeoid({self.path!r})"


class PyprojGeoid(GeoidModel):
    """Undulations from PROJ through pyproj.

    Parameters
    ----------
    vertical_crs:
        A vertical CRS, for example ``"EPSG:5773"`` (EGM96 height) or
        ``"EPSG:3855"`` (EGM2008 height).
    network:
        Let PROJ download the geoid grid from the PROJ CDN.

    PROJ needs the geoid grid. Without it, PROJ falls back to a "ballpark"
    transformation with ``N = 0``. This class refuses that fallback. To get
    the grid, set ``network=True``, run ``projsync --file <grid>``, or use
    :class:`RasterGeoid` with a downloaded grid file.
    """

    def __init__(self, vertical_crs: str = "EPSG:5773", network: bool = False) -> None:
        import pyproj
        from pyproj import CRS, Transformer

        if network:
            pyproj.network.set_network_enabled(True)

        vcrs = CRS.from_user_input(vertical_crs)
        code = vcrs.to_epsg()
        if code is None:
            raise ValueError(f"{vertical_crs!r} has no EPSG code")

        self.vertical_crs = vcrs
        # WGS84 3-D (ellipsoidal height) to WGS84 2-D + the vertical CRS.
        self._tr = Transformer.from_crs("EPSG:4979", f"EPSG:4326+{code}", always_xy=True)

        if "ballpark" in self._tr.description.lower():
            raise RuntimeError(
                f"PROJ has no geoid grid for {vcrs.name}. Use network=True, run "
                "'projsync' for the grid, or use RasterGeoid with a grid file."
            )

    def undulation(self, lat: ArrayLike, lon: ArrayLike) -> FloatArray:
        lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        # The orthometric height of a point on the ellipsoid is -N.
        _, _, H = self._tr.transform(lon, lat, np.zeros_like(lat))
        H = np.asarray(H, float)
        H[~np.isfinite(H)] = np.nan
        return -H

    def __repr__(self) -> str:
        return f"PyprojGeoid({self.vertical_crs.name!r})"


def resolve_geoid(spec: GeoidSpec) -> SupportsUndulation | None:
    """Turn a user geoid specification into a geoid model.

    * ``None``: no geoid (heights are already ellipsoidal).
    * a number: :class:`ConstantGeoid`.
    * ``"egm96"``, ``"egm2008"`` or ``"EPSG:<code>"``: :class:`PyprojGeoid`.
    * a file path: :class:`RasterGeoid`.
    * a :class:`GeoidModel` (or any object with ``undulation``): unchanged.
    """
    if spec is None:
        return None
    if isinstance(spec, SupportsUndulation):
        return spec
    if isinstance(spec, (int, float, np.floating, np.integer)):
        return ConstantGeoid(float(spec))
    if isinstance(spec, (str, os.PathLike)):
        s = os.fspath(spec)
        if s.lower() in _NAMED_GEOIDS:
            return PyprojGeoid(_NAMED_GEOIDS[s.lower()])
        if s.upper().startswith("EPSG:"):
            return PyprojGeoid(s)
        return RasterGeoid(s)
    raise TypeError(f"cannot use {spec!r} as a geoid model")
