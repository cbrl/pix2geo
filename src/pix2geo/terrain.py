"""Digital elevation model loading and grid coordinate conversion.

Surface model
-------------
The terrain surface is the bilinear interpolation of the elevation posts.
The posts sit at the raster pixel centers. Grid coordinates ``(x, y)`` are
fractional (column, row) indices with integer values at pixel centers. A
*cell* ``(i, j)`` is the square between the four posts ``(i, j)``,
``(i, j+1)``, ``(i+1, j)`` and ``(i+1, j+1)``. A raster with ``R x C`` posts
has ``(R-1) x (C-1)`` cells. A cell with a nodata post is a hole.

All heights inside a :class:`Terrain` are WGS84 ellipsoidal heights. The
loader adds the geoid undulation to orthometric (MSL) source data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from typing import Any, Literal, Union

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from . import _raster
from ._geodesy import wrap_lon
from ._interp import bilinear_sample, upsample_bilinear
from ._typing import (
    ArrayLike,
    Bounds,
    CRSLike,
    FloatArray,
    GridLut,
    GridParams,
    HeightGrid,
    RasterSource,
)
from .backends._kernel import make_grid_params
from .geoid import GeoidSpec, SupportsUndulation, resolve_geoid
from .quadtree import HeightPyramid

__all__ = ["FillNodata", "HeightPyramid", "Terrain"]

#: ``None`` (keep holes), a fill height, or ``"interpolate"``.
FillNodata = Union[None, float, Literal["interpolate"]]

#: The affine transform of a raster: an ``Affine`` or its first six coefficients.
TransformLike = Union[Affine, Sequence[float]]

#: ``(params, lut_x, lut_y)``. See :meth:`Terrain.grid_mapping`.
GridMapping = tuple[GridParams, GridLut, GridLut]


class Terrain:
    """An elevation model in WGS84 ellipsoidal heights, ready for ray casting.

    Use :meth:`from_file`, :meth:`from_files` or :meth:`from_array` to make one.

    Parameters
    ----------
    heights:
        2-D elevation posts. NaN, infinity, masked entries (of a NumPy
        masked array) and the ``nodata`` value mark nodata.
    transform:
        The affine transform from (column, row) of the pixel *corner* to CRS
        coordinates, as in rasterio.
    crs:
        The horizontal CRS of the raster. Anything pyproj accepts.
    geoid:
        The geoid model of the source heights. ``None`` means the source
        heights are already WGS84 ellipsoidal heights. See
        :func:`pix2geo.geoid.resolve_geoid` for the accepted values.
    nodata:
        Optional sentinel height that marks nodata, for example ``-9999``.
    fill_nodata:
        ``None`` keeps holes. A number fills holes with that height (in the
        source vertical datum). ``"interpolate"`` fills holes with
        :func:`rasterio.fill.fillnodata`.
    geoid_max_grid:
        The geoid is evaluated on a sparse grid of at most this many points
        on each axis and interpolated between. The geoid is smooth, so this
        loses no accuracy and saves time on large rasters.

    The terrain arrays (:attr:`heights`, the quadtree and the grid mapping)
    are read-only. The quadtree, the height range and the GPU copies come
    from the heights, so a change to the heights would make them wrong.
    Make a new :class:`Terrain` for new heights.
    """

    #: Largest lookup table size on each axis (projected CRSs only).
    lut_max_size = 1024

    #: Target lookup table spacing in raster pixels (projected CRSs only).
    lut_step = 4

    def __init__(
        self,
        heights: ArrayLike,
        transform: TransformLike,
        crs: CRSLike,
        *,
        geoid: GeoidSpec = None,
        nodata: float | None = None,
        fill_nodata: FillNodata = None,
        geoid_max_grid: int = 512,
    ) -> None:
        h = _as_height_grid(heights, nodata)

        if fill_nodata is not None:
            h = _fill_nodata(h, fill_nodata)

        self.transform = transform if isinstance(transform, Affine) else Affine(*transform[:6])
        self.crs = CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
        self.geoid: SupportsUndulation | None = resolve_geoid(geoid)

        self._inv = ~self.transform
        wgs84 = CRS.from_epsg(4326)
        self._is_wgs84 = self.crs.equals(wgs84, ignore_axis_order=True)
        self._to_crs = Transformer.from_crs(wgs84, self.crs, always_xy=True)
        self._to_lonlat = Transformer.from_crs(self.crs, wgs84, always_xy=True)

        # For a geographic raster, longitudes wrap to within 180 degrees of the
        # raster center. A ray that crosses the antimeridian then stays continuous.
        self._lon_center: float | None = None
        if self.crs.is_geographic:
            self._lon_center, _ = self.transform * (h.shape[1] / 2.0, h.shape[0] / 2.0)

        if self.geoid is not None:
            h += self._geoid_grid(self.geoid, h.shape, geoid_max_grid)

        finite = h[np.isfinite(h)]
        if finite.size == 0:
            raise ValueError("the elevation model has no valid data")

        # The derived state below needs heights that do not change.
        h.flags.writeable = False
        self.heights: HeightGrid = h
        self.h_min = float(finite.min())
        self.h_max = float(finite.max())
        self.pyramid = HeightPyramid(h)
        self._grid_mapping: GridMapping | None = None

        # Device copies of the arrays, filled on demand by GPU backends.
        self._device_cache: dict[str, Any] = {}

    # ---- Constructors -------------------------------------------------------

    @classmethod
    def from_array(
        cls,
        heights: ArrayLike,
        transform: TransformLike,
        crs: CRSLike = "EPSG:4326",
        *,
        geoid: GeoidSpec = None,
        nodata: float | None = None,
        fill_nodata: FillNodata = None,
        geoid_max_grid: int = 512,
    ) -> Terrain:
        """Terrain from an in-memory array. See the class parameters."""
        return cls(
            heights,
            transform,
            crs,
            geoid=geoid,
            nodata=nodata,
            fill_nodata=fill_nodata,
            geoid_max_grid=geoid_max_grid,
        )

    @classmethod
    def from_file(
        cls,
        source: RasterSource,
        *,
        band: int = 1,
        bounds: Bounds | None = None,
        margin: float = 0.0,
        geoid: GeoidSpec = None,
        fill_nodata: FillNodata = None,
        geoid_max_grid: int = 512,
    ) -> Terrain:
        """Load a raster that rasterio can read (GeoTIFF, DTED, IMG, VRT, etc).

        Parameters
        ----------
        source:
            The raster file, or a rasterio dataset that is open for reading
            (for example a ``WarpedVRT``). The caller keeps ownership of an
            open dataset, so this method does not close it.
        band:
            The 1-based band index with the elevations.
        bounds:
            Optional ``(west, south, east, north)`` in WGS84 degrees. Only
            this area (plus ``margin`` degrees) is read.
        geoid, fill_nodata, geoid_max_grid:
            See the class parameters.
        """
        with _raster.opened(source) as ds:
            if ds.crs is None:
                raise ValueError(f"{ds.name} has no CRS")
            _raster.warn_if_msl_without_geoid([ds], geoid)

            window = None
            if bounds is not None:
                crs_bounds = _raster.wgs84_bounds_to_crs(bounds, ds.crs, margin)
                window = _raster.padded_window(ds, crs_bounds, pad=1)
                if window.width < 2 or window.height < 2:
                    raise ValueError("the bounds do not overlap the raster")

            heights = _raster.read_band(ds, band, window)
            transform = ds.window_transform(window) if window is not None else ds.transform
            crs = ds.crs

        return cls(
            heights,
            transform,
            crs,
            geoid=geoid,
            fill_nodata=fill_nodata,
            geoid_max_grid=geoid_max_grid,
        )

    @classmethod
    def from_files(
        cls,
        sources: Iterable[RasterSource],
        *,
        band: int = 1,
        bounds: Bounds | None = None,
        margin: float = 0.0,
        geoid: GeoidSpec = None,
        fill_nodata: FillNodata = None,
        geoid_max_grid: int = 512,
    ) -> Terrain:
        """Mosaic several rasters (for example DTED tiles) into one terrain.

        Each source is a path or an open dataset, as for :meth:`from_file`.
        All the rasters must use the same CRS. The other parameters are the
        same as for :meth:`from_file`.
        """
        from rasterio.merge import merge

        sources = list(sources)
        if not sources:
            raise ValueError("no files given")

        # The stack closes the datasets that it opened, also on an error.
        with ExitStack() as stack:
            datasets = [stack.enter_context(_raster.opened(s)) for s in sources]

            crs = datasets[0].crs
            if any(ds.crs != crs for ds in datasets[1:]):
                raise ValueError("all the rasters must use the same CRS")

            _raster.warn_if_msl_without_geoid(datasets, geoid)

            crs_bounds = None
            if bounds is not None:
                crs_bounds = _raster.wgs84_bounds_to_crs(bounds, crs, margin)

            mosaic, transform = merge(
                datasets, bounds=crs_bounds, indexes=[band], dtype="float32", nodata=np.nan
            )

        return cls(
            mosaic[0],
            transform,
            crs,
            geoid=geoid,
            fill_nodata=fill_nodata,
            geoid_max_grid=geoid_max_grid,
        )

    # ---- Coordinate Conversion & Queries ------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape

    def lonlat_to_grid(self, lon: ArrayLike, lat: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """WGS84 longitude/latitude to grid coordinates (pixel-center origin)."""
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)

        # (lon, lat) to raster CRS coordinates (X, Y).
        if self._is_wgs84:
            X, Y = lon, lat
        else:
            X, Y = (np.asarray(c, float) for c in self._to_crs.transform(lon, lat))
        if self._lon_center is not None:
            X = wrap_lon(X, self._lon_center)

        # The inverse transform gives (col, row) with the origin at the pixel
        # corner. The -0.5 moves the origin to the pixel center.
        inv = self._inv
        col = inv.a * X + inv.b * Y + inv.c
        row = inv.d * X + inv.e * Y + inv.f

        return col - 0.5, row - 0.5

    def grid_to_lonlat(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Grid coordinates (pixel-center origin) to WGS84 longitude/latitude."""
        # The +0.5 moves the origin to the pixel corner, as the transform needs.
        t = self.transform
        c = np.asarray(x, float) + 0.5
        r = np.asarray(y, float) + 0.5
        X = t.a * c + t.b * r + t.c
        Y = t.d * c + t.e * r + t.f

        if self._is_wgs84:
            return X, Y

        lon, lat = self._to_lonlat.transform(X, Y)
        return np.asarray(lon, float), np.asarray(lat, float)

    def height_at(self, lat: ArrayLike, lon: ArrayLike) -> FloatArray:
        """Bilinear terrain height (WGS84 ellipsoidal) at lat/lon. NaN outside the data."""
        x, y = self.lonlat_to_grid(lon, lat)
        return bilinear_sample(self.heights, x, y)

    def orthometric_height_at(self, lat: ArrayLike, lon: ArrayLike) -> FloatArray:
        """Terrain height in the source vertical datum (MSL when a geoid is set)."""
        h = self.height_at(lat, lon)
        if self.geoid is None:
            return h

        return h - self.geoid.undulation(lat, lon)

    def bounds_wgs84(self) -> Bounds:
        """``(west, south, east, north)`` of the data area in WGS84 degrees."""
        lon, lat = self.grid_to_lonlat(*_outline(*self.shape, points_per_edge=33))
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())

    def grid_mapping(self) -> GridMapping:
        """The (lon, lat) to grid mapping for the compute kernels.

        Returns ``(params, lut_x, lut_y)``. See :mod:`pix2geo.backends._kernel`.
        A WGS84 geographic raster maps with an exact affine transform. Other
        CRSs map through lookup tables that pyproj fills on a regular
        lon/lat lattice. The kernels interpolate the tables bilinearly. The
        lattice spacing is about ``lut_step`` raster pixels, so the
        interpolation error is far below one pixel.
        """
        if self._grid_mapping is None:
            if self._is_wgs84:
                mapping = self._affine_grid_mapping()
            else:
                mapping = self._lut_grid_mapping()

            # The GPU backends copy these arrays once, so they must not change.
            for array in mapping:
                array.flags.writeable = False
            self._grid_mapping = mapping

        return self._grid_mapping

    def __repr__(self) -> str:
        return (
            f"Terrain(shape={self.shape}, crs={self.crs.to_string()!r}, "
            f"h=[{self.h_min:.1f}, {self.h_max:.1f}] m, geoid={self.geoid!r})"
        )

    # ---- Internals ----------------------------------------------------------

    def _affine_grid_mapping(self) -> GridMapping:
        # The inverse transform, with the -0.5 pixel-center shift in its offsets.
        # The kernels do not read the lookup tables in this mode.
        inv = self._inv
        affine = (inv.a, inv.b, inv.c - 0.5, inv.d, inv.e, inv.f - 0.5)
        unused = np.zeros((2, 2))

        return make_grid_params(affine, self._lon_center), unused, unused

    def _lut_grid_mapping(self) -> GridMapping:
        rows, cols = self.shape

        # The lattice covers the raster outline one pixel beyond the outer posts.
        lon, lat = self.grid_to_lonlat(*_outline(rows, cols, points_per_edge=65, pad=1.0))
        center = float(self.grid_to_lonlat(cols / 2.0, rows / 2.0)[0])
        lon = wrap_lon(lon, center)

        # The lon/lat box around the outline, plus a 1% margin.
        w, e = float(np.nanmin(lon)), float(np.nanmax(lon))
        s, n = float(np.nanmin(lat)), float(np.nanmax(lat))
        pad = 0.01 * max(e - w, n - s, 1e-9)

        # About one lattice point every lut_step raster pixels. Row 0 is north,
        # as in the raster.
        nlon = int(np.clip(math.ceil(cols / self.lut_step) + 1, 2, self.lut_max_size))
        nlat = int(np.clip(math.ceil(rows / self.lut_step) + 1, 2, self.lut_max_size))
        lons = np.linspace(w - pad, e + pad, nlon)
        lats = np.linspace(n + pad, s - pad, nlat)

        # The exact grid coordinates at each lattice point. pyproj gives inf
        # where a projection fails. Store those as NaN.
        gx, gy = self.lonlat_to_grid(*np.meshgrid(lons, lats))
        gx = np.where(np.isfinite(gx), gx, np.nan)
        gy = np.where(np.isfinite(gy), gy, np.nan)

        # (lon, lat) to fractional lattice indices.
        dlon = lons[1] - lons[0]
        dlat = lats[1] - lats[0]
        affine = (1.0 / dlon, 0.0, -lons[0] / dlon, 0.0, 1.0 / dlat, -lats[0] / dlat)
        params = make_grid_params(affine, center, use_lut=True)

        return params, np.ascontiguousarray(gx), np.ascontiguousarray(gy)

    def _geoid_grid(
        self, geoid: SupportsUndulation, shape: tuple[int, int], max_grid: int
    ) -> NDArray[np.float32]:
        """Geoid undulation at every post, from a sparse grid of geoid samples."""
        rows, cols = shape

        # Sample every step-th post, and always the last row and column, so that
        # the samples cover the full grid.
        step = max(1, math.ceil(max(rows, cols) / max(int(max_grid), 2)))
        rr = np.unique(np.r_[np.arange(0, rows, step), rows - 1])
        cc = np.unique(np.r_[np.arange(0, cols, step), cols - 1])
        gc, gr = np.meshgrid(cc.astype(float), rr.astype(float))

        # Evaluate the geoid at the samples, then interpolate to every post.
        lon, lat = self.grid_to_lonlat(gc, gr)
        N = np.asarray(geoid.undulation(lat, lon), float)
        if not np.all(np.isfinite(N)):
            raise ValueError("the geoid model does not cover the elevation model")

        if step == 1:
            return N.astype(np.float32)

        return upsample_bilinear(N, rr, cc, shape)


def _as_height_grid(heights: ArrayLike, nodata: float | None) -> HeightGrid:
    """A float32 copy of ``heights`` with NaN at each nodata post.

    The masked entries of a masked array, the non-finite values and the
    ``nodata`` value are nodata.
    """
    # np.array drops the mask of a masked array, so fill the masked entries first.
    h = np.array(np.ma.asarray(heights, dtype=np.float32).filled(np.nan), dtype=np.float32)
    if h.ndim != 2:
        raise ValueError("heights must be a 2-D array")

    # An infinite post would put infinity in the quadtree and stop all pruning.
    holes = ~np.isfinite(h)
    if nodata is not None:
        holes |= h == np.float32(nodata)
    h[holes] = np.nan

    return h


def _fill_nodata(h: HeightGrid, fill: float | str) -> HeightGrid:
    """Fill the NaN posts of ``h`` with a number or by interpolation."""
    interpolate = isinstance(fill, str)
    if interpolate and fill != "interpolate":
        raise ValueError("fill_nodata must be None, a number or 'interpolate'")

    holes = np.isnan(h)
    if not holes.any():
        return h

    if not interpolate:
        h[holes] = float(fill)
        return h

    from rasterio.fill import fillnodata

    # fillnodata reads the holes from the mask (0 = hole), so their values have no effect.
    filled = fillnodata(
        np.nan_to_num(h, nan=0.0),
        mask=(~holes).astype(np.uint8),
        max_search_distance=max(h.shape),
    )

    return filled.astype(np.float32)


def _outline(
    rows: int,
    cols: int,
    points_per_edge: int,
    pad: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Grid coordinates of points on the four edges of the raster.

    The edges pass through the outer posts, or ``pad`` pixels beyond them.
    """
    x0, x1 = -pad, cols - 1 + pad
    y0, y1 = -pad, rows - 1 + pad
    xs = np.linspace(x0, x1, points_per_edge)
    ys = np.linspace(y0, y1, points_per_edge)
    n = points_per_edge

    # The edges in order: top, right, bottom, left.
    x = np.concatenate([xs, np.full(n, x1), xs, np.full(n, x0)])
    y = np.concatenate([np.full(n, y0), ys, np.full(n, y1), ys])

    return x, y
