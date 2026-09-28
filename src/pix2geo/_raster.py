"""rasterio helpers used by the terrain and geoid loaders."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ._typing import Bounds, CRSLike, RasterSource

if TYPE_CHECKING:
    from rasterio.io import DatasetReader
    from rasterio.windows import Window


@contextmanager
def opened(source: RasterSource) -> Generator[DatasetReader, None, None]:
    """Provides ``source`` (a path or dataset) as an open dataset.

    A path opens here and closes after use. An open dataset belongs to the
    caller, so it stays open.
    """
    if not isinstance(source, (str, os.PathLike)):
        yield source
        return

    import rasterio

    with rasterio.open(source) as ds:
        yield ds


def wgs84_bounds_to_crs(bounds: Sequence[float], crs: CRSLike, margin: float = 0.0) -> Bounds:
    """Transform WGS84 ``(west, south, east, north)`` plus ``margin`` degrees into ``crs``."""
    from rasterio.warp import transform_bounds

    w, s, e, n = bounds

    return transform_bounds(
        "EPSG:4326",
        crs,
        w - margin,
        s - margin,
        e + margin,
        n + margin,
        densify_pts=21,
    )


def padded_window(ds: DatasetReader, bounds: Sequence[float], pad: int) -> Window:
    """The window of ``ds`` that covers ``bounds``, grown by ``pad`` pixels.

    ``bounds`` is ``(left, bottom, right, top)`` in the CRS of the dataset. The
    window is clipped to the raster, so it can be empty.
    """
    from rasterio.windows import Window, from_bounds

    win = from_bounds(*bounds, transform=ds.transform)

    # Round the fractional window out to whole pixels, grow it, and clip it to the raster.
    c0 = max(math.floor(win.col_off) - pad, 0)
    r0 = max(math.floor(win.row_off) - pad, 0)
    c1 = min(math.ceil(win.col_off + win.width) + pad, ds.width)
    r1 = min(math.ceil(win.row_off + win.height) + pad, ds.height)

    return Window(c0, r0, max(c1 - c0, 0), max(r1 - r0, 0))


def read_band(
    ds: DatasetReader,
    band: int = 1,
    window: Window | None = None,
    dtype: DTypeLike = np.float32,
) -> NDArray[np.floating]:
    """Read one band as floats, with NaN for nodata and the band scale and offset applied."""
    data = ds.read(band, window=window, out_dtype=dtype)
    data[ds.read_masks(band, window=window) == 0] = np.nan

    scale = ds.scales[band - 1] if ds.scales else None
    offset = ds.offsets[band - 1] if ds.offsets else None
    if scale not in (None, 1.0) or offset not in (None, 0.0):
        data = (data * (scale or 1.0) + (offset or 0.0)).astype(dtype, copy=False)

    return data


def warn_if_msl_without_geoid(datasets: Iterable[DatasetReader], geoid: object) -> None:
    """Warn when a dataset says that its heights are MSL but no geoid is given."""
    if geoid is not None:
        return

    for ds in datasets:
        if ds.tags().get("DTED_VerticalDatum", "").strip().upper() == "MSL":
            warnings.warn(
                f"{ds.name} has MSL heights, but no geoid was given. The terrain will be "
                "too low or too high by the geoid undulation (up to about 100 m). Pass "
                "geoid='egm96' (or a grid file), or geoid=0 to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
            return
