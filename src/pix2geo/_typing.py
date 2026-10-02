"""Type aliases for the values that pass between the modules."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from rasterio.io import DatasetReader

__all__ = [
    "ArrayLike",
    "BoolArray",
    "Bounds",
    "CRSLike",
    "FloatArray",
    "GridLut",
    "GridParams",
    "HeightGrid",
    "IntArray",
    "LevelOffsets",
    "LevelShapes",
    "NodeStack",
    "PathLike",
    "PyramidData",
    "RasterSource",
    "RayLimits",
]

#: A file path.
PathLike = Union[str, os.PathLike[str]]

#: A raster file path, or a rasterio dataset that is open for reading.
RasterSource = Union[str, os.PathLike[str], "DatasetReader"]

#: ``(west, south, east, north)``, or ``(left, bottom, right, top)`` in a CRS.
Bounds = tuple[float, float, float, float]

#: Anything that ``pyproj.CRS.from_user_input`` accepts: an EPSG string or
#: code, WKT, PROJ JSON, or a pyproj or rasterio CRS object.
CRSLike = Any

#: A float64 array.
FloatArray = NDArray[np.float64]
#: An int64 array.
IntArray = NDArray[np.int64]
#: A boolean array.
BoolArray = NDArray[np.bool_]

# ---- Kernel arrays ----------------------------------------------------------

#: Elevation posts, shape ``(rows, cols)``, in WGS84 ellipsoidal meters.
#: NaN marks nodata.
HeightGrid = NDArray[np.float32]

#: All the max quadtree levels in one flat array. See
#: :class:`~pix2geo.quadtree.HeightPyramid`.
PyramidData = NDArray[np.float32]

#: Start index of each quadtree level in :data:`PyramidData`, shape ``(levels,)``.
LevelOffsets = NDArray[np.int64]

#: ``(rows, cols)`` of each quadtree level, shape ``(levels, 2)``. Level 0 is
#: the cell level. The CuPy backend uploads it as int32.
LevelShapes = NDArray[np.int64]

#: The packed (lon, lat) to grid mapping, shape ``(N_PARAMS,)``. See
#: :func:`~pix2geo.backends._kernel.make_grid_params`.
GridParams = NDArray[np.float64]

#: The ray clip settings of one batch, shape ``(N_LIMITS,)``. See
#: :func:`~pix2geo.backends._kernel.make_ray_limits`.
RayLimits = NDArray[np.float64]

#: Grid coordinates (x or y) on a regular lon/lat lattice, for projected CRSs.
#: A ``(2, 2)`` dummy for geographic rasters.
GridLut = NDArray[np.float64]

#: The depth-first traversal stack, shape ``(size, 3)``. Each row is one
#: quadtree node: ``(level, row, column)``. int64 on the CPU. int32 on the GPU,
#: where it halves the memory traffic.
NodeStack = Union[NDArray[np.int64], NDArray[np.int32]]
