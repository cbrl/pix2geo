"""Small grid interpolation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ._typing import ArrayLike, FloatArray, IntArray


def bilinear_sample(grid: NDArray[np.floating], x: ArrayLike, y: ArrayLike) -> FloatArray:
    """Bilinear interpolation of ``grid`` at pixel-center coordinates.

    ``x`` is the column and ``y`` is the row, with integer values at pixel
    centers. Points outside ``[0, cols-1] x [0, rows-1]`` give NaN. NaN
    values in ``grid`` propagate into the result.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    rows, cols = grid.shape
    out = np.full(np.broadcast(x, y).shape, np.nan)

    x, y = np.broadcast_arrays(x, y)

    ok = (x >= 0) & (x <= cols - 1) & (y >= 0) & (y <= rows - 1)
    if not ok.any():
        return out

    xs = x[ok]
    ys = y[ok]

    # The cell (i, j) that holds each point. A point on the last row or column
    # goes into the last cell, so the clamp keeps i + 1 and j + 1 in the grid.
    j = np.minimum(np.floor(xs).astype(np.int64), max(cols - 2, 0))
    i = np.minimum(np.floor(ys).astype(np.int64), max(rows - 2, 0))

    # The position of the point in its cell, from 0 to 1.
    u = xs - j
    v = ys - i

    # The far corner of the cell. The clamp is for a grid with one row or column.
    j1 = np.minimum(j + 1, cols - 1)
    i1 = np.minimum(i + 1, rows - 1)

    # Interpolate along x on the top and bottom edges, then along y.
    top = grid[i, j] * (1 - u) + grid[i, j1] * u
    bottom = grid[i1, j] * (1 - u) + grid[i1, j1] * u
    out[ok] = top * (1 - v) + bottom * v

    return out


def upsample_bilinear(
    coarse: ArrayLike,
    rows: ArrayLike,
    cols: ArrayLike,
    shape: tuple[int, int],
    dtype: DTypeLike = np.float32,
) -> NDArray[np.floating]:
    """Expand values on a sparse grid to a full grid with separable linear interpolation.

    ``coarse[a, b]`` is the value at full-grid position ``(rows[a], cols[b])``.
    ``rows`` and ``cols`` must be increasing and must cover the full grid.
    """
    full_r = np.arange(shape[0], dtype=float)
    full_c = np.arange(shape[1], dtype=float)

    def weights(src: FloatArray, dst: FloatArray) -> tuple[IntArray, IntArray, FloatArray]:
        # For each dst position: the two src neighbours and the weight of the second.
        if len(src) == 1:
            # One sample: every position takes it with the full weight.
            z = np.zeros(len(dst), dtype=np.int64)
            return z, z, np.zeros(len(dst))

        # The src interval [k, k + 1] that holds each dst position.
        k = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, len(src) - 2)
        w = (dst - src[k]) / (src[k + 1] - src[k])
        return k, k + 1, w

    # Interpolate along the columns first, then along the rows.
    values = np.asarray(coarse, dtype=dtype)
    c0, c1, wc = weights(np.asarray(cols, float), full_c)
    tmp = values[:, c0] * (1 - wc.astype(dtype)) + values[:, c1] * wc.astype(dtype)

    r0, r1, wr = weights(np.asarray(rows, float), full_r)
    wr_col = wr.astype(dtype)[:, None]

    return tmp[r0, :] * (1 - wr_col) + tmp[r1, :] * wr_col
