"""The max quadtree that lets the ray caster skip empty space above the terrain."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._typing import LevelOffsets, LevelShapes, PyramidData

__all__ = ["HeightPyramid"]


class HeightPyramid:
    """A max quadtree (maximum mipmap) over the terrain cells.

    Each cell in level 0 holds the maximum of its four corner posts. Each
    higher level holds the maximum of a 2x2 block of the level below. The top
    level is one node that covers the whole raster. Holes have a value of
    ``-inf``, so the ray caster never descends into them.

    The levels are stored in one flat array for the compute kernels.
    Node ``(i, j)`` of level ``k`` is located at
    ``data[offsets[k] + i * shapes[k, 1] + j]``.
    :attr:`levels` holds 2-D views into that array.
    """

    def __init__(self, heights: ArrayLike) -> None:
        h = np.asarray(heights, dtype=np.float32)
        if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 2:
            raise ValueError("the elevation grid must be 2-D with at least 2x2 posts")

        # Each level halves both axes of the level below, until one node remains.
        levels = [_cell_maximum(h)]
        while levels[-1].shape != (1, 1):
            levels.append(_reduce_2x2(levels[-1]))

        # Store the levels one after the other. offsets[k] is the start of level k.
        self.shapes: LevelShapes = np.array([lv.shape for lv in levels], dtype=np.int64)
        sizes = self.shapes[:, 0] * self.shapes[:, 1]
        self.offsets: LevelOffsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self.data: PyramidData = np.concatenate([lv.ravel() for lv in levels])

        # The GPU backends copy these arrays once, so they must not change. The
        # level views below get the read-only flag from the data.
        for array in (self.shapes, self.offsets, self.data):
            array.flags.writeable = False

        self.levels: list[NDArray[np.float32]] = [
            self.data[o : o + n].reshape(shape)
            for o, n, shape in zip(self.offsets, sizes, self.shapes)
        ]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def nbytes(self) -> int:
        return self.data.nbytes


def _cell_maximum(h: NDArray[np.float32]) -> NDArray[np.float32]:
    """The maximum of the four corner posts of each cell. A NaN corner gives ``-inf``."""
    # The four slices are the top-left, top-right, bottom-left and bottom-right
    # corners of every cell. np.maximum propagates NaN.
    cell = np.maximum(np.maximum(h[:-1, :-1], h[:-1, 1:]), np.maximum(h[1:, :-1], h[1:, 1:]))
    cell[np.isnan(cell)] = -np.inf

    return cell


def _reduce_2x2(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """The maximum of each 2x2 block. An odd edge is padded with ``-inf``."""
    pad_r, pad_c = a.shape[0] % 2, a.shape[1] % 2
    if pad_r or pad_c:
        a = np.pad(a, ((0, pad_r), (0, pad_c)), constant_values=-np.inf)

    # The reshape puts the rows of each 2x2 block on axis 1 and its columns on axis 3.
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).max(axis=(1, 3))
