"""Numba backend. Compiles the shared kernel to parallel machine code.

Importing this module raises ``ImportError`` when Numba is not installed.
The compiled code is cached on disk, so only the first run pays the
compile time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, cast

import numba

from .._typing import FloatArray, RayLimits
from ._kernel import Decorator, build_kernels, run_cpu_kernel

if TYPE_CHECKING:
    from ..terrain import Terrain


kernels = build_kernels(
    cast(Decorator, numba.njit(cache=True, nogil=True)),
    cast(Decorator, numba.njit(cache=True, nogil=True, parallel=True)),
    cast(Callable[[int], Iterable[int]], numba.prange),
)

# Rays near the horizon cost far more than steep rays. Small chunks balance
# this load between the threads.
CHUNK_SIZE = 64


def trace_rays(
    terrain: Terrain,
    orig: FloatArray,
    dirs: FloatArray,
    limits: RayLimits,
) -> FloatArray:
    """Run the kernel. See :meth:`pix2geo.backends.Backend.trace_rays`."""
    with numba.parallel_chunksize(CHUNK_SIZE):
        return run_cpu_kernel(kernels.trace_rays, terrain, orig, dirs, limits)
