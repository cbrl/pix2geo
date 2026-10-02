"""CuPy backend. Compiles the shared kernel to CUDA with ``cupyx.jit``.

Importing this module raises ``ImportError`` when CuPy is not installed. The
:func:`check_usable` check also needs a CUDA device and a working compiler.

Differences from the CPU backends
---------------------------------
* The quadtree walk uses float32 and int32, because consumer GPUs run 64-bit
  math at a small fraction of the 32-bit rate. The geodetic math stays
  float64. This makes the kernel about 6x faster.
* ``cupyx.jit`` compiles CuPy ufuncs, not ``math`` functions. The kernel
  therefore gets :data:`CUPY_MATH`.
* ``cupyx.jit`` has no thread-local arrays. Each thread gets its traversal
  stack from one buffer in device memory, and it traces rays in a
  grid-stride loop. The buffer puts the same stack entry of adjacent
  threads side by side, so their memory accesses coalesce.

Accuracy
--------
The float32 walk, FMA operations and the CUDA math functions make the hits
differ slightly from the CPU backends. Most hits are less than 1 mm apart.
A ray that only grazes the terrain can move further, because any small
change (also 1 mm of DEM noise) moves its hit far.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from functools import cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import cupy as cp
import numpy as np
from cupyx import jit

from .._typing import FloatArray, RayLimits
from ._kernel import _F, N_LIMITS, N_PARAMS, Decorator, KernelMath, build_kernels, stack_size

if TYPE_CHECKING:
    from ..terrain import Terrain

#: The kernel math functions as CuPy ufuncs, which ``cupyx.jit`` compiles.
#: ``nan`` is a float64 constant. A Python float would be float32 in "cuda"
#: mode, and a kernel function cannot return float32 and float64 values.
CUPY_MATH = cast(
    KernelMath,
    SimpleNamespace(
        nan=np.float64(math.nan),
        sqrt=cp.sqrt,
        atan2=cp.arctan2,
        degrees=cp.degrees,
        isfinite=cp.isfinite,
    ),
)

#: Threads in one block.
BLOCK = 128


def _identity(func: _F) -> _F:
    return func


@contextmanager
def _quiet_jit() -> Generator[None]:
    """Hide two ``cupyx.jit`` warnings that do not apply to this backend.

    Each ``rawkernel()`` call warns that ``cupyx.jit`` is experimental. The
    compiler warns about each decorator whose source text does not contain
    "rawkernel", such as the ``@jit`` of the shared kernel.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "cupyx.jit.rawkernel is experimental", FutureWarning)
        warnings.filterwarnings("ignore", "Decorator .* may not supported in JIT", RuntimeWarning)
        yield


# The "cuda" mode follows the C++ type rules. A float literal then has the type
# of the other operand, so the float32 walk stays float32 and the float64 math
# stays float64. The "numpy" mode would make each literal float64. A variable
# keeps the type of its first value: a later value of another type is an error.
with _quiet_jit():
    _device_function = cast(Decorator, jit.rawkernel(device=True, mode="cuda"))
    _global_function = jit.rawkernel(mode="cuda")

# This backend does not use the trace_rays of the shared kernel, because it has
# a prange loop. The kernel below replaces it.
kernels = build_kernels(_device_function, _identity, range, CUPY_MATH, cp.float32, cp.int32)
trace_ray = kernels.trace_ray


@_global_function
def trace_rays_kernel(
    hgt: cp.ndarray,
    pyr: cp.ndarray,
    offs: cp.ndarray,
    shp: cp.ndarray,
    params: cp.ndarray,
    lut_x: cp.ndarray,
    lut_y: cp.ndarray,
    limits: cp.ndarray,
    orig: cp.ndarray,
    dirs: cp.ndarray,
    stacks: cp.ndarray,
    out: cp.ndarray,
) -> None:
    # The arguments are those of the shared trace_rays, plus one stack for each
    # thread. Thread tid traces the rays tid, tid + T, tid + 2 T, and so on,
    # where T is the number of threads.
    tid = jit.grid(1)
    stack = stacks[tid]

    for r in range(tid, dirs.shape[0], jit.gridsize(1)):
        trace_ray(hgt, pyr, offs, shp, params, lut_x, lut_y, limits, orig, dirs, r, stack, out)


def _stacks(n_threads: int, num_levels: int) -> cp.ndarray:
    """One int32 traversal stack for each thread, shape ``(n_threads, size, 3)``.

    The memory order is ``(size, 3, n_threads)``. Thus entry ``[s, c]`` of
    adjacent threads is at adjacent addresses.
    """
    size = stack_size(num_levels)

    return cp.empty((size, 3, n_threads), dtype=cp.int32).transpose(2, 0, 1)


@cache
def _resident_threads(device_id: int) -> int:
    """The largest number of threads that device ``device_id`` runs at one time."""
    attributes = cp.cuda.Device(device_id).attributes

    return int(attributes["MultiProcessorCount"] * attributes["MaxThreadsPerMultiProcessor"])


def _launch(
    dev: dict[str, cp.ndarray],
    num_levels: int,
    orig: FloatArray,
    dirs: FloatArray,
    limits: RayLimits,
) -> cp.ndarray:
    """Run the kernel on the device terrain arrays ``dev`` and return the ``(4, n)`` hits."""
    n = len(dirs)

    # One thread for each ray, up to the resident threads of the device. More
    # threads would only wait, and each thread needs a stack.
    resident = _resident_threads(cp.cuda.Device().id)
    blocks = max(1, min(-(-n // BLOCK), resident // BLOCK))

    # The first call with new argument types compiles the kernel.
    out = cp.empty((4, n), dtype=cp.float64)
    with _quiet_jit():
        trace_rays_kernel(
            (blocks,),
            (BLOCK,),
            (
                dev["hgt"], dev["pyr"], dev["offs"], dev["shp"], dev["params"],
                dev["lut_x"], dev["lut_y"], cp.asarray(limits),
                cp.asarray(orig), cp.asarray(dirs),
                _stacks(blocks * BLOCK, num_levels), out,
            ),
        )  # fmt: skip

    return out


def check_usable() -> None:
    """Raise an exception when no CUDA device is present or the kernel does not compile.

    A small dummy raycast compiles the kernel here so a missing CUDA compiler
    shows eagerly when the backend is selected, instead of the first ray batch.
    The dummy arrays have the same types as real ones, so the real batches also
    use the compiled kernel from the cache.
    """
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device is present")

    dev = {
        "hgt": cp.zeros((2, 2), dtype=cp.float32),
        "pyr": cp.zeros(1, dtype=cp.float32),
        "offs": cp.zeros(1, dtype=cp.int64),
        "shp": cp.ones((1, 2), dtype=cp.int32),
        "params": cp.zeros(N_PARAMS),
        "lut_x": cp.zeros((2, 2)),
        "lut_y": cp.zeros((2, 2)),
    }
    one_ray = np.zeros((1, 3))
    _launch(dev, 1, one_ray, one_ray, np.zeros(N_LIMITS))
    cp.cuda.Device().synchronize()


def _device_terrain(terrain: Terrain) -> dict[str, cp.ndarray]:
    """The terrain arrays on the GPU. They upload once for each terrain."""
    cached = terrain._device_cache.get("cupy")
    if cached is None:
        pyr = terrain.pyramid
        params, lut_x, lut_y = terrain.grid_mapping()
        cached = {
            "hgt": cp.asarray(terrain.heights),
            "pyr": cp.asarray(pyr.data),
            "offs": cp.asarray(pyr.offsets),
            "shp": cp.asarray(pyr.shapes, dtype=cp.int32),
            "params": cp.asarray(params),
            "lut_x": cp.asarray(lut_x),
            "lut_y": cp.asarray(lut_y),
        }
        terrain._device_cache["cupy"] = cached

    return cached


def trace_rays(
    terrain: Terrain,
    orig: FloatArray,
    dirs: FloatArray,
    limits: RayLimits,
) -> FloatArray:
    """Run the kernel. See :meth:`pix2geo.backends.Backend.trace_rays`."""
    if len(dirs) == 0:
        return np.empty((4, 0))

    dev = _device_terrain(terrain)
    out = _launch(dev, terrain.pyramid.num_levels, orig, dirs, limits)

    return cp.asnumpy(out)
