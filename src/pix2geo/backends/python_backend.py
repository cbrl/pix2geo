"""Python backend. Runs the shared kernel without compilation.

This backend is always available. It is slow, so use it for a few rays or
as a reference for the compiled backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._typing import FloatArray, IntArray
from ._kernel import _F, build_kernels, run_cpu_kernel

if TYPE_CHECKING:
    from ..terrain import Terrain


def _identity(func: _F) -> _F:
    return func


kernels = build_kernels(_identity, _identity, range)


def trace_rays(
    terrain: Terrain,
    orig: FloatArray,
    dirs: FloatArray,
    t0: FloatArray,
    t1: FloatArray,
    nseg: IntArray,
) -> FloatArray:
    """Run the kernel. See :meth:`pix2geo.backends.Backend.trace_rays`."""
    return run_cpu_kernel(kernels.trace_rays, terrain, orig, dirs, t0, t1, nseg)
