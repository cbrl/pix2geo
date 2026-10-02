"""Compute backends for the ray / terrain intersection kernel.

Backends
--------
``"numba"``
    Multi-threaded CPU code compiled by Numba. Needs ``numba``.
``"cupy"``
    One CUDA thread per ray. Needs ``cupy`` and an NVIDIA GPU.
``"python"``
    The same algorithm in plain Python. Always available, but slow. Use it
    for a few rays or as a reference.
``"auto"``
    Picks a backend for each call. Large batches go to CuPy when it is
    usable. Other batches go to Numba, then CuPy, then Python.

Each backend lives in the module ``<name>_backend``, which has a function
``trace_rays(terrain, orig, dirs, limits)`` and optionally
``check_usable()``, which raises an exception when the backend cannot run.

Set ``PIX2GEO_DISABLE_NUMBA=1`` or ``PIX2GEO_DISABLE_CUPY=1`` to hide a
backend, for example to test the fallback.

A backend that is requested by name and cannot run raises
:class:`BackendUnavailableError`. Its ``__cause__`` is the original error,
such as ``ImportError`` from a Numba that does not match NumPy. ``"auto"``
never raises it.
"""

from __future__ import annotations

import importlib
import os
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from .._typing import ArrayLike, FloatArray, RayLimits
from ._kernel import make_ray_limits

if TYPE_CHECKING:
    from ..terrain import Terrain

__all__ = [
    "AutoBackend",
    "Backend",
    "BackendUnavailableError",
    "CupyBackend",
    "NumbaBackend",
    "PythonBackend",
    "available_backends",
    "get_backend",
]


#: The available backend identifiers
BackendChoice = Literal["auto", "numba", "cupy", "python"]


#: Backend names in the order of preference for small batches.
_PREFERENCE: tuple[BackendChoice, ...] = ("numba", "cupy", "python")


def _disabled(name: str) -> bool:
    return os.environ.get(f"PIX2GEO_DISABLE_{name.upper()}", "").strip() not in ("", "0")


class BackendUnavailableError(RuntimeError):
    """A backend that was requested by name cannot run on this system."""


@cache
def _probe(name: str) -> ModuleType | Exception:
    """The module of backend ``name``, or the error that stops it on this system."""
    if name != "python" and _disabled(name):
        return RuntimeError(f"PIX2GEO_DISABLE_{name.upper()} is set")

    # Keep the error. It tells the user what to fix when they request the backend.
    try:
        module = importlib.import_module(f"{__name__}.{name}_backend")
        check_usable = getattr(module, "check_usable", None)
        if check_usable is not None:
            check_usable()
    except Exception as exc:  # not installed, or a broken install (for example no CUDA)
        return exc

    return module


def _load(name: str) -> ModuleType | None:
    """The module of backend ``name``, or None when it cannot run on this system."""
    found = _probe(name)

    return found if isinstance(found, ModuleType) else None


class Backend:
    """Runs the intersection kernel on a batch of ECEF rays."""

    name = "base"

    def trace_rays(
        self,
        terrain: Terrain,
        orig: ArrayLike,
        dirs: ArrayLike,
        *,
        max_range: float | None = None,
        max_segment_length: float = 500.0,
    ) -> FloatArray:
        """Find the first terrain hit on each ray.

        Ray ``r`` is ``orig[r] + t * dirs[r]`` for ``t >= 0``, with unit
        directions ``dirs`` of shape ``(n, 3)``. ``orig`` has the shape
        ``(n, 3)``, or ``(1, 3)`` for one origin for all the rays. The kernel
        clips each ray to the height band of the terrain and to ``max_range``.
        It cuts the rest into equal segments of at most ``max_segment_length``
        meters.

        Returns a ``(4, n)`` array: the ray parameter ``t`` of the first hit,
        and the latitude, longitude (degrees) and ellipsoidal height of the
        hit point. A miss gives NaN in all four rows.
        """
        return self._trace_rays(
            terrain,
            np.ascontiguousarray(orig, dtype=np.float64),
            np.ascontiguousarray(dirs, dtype=np.float64),
            make_ray_limits(terrain, max_range, max_segment_length),
        )

    def _trace_rays(
        self,
        terrain: Terrain,
        orig: FloatArray,
        dirs: FloatArray,
        limits: RayLimits,
    ) -> FloatArray:
        """Subclass hook. The arrays are C-contiguous float64."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<pix2geo backend {self.name!r}>"


class _ModuleBackend(Backend):
    """A backend that runs the ``trace_rays`` function of its backend module."""

    requirements = "a working install of pix2geo"

    def __init__(self) -> None:
        found = _probe(self.name)
        if not isinstance(found, ModuleType):
            raise BackendUnavailableError(
                f"the {self.name} backend is not available "
                f"({type(found).__name__}: {found}). It needs {self.requirements}."
            ) from found

        self._module: ModuleType = found

    def _trace_rays(
        self,
        terrain: Terrain,
        orig: FloatArray,
        dirs: FloatArray,
        limits: RayLimits,
    ) -> FloatArray:
        return self._module.trace_rays(terrain, orig, dirs, limits)


class PythonBackend(_ModuleBackend):
    name = "python"


class NumbaBackend(_ModuleBackend):
    name = "numba"
    requirements = "numba: pip install numba"


class CupyBackend(_ModuleBackend):
    name = "cupy"
    requirements = "cupy, an NVIDIA GPU, and a CUDA compiler"


class AutoBackend(Backend):
    """Chooses a backend for each call from the batch size.

    The check for CuPy runs only when a batch could use it, so small
    batches never pay the CUDA start-up time.
    """

    name = "auto"

    def __init__(self, gpu_min_rays: int = 20000) -> None:
        self.gpu_min_rays = int(gpu_min_rays)

    def select(self, n_rays: int) -> Backend:
        """The backend that a batch of ``n_rays`` rays uses."""
        if n_rays >= self.gpu_min_rays and _load("cupy") is not None:
            return get_backend("cupy")

        name = next(name for name in _PREFERENCE if _load(name) is not None)
        return get_backend(cast(BackendChoice, name))

    def _trace_rays(
        self,
        terrain: Terrain,
        orig: FloatArray,
        dirs: FloatArray,
        limits: RayLimits,
    ) -> FloatArray:
        return self.select(len(dirs))._trace_rays(terrain, orig, dirs, limits)


_BACKENDS: dict[BackendChoice, type[Backend]] = {
    "auto": AutoBackend,
    "numba": NumbaBackend,
    "cupy": CupyBackend,
    "python": PythonBackend,
}


def available_backends() -> list[str]:
    """Names of the backends that work on this system, in order of preference."""
    return [name for name in _PREFERENCE if _load(name) is not None]


def get_backend(name: BackendChoice | Backend = "auto") -> Backend:
    """Make a backend from its name. A :class:`Backend` passes through."""
    if isinstance(name, Backend):
        return name

    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"unknown backend {name!r}; use 'auto', 'numba', 'cupy' or 'python'")

    return cls()
