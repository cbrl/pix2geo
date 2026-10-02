"""Camera intrinsics: pixel <-> ray conversion in the camera optical frame.

The camera optical frame follows the OpenCV convention:

* +X points to the right of the image
* +Y points down the image
* +Z points forward along the optical axis

Pixel coordinates have their origin at the top-left of the image. Integer
coordinates are the centers of pixels, as in OpenCV. The top-left pixel center
is ``(0, 0)`` and the top-left image corner is ``(-0.5, -0.5)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ._typing import ArrayLike, BoolArray, FloatArray

__all__ = ["CameraIntrinsics"]


def _as_dist(dist: Sequence[float] | None) -> tuple[float, ...]:
    if dist is None:
        return ()

    d = tuple(float(x) for x in np.asarray(dist, dtype=float).ravel())
    if len(d) not in (0, 4, 5, 8):
        raise ValueError(
            "distortion must have 4, 5 or 8 OpenCV coefficients "
            "(k1, k2, p1, p2[, k3[, k4, k5, k6]])"
        )
    if not any(d):
        return ()

    return d


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics with optional OpenCV lens distortion.

    Parameters
    ----------
    width, height:
        Image size in pixels.
    fx, fy:
        Focal lengths in pixels.
    cx, cy:
        Principal point in pixels (pixel-center convention, see module doc).
    skew:
        Axis skew term ``K[0, 1]``. Usually zero.
    dist:
        OpenCV distortion coefficients ``(k1, k2, p1, p2[, k3[, k4, k5, k6]])``.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0
    dist: tuple[float, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "dist", _as_dist(self.dist))

        if self.width < 1 or self.height < 1:
            raise ValueError("the image size must be positive")
        if not np.all(np.isfinite([self.fx, self.fy, self.cx, self.cy, self.skew, *self.dist])):
            raise ValueError("the intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")

    # ---- Constructors -------------------------------------------------------

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov: float | None = None,
        vfov: float | None = None,
        dfov: float | None = None,
        degrees: bool = True,
        dist: Sequence[float] | None = None,
    ) -> CameraIntrinsics:
        """Make intrinsics from the image resolution and the field of view.

        Give one of ``hfov``, ``vfov`` or ``dfov`` (diagonal) for square
        pixels. Give ``hfov`` and ``vfov`` together for non-square pixels.
        The field of view spans the full image, edge to edge. The principal
        point is the image center.
        """

        def focal(size: float, fov: float) -> float:
            # Focal length in pixels for a field of view that spans `size` pixels.
            # A pinhole camera cannot see 180 degrees or more. NaN also fails here.
            angle = np.radians(fov) if degrees else fov
            if not 0.0 < angle < np.pi:
                raise ValueError("the field of view must be more than 0 and less than 180 degrees")
            return float((size / 2.0) / np.tan(angle / 2.0))

        # One field of view gives square pixels.
        if dfov is not None:
            if hfov is not None or vfov is not None:
                raise ValueError("dfov cannot be combined with hfov or vfov")
            fx = fy = focal(float(np.hypot(width, height)), dfov)
        elif hfov is not None and vfov is not None:
            fx, fy = focal(width, hfov), focal(height, vfov)
        elif hfov is not None:
            fx = fy = focal(width, hfov)
        elif vfov is not None:
            fx = fy = focal(height, vfov)
        else:
            raise ValueError("give at least one of hfov, vfov or dfov")

        return cls(
            width=int(width),
            height=int(height),
            fx=fx,
            fy=fy,
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            dist=_as_dist(dist),
        )

    @classmethod
    def from_matrix(
        cls,
        K: ArrayLike,
        width: int,
        height: int,
        dist: Sequence[float] | None = None,
    ) -> CameraIntrinsics:
        """Make intrinsics from a 3x3 camera matrix (OpenCV layout).

        ``K`` must have the form ``[[fx, s, cx], [0, fy, cy], [0, 0, 1]]``, up
        to a scale factor.
        """
        m = np.asarray(K, dtype=float)
        if m.shape != (3, 3) or not np.all(np.isfinite(m)) or m[2, 2] == 0:
            raise ValueError("K must be a finite 3x3 matrix with K[2, 2] != 0")

        # K is upper triangular. This check finds a transposed K, which would
        # otherwise give a principal point of (0, 0).
        m = m / m[2, 2]
        lower = np.abs([m[1, 0], m[2, 0], m[2, 1]])
        if lower.max() > 1e-9 * np.abs(m).max():
            raise ValueError("K must have the form [[fx, s, cx], [0, fy, cy], [0, 0, 1]]")

        return cls(
            width=int(width),
            height=int(height),
            fx=float(m[0, 0]),
            fy=float(m[1, 1]),
            cx=float(m[0, 2]),
            cy=float(m[1, 2]),
            skew=float(m[0, 1]),
            dist=_as_dist(dist),
        )

    # ---- Properties ---------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        """The 3x3 camera matrix."""
        return np.array(
            [
                [self.fx, self.skew, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def hfov(self) -> float:
        """Horizontal field of view in degrees (no distortion)."""
        return float(np.degrees(2 * np.arctan2(self.width / 2.0, self.fx)))

    @property
    def vfov(self) -> float:
        """Vertical field of view in degrees (no distortion)."""
        return float(np.degrees(2 * np.arctan2(self.height / 2.0, self.fy)))

    # ---- Distortion Models --------------------------------------------------

    def _distort(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Undistorted to distorted normalized image coordinates."""
        # Missing coefficients are zero.
        d = self.dist + (0.0,) * (8 - len(self.dist))
        k1, k2, p1, p2, k3, k4, k5, k6 = d

        # The radial factor is a ratio of polynomials in r^2 (k1 to k6). The
        # tangential terms (p1, p2) model a lens that is not parallel to the sensor.
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

        return xd, yd

    def _distort_with_jacobian(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """:meth:`_distort`, and its Jacobian ``[[j11, j12], [j12, j22]]`` (symmetric)."""
        d = self.dist + (0.0,) * (8 - len(self.dist))
        k1, k2, p1, p2, k3, k4, k5, k6 = d

        # The radial factor num / den and its derivative with respect to r^2.
        r2 = x * x + y * y
        num = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        den = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        radial = num / den
        d_num = k1 + r2 * (2 * k2 + 3 * k3 * r2)
        d_den = k4 + r2 * (2 * k5 + 3 * k6 * r2)
        d_radial = (d_num - radial * d_den) / den

        xy = x * y
        xd = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * xy

        # The partial derivatives. d(r^2)/dx = 2 x and d(r^2)/dy = 2 y.
        j11 = radial + 2 * x * x * d_radial + 2 * p1 * y + 6 * p2 * x
        j12 = 2 * xy * d_radial + 2 * p1 * x + 2 * p2 * y
        j22 = radial + 2 * y * y * d_radial + 6 * p1 * y + 2 * p2 * x

        return xd, yd, j11, j12, j22

    def _undistort(
        self,
        xd: FloatArray,
        yd: FloatArray,
        iterations: int = 20,
    ) -> tuple[FloatArray, FloatArray]:
        """Invert the distortion model with Newton iterations.

        Points where the model has no inverse (outside the valid area of the
        lens model) give NaN.
        """
        # The first guess is the distorted point itself. Distortion is small
        # near the image center.
        x = xd.copy()
        y = yd.copy()

        for _ in range(iterations):
            # The error (ex, ey) of the current guess, after distortion.
            fx_, fy_, j11, j12, j22 = self._distort_with_jacobian(x, y)
            ex = fx_ - xd
            ey = fy_ - yd
            if np.all(np.abs(ex) + np.abs(ey) < 1e-14):
                break

            # Newton step: solve J (step_x, step_y) = (ex, ey) with the 2x2 inverse.
            # The determinant clamp prevents a division by zero.
            det = j11 * j22 - j12 * j12
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            x = x - (j22 * ex - j12 * ey) / det
            y = y - (j11 * ey - j12 * ex) / det
        else:
            # No break: get the error of the last step.
            fx_, fy_ = self._distort(x, y)
            ex = fx_ - xd
            ey = fy_ - yd

        # Points that did not converge have no inverse.
        bad = np.hypot(ex, ey) > 1e-9
        if np.any(bad):
            x = np.where(bad, np.nan, x)
            y = np.where(bad, np.nan, y)

        return x, y

    # ---- Pixel <-> Ray ------------------------------------------------------

    def pixel_to_normalized(self, u: ArrayLike, v: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Pixel coordinates to undistorted normalized image coordinates."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # Invert K. Solve the row first, because the column has the skew term.
        yn = (v - self.cy) / self.fy
        xn = (u - self.cx - self.skew * yn) / self.fx
        if self.dist:
            xn, yn = self._undistort(xn, yn)

        return xn, yn

    def pixel_to_ray(self, u: ArrayLike, v: ArrayLike) -> FloatArray:
        """Pixel coordinates to unit ray directions in the camera frame.

        Returns an array of shape ``(..., 3)``.
        """
        # The point (xn, yn, 1) on the z = 1 plane gives the ray direction.
        # Divide it by its length, and write it into the output in place.
        xn, yn = self.pixel_to_normalized(u, v)
        inv = 1.0 / np.sqrt(xn * xn + yn * yn + 1.0)
        rays = np.empty((*inv.shape, 3))
        np.multiply(xn, inv, out=rays[..., 0])
        np.multiply(yn, inv, out=rays[..., 1])
        rays[..., 2] = inv

        return rays

    def ray_to_pixel(self, rays: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Camera-frame directions (or points) to pixel coordinates.

        Directions with ``z <= 0`` point behind the camera and give NaN.
        """
        rays = np.asarray(rays, dtype=float)
        # Project onto the z = 1 plane: the normalized image coordinates.
        z = rays[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(z > 0, rays[..., 0] / z, np.nan)
            y = np.where(z > 0, rays[..., 1] / z, np.nan)
        if self.dist:
            x, y = self._distort(x, y)

        # Apply K.
        u = self.fx * x + self.skew * y + self.cx
        v = self.fy * y + self.cy

        return u, v

    def contains(self, u: ArrayLike, v: ArrayLike) -> BoolArray:
        """True where a pixel coordinate is inside the image area."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        return (u >= -0.5) & (u <= self.width - 0.5) & (v >= -0.5) & (v <= self.height - 0.5)
