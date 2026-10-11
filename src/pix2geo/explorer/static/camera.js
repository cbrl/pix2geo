// Camera math for the page: the pinhole model with OpenCV distortion of
// pix2geo.camera, and small WGS84 helpers. The server does all the ray casts.
// The page needs this math to draw the camera view and its overlays.

// A camera is {width, height, fx, fy, cx, cy, skew, dist: [k1, k2, p1, p2, k3, k4, k5, k6]}.

export const hasDistortion = (cam) => cam.dist.some((d) => d !== 0);

// Undistorted to distorted normalized coordinates, and the symmetric Jacobian.
function distortJ(cam, x, y) {
  const [k1, k2, p1, p2, k3, k4, k5, k6] = cam.dist;
  const r2 = x * x + y * y;
  const num = 1 + r2 * (k1 + r2 * (k2 + r2 * k3));
  const den = 1 + r2 * (k4 + r2 * (k5 + r2 * k6));
  const radial = num / den;
  const dNum = k1 + r2 * (2 * k2 + 3 * k3 * r2);
  const dDen = k4 + r2 * (2 * k5 + 3 * k6 * r2);
  const dRadial = (dNum - radial * dDen) / den;
  const xy = x * y;
  return {
    xd: x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x * x),
    yd: y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * xy,
    j11: radial + 2 * x * x * dRadial + 2 * p1 * y + 6 * p2 * x,
    j12: 2 * xy * dRadial + 2 * p1 * x + 2 * p2 * y,
    j22: radial + 2 * y * y * dRadial + 6 * p1 * y + 2 * p2 * x,
  };
}

export function distort(cam, x, y) {
  const d = distortJ(cam, x, y);
  return [d.xd, d.yd];
}

// Invert the distortion with Newton steps. No convergence gives NaN.
export function undistort(cam, xd, yd) {
  let x = xd;
  let y = yd;
  for (let i = 0; i < 25; i++) {
    const d = distortJ(cam, x, y);
    const ex = d.xd - xd;
    const ey = d.yd - yd;
    if (Math.abs(ex) + Math.abs(ey) < 1e-13) return [x, y];
    let det = d.j11 * d.j22 - d.j12 * d.j12;
    if (Math.abs(det) < 1e-12) det = 1e-12;
    x -= (d.j22 * ex - d.j12 * ey) / det;
    y -= (d.j11 * ey - d.j12 * ex) / det;
  }
  const [fx, fy] = distort(cam, x, y);
  return Math.hypot(fx - xd, fy - yd) < 1e-9 ? [x, y] : [NaN, NaN];
}

// Pixel to undistorted normalized image coordinates.
export function pixelToNormalized(cam, u, v) {
  const yn = (v - cam.cy) / cam.fy;
  const xn = (u - cam.cx - cam.skew * yn) / cam.fx;
  return hasDistortion(cam) ? undistort(cam, xn, yn) : [xn, yn];
}

// Undistorted normalized image coordinates to a pixel.
export function normalizedToPixel(cam, x, y) {
  const [xd, yd] = hasDistortion(cam) ? distort(cam, x, y) : [x, y];
  return [cam.fx * xd + cam.skew * yd + cam.cx, cam.fy * yd + cam.cy];
}

// A direction in the scene frame to a pixel. `axes` holds the camera x, y and
// z axes in the scene frame. Null for a direction behind the camera.
export function directionToPixel(cam, axes, d) {
  const dot = (a) => a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
  const z = dot(axes[2]);
  if (z <= 1e-9) return null;
  return normalizedToPixel(cam, dot(axes[0]) / z, dot(axes[1]) / z);
}

// The box of normalized coordinates that holds a rectangle of pixel edges
// [u0, u1] x [v0, v1], with a small margin. Distortion bends the edges, so
// the box comes from points along all four edges.
export function normalizedBox(cam, u0, v0, u1, v1, n = 16) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  const add = (u, v) => {
    const [x, y] = pixelToNormalized(cam, u, v);
    if (!Number.isFinite(x)) return;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  };
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    add(u0 + (u1 - u0) * t, v0);
    add(u0 + (u1 - u0) * t, v1);
    add(u0, v0 + (v1 - v0) * t);
    add(u1, v0 + (v1 - v0) * t);
  }
  if (!Number.isFinite(x0)) return null;
  const mx = 0.01 * (x1 - x0) + 1e-6;
  const my = 0.01 * (y1 - y0) + 1e-6;
  return [x0 - mx, y0 - my, x1 + mx, y1 + my];
}

// The row-major projection matrix that maps the normalized box [x0, x1] x
// [y0, y1] onto the clip square. The view space is the three.js camera space:
// x right, y up, and the camera looks down -z. The image y axis points down.
export function boxProjection(box, near, far) {
  const [x0, y0, x1, y1] = box;
  const dx = x1 - x0;
  const dy = y1 - y0;
  return [
    2 / dx, 0, 2 * x0 / dx + 1, 0,
    0, 2 / dy, -(1 + 2 * y0 / dy), 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, -1, 0,
  ];
}

export const hfov = (cam) => 2 * Math.atan2(cam.width / 2, cam.fx) * 180 / Math.PI;

// The camera with a new horizontal field of view. The pixel aspect stays.
export function withHfov(cam, degrees) {
  const fx = cam.width / 2 / Math.tan(degrees * Math.PI / 360);
  return { ...cam, fx, fy: cam.fy * fx / cam.fx };
}

// The camera with a new image size. The horizontal field of view and the
// pixel aspect stay. The principal point keeps its place in the image.
export function withSize(cam, width, height) {
  const s = width / cam.width;
  return {
    ...cam,
    width,
    height,
    fx: cam.fx * s,
    fy: cam.fy * s,
    cx: (cam.cx + 0.5) * width / cam.width - 0.5,
    cy: (cam.cy + 0.5) * height / cam.height - 0.5,
  };
}

// ---- WGS84 ------------------------------------------------------------------

const A = 6378137.0;
const E2 = 6.69437999014e-3;

// Degrees of latitude and of longitude for one meter north and one meter east.
export function degreesPerMeter(lat) {
  const s = Math.sin(lat * Math.PI / 180);
  const w = Math.sqrt(1 - E2 * s * s);
  const meridian = A * (1 - E2) / (w * w * w);
  const normal = A / w;
  const r2d = 180 / Math.PI;
  return [r2d / meridian, r2d / (normal * Math.max(Math.cos(lat * Math.PI / 180), 1e-6))];
}
