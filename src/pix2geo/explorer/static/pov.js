// The camera view: the terrain as the camera sees it, with its intrinsics and
// lens distortion, and a 2D overlay of the picks, the rays and the layers.
//
// The render has two passes. The first renders the terrain with a pinhole
// projection of a box of normalized coordinates into a render target. The
// second (REMAP_FRAG) maps each screen pixel to its image pixel, undoes the
// distortion, and reads the render target. The view can zoom and pan, so
// the first pass covers only the visible part of the image.

import * as THREE from 'three';

import { REMAP_FRAG, REMAP_VERT } from './shaders.js';
import { boxProjection, directionToPixel, hasDistortion, normalizedBox } from './camera.js';
import { clamp } from './util.js';

const V3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// The largest distance in CSS pixels from a pick for a drag.
const PICK_RADIUS = 10;

export class PovView {
  constructor(container, overlay, scene3d, handlers) {
    this.container = container;
    this.overlay = overlay;
    this.ctx = overlay.getContext('2d');
    this.scene3d = scene3d;
    this.handlers = handlers;
    this.zoom = 1;
    this.center = null;
    this.cam = null;
    this.state = null;
    this.horizon = [];
    this.mode = null;

    // The camera of the pinhole pass. Its matrices come from the pose.
    this.camera = new THREE.PerspectiveCamera();
    this.camera.matrixAutoUpdate = false;
    this.camera.near = 0.5;
    this.camera.far = 1e7;
    this.camera.layers.set(0);

    this.pinhole = new THREE.WebGLRenderTarget(1, 1, { samples: 4 });
    this.pinholeFull = new THREE.WebGLRenderTarget(1, 1, { samples: 4 });
    this.image = new THREE.WebGLRenderTarget(1, 1);

    this.remapMaterial = new THREE.ShaderMaterial({
      vertexShader: REMAP_VERT,
      fragmentShader: REMAP_FRAG,
      uniforms: {
        uTex: { value: null },
        uRegion: { value: new THREE.Vector4() },
        uBox: { value: new THREE.Vector4(0, 0, 1, 1) },
        uK: { value: new THREE.Vector4() },
        uSkew: { value: 0 },
        uDist: { value: new Array(8).fill(0) },
        uHasDist: { value: 0 },
        uSize: { value: new THREE.Vector2(1, 1) },
        uUp: { value: new THREE.Vector3(0, -1, 0) },
        uSinDip: { value: 0 },
        uSkyTop: { value: new THREE.Color('#5d8fc9') },
        uSkyLow: { value: new THREE.Color('#c4d6e8') },
        // The ground past the drawn terrain is far away, so it has a haze color.
        uGround: { value: new THREE.Color('#9eaab2') },
        uLetterbox: { value: new THREE.Color('#111317') },
      },
      depthTest: false,
      depthWrite: false,
    });
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.remapMaterial);
    quad.frustumCulled = false;
    this.quadScene = new THREE.Scene();
    this.quadScene.add(quad);
    this.quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    this.bindEvents();
  }

  setTheme(c) {
    this.colors = c;
    this.remapMaterial.uniforms.uLetterbox.value.set(c.letterbox);
  }

  setCamera(cam) {
    const resized = !this.cam || this.cam.width !== cam.width || this.cam.height !== cam.height;
    this.cam = cam;
    if (resized) this.fit();
    this.fullBox = normalizedBox(cam, -0.5, -0.5, cam.width - 0.5, cam.height - 0.5);
    if (this.state) this.computeHorizon();
  }

  // The camera state of /api/cast: position, axes, local frame and horizon dip.
  setState(state) {
    this.state = state;
    this.computeHorizon();
  }

  fit() {
    this.zoom = 1;
    this.center = null;
  }

  // ---- Display transform ------------------------------------------------------

  // CSS pixels per image pixel, and the image pixel edges at the view edges.
  view() {
    const rect = this.container.getBoundingClientRect();
    const { width: W, height: H } = this.cam;
    const s = Math.min(rect.width / W, rect.height / H) * this.zoom;
    const c = this.center ?? [W / 2 - 0.5, H / 2 - 0.5];
    const u0 = c[0] - rect.width / (2 * s);
    const v0 = c[1] - rect.height / (2 * s);
    return { rect, s, u0, v0, u1: u0 + rect.width / s, v1: v0 + rect.height / s };
  }

  pixelAt(clientX, clientY) {
    const v = this.view();
    return [v.u0 + (clientX - v.rect.left) / v.s, v.v0 + (clientY - v.rect.top) / v.s];
  }

  inImage(u, v) {
    return u >= -0.5 && v >= -0.5 && u <= this.cam.width - 0.5 && v <= this.cam.height - 0.5;
  }

  // Zoom by `factor` and keep the image pixel under the client point in place.
  zoomAt(factor, clientX, clientY) {
    const before = this.pixelAt(clientX, clientY);
    this.zoom = clamp(this.zoom * factor, 1, 64);
    if (this.zoom === 1) {
      this.center = null;
      return;
    }
    const v = this.view();
    const after = [v.u0 + (clientX - v.rect.left) / v.s, v.v0 + (clientY - v.rect.top) / v.s];
    const c = this.center ?? [this.cam.width / 2 - 0.5, this.cam.height / 2 - 0.5];
    this.center = [c[0] + before[0] - after[0], c[1] + before[1] - after[1]];
    this.clampCenter();
  }

  clampCenter() {
    if (!this.center) return;
    const { width: W, height: H } = this.cam;
    this.center[0] = clamp(this.center[0], -0.5, W - 0.5);
    this.center[1] = clamp(this.center[1], -0.5, H - 0.5);
  }

  // ---- Render -----------------------------------------------------------------

  render(renderer, canvasHeight) {
    const v = this.view();
    const rect = v.rect;
    if (rect.width < 2 || rect.height < 2) return;
    const y = canvasHeight - rect.bottom;
    const ready = this.cam && this.state && this.scene3d.terrain;

    // The pinhole pass covers the visible part of the image, at about one
    // texel for each screen pixel.
    let box = null;
    if (ready) {
      const a0 = Math.max(v.u0, -0.5), a1 = Math.min(v.u1, this.cam.width - 0.5);
      const b0 = Math.max(v.v0, -0.5), b1 = Math.min(v.v1, this.cam.height - 0.5);
      if (a1 > a0 && b1 > b0) box = normalizedBox(this.cam, a0, b0, a1, b1);
      if (box) {
        const k = v.s * renderer.getPixelRatio();
        const w = clamp(Math.ceil((box[2] - box[0]) * this.cam.fx * k), 1, 4096);
        const h = clamp(Math.ceil((box[3] - box[1]) * this.cam.fy * k), 1, 4096);
        this.renderPinhole(renderer, box, this.pinhole, w, h);
      }
    }

    renderer.setRenderTarget(null);
    renderer.setViewport(rect.left, y, rect.width, rect.height);
    renderer.setScissor(rect.left, y, rect.width, rect.height);
    this.remap(renderer, box ? this.pinhole.texture : null, box, [v.u0, v.v0, v.u1, v.v1], ready);
  }

  // The full image at most 1024 pixels wide, into `this.image`, for the 3D view.
  renderImage(renderer) {
    if (!this.cam || !this.state || !this.scene3d.terrain || !this.fullBox) return;
    const { width: W, height: H } = this.cam;
    const w = Math.min(1024, W);
    const h = Math.max(1, Math.round((w * H) / W));
    const box = this.fullBox;
    const k = w / W;
    this.renderPinhole(
      renderer,
      box,
      this.pinholeFull,
      clamp(Math.ceil((box[2] - box[0]) * this.cam.fx * k), 1, 4096),
      clamp(Math.ceil((box[3] - box[1]) * this.cam.fy * k), 1, 4096),
    );
    this.image.setSize(w, h);
    renderer.setRenderTarget(this.image);
    this.remap(renderer, this.pinholeFull.texture, box, [-0.5, -0.5, W - 0.5, H - 0.5], true);
    renderer.setRenderTarget(null);
  }

  renderPinhole(renderer, box, target, w, h) {
    const st = this.state;
    const [ax, ay, az] = st.axes;

    // three.js cameras look down -z with y up: the columns are x, -y and -z
    // of the camera optical frame.
    const cam3 = this.camera;
    cam3.matrix.makeBasis(V3(ax), V3(ay).negate(), V3(az).negate()).setPosition(V3(st.position));
    cam3.matrixWorldNeedsUpdate = true;
    cam3.projectionMatrix.set(...boxProjection(box, cam3.near, cam3.far));
    cam3.projectionMatrixInverse.copy(cam3.projectionMatrix).invert();

    // The camera sees the true terrain, without the exaggeration of the 3D view.
    const sc = this.scene3d;
    sc.root.scale.z = 1;
    const u = sc.terrainMaterial.uniforms;
    u.uPass.value = 1;
    u.uFogDistance.value = Math.max(20000, 40 * Math.abs(st.agl ?? 1000));

    target.setSize(w, h);
    renderer.setRenderTarget(target);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(sc.scene, cam3);
    renderer.setRenderTarget(null);
    sc.root.scale.z = sc.ve;
    u.uPass.value = 0;
  }

  remap(renderer, texture, box, region, ready) {
    const u = this.remapMaterial.uniforms;
    const cam = this.cam;
    u.uTex.value = texture;
    u.uRegion.value.set(...region);
    u.uBox.value.set(...(box ?? [0, 0, 1, 1]));
    u.uSize.value.set(cam ? cam.width : 1, cam ? cam.height : 1);
    if (cam) {
      u.uK.value.set(cam.fx, cam.fy, cam.cx, cam.cy);
      u.uSkew.value = cam.skew;
      u.uDist.value = cam.dist;
      u.uHasDist.value = hasDistortion(cam) ? 1 : 0;
    }
    if (ready) {
      // The local up direction in the camera frame splits the sky from the ground.
      const [ax, ay, az] = this.state.axes;
      const up = this.state.local[2];
      u.uUp.value.set(dot(ax, up), dot(ay, up), dot(az, up));
      u.uSinDip.value = Math.sin((this.state.dip * Math.PI) / 180);
    }
    renderer.render(this.quadScene, this.quadCamera);
  }

  // The horizon of the ellipsoid in the image: directions at the dip angle
  // below the local horizontal, all around the camera. Only directions
  // inside the image box (plus a margin) go through the distortion model.
  computeHorizon() {
    const st = this.state;
    const cam = this.cam;
    this.horizon = [];
    if (!st || !cam || !this.fullBox) return;
    const [E, N, U] = st.local;
    const dip = (st.dip * Math.PI) / 180;
    const [x0, y0, x1, y1] = this.fullBox;
    const mx = 0.1 * (x1 - x0), my = 0.1 * (y1 - y0);
    const n = 720;
    for (let i = 0; i <= n; i++) {
      const az = (2 * Math.PI * i) / n;
      const h = Math.cos(dip);
      const d = [0, 1, 2].map((k) => h * (Math.sin(az) * E[k] + Math.cos(az) * N[k]) - Math.sin(dip) * U[k]);
      const z = dot(st.axes[2], d);
      let px = null;
      if (z > 1e-6) {
        const x = dot(st.axes[0], d) / z, y = dot(st.axes[1], d) / z;
        if (x > x0 - mx && x < x1 + mx && y > y0 - my && y < y1 + my) px = directionToPixel(cam, st.axes, d);
      }
      this.horizon.push(px);
    }
  }

  // ---- Overlay ------------------------------------------------------------------

  // Draw the 2D overlay. `o` holds the picks and the display options.
  drawOverlay(o) {
    const canvas = this.overlay;
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.container.getBoundingClientRect();
    const cw = Math.round(rect.width * dpr), ch = Math.round(rect.height * dpr);
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!this.cam) return;

    const v = this.view();
    const { width: W, height: H } = this.cam;
    const X = (u) => (u - v.u0) * v.s;
    const Y = (vv) => (vv - v.v0) * v.s;
    const c = this.colors;

    ctx.save();
    ctx.beginPath();
    ctx.rect(X(-0.5), Y(-0.5), W * v.s, H * v.s);
    ctx.clip();

    // The photo, then the pix2geo layer. Each sample of the layer is the
    // center of a cell of `step` pixels.
    if (o.photo && o.photoOpacity > 0) {
      ctx.globalAlpha = o.photoOpacity;
      ctx.drawImage(o.photo, X(-0.5), Y(-0.5), W * v.s, H * v.s);
    }
    if (o.layer && o.layerOpacity > 0) {
      const { canvas: lc, step, cols, rows } = o.layer;
      ctx.globalAlpha = o.layerOpacity;
      ctx.imageSmoothingEnabled = v.s * step < 24;
      ctx.drawImage(lc, X(-step / 2), Y(-step / 2), cols * step * v.s, rows * step * v.s);
      ctx.imageSmoothingEnabled = true;
    }
    ctx.globalAlpha = 1;

    // The dots of the fan rays, one path for each ray state. A dark edge
    // keeps the dots visible on light terrain.
    if (o.grid) {
      const { pixels, status } = o.grid;
      const paths = [new Path2D(), new Path2D(), new Path2D()];
      for (let i = 0; i < pixels.length; i++) {
        const x = X(pixels[i][0]), y = Y(pixels[i][1]);
        paths[status[i]].moveTo(x + 2.6, y);
        paths[status[i]].arc(x, y, 2.6, 0, 2 * Math.PI);
      }
      ctx.lineWidth = 1;
      ctx.strokeStyle = c.miss;
      ctx.stroke(paths[0]);
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
      for (const [s, fill] of [[1, c.ray], [2, c.fallback]]) {
        ctx.fillStyle = fill;
        ctx.fill(paths[s]);
        ctx.stroke(paths[s]);
      }
    }

    // The horizon of the ellipsoid.
    if (o.horizon && this.horizon.length) {
      ctx.setLineDash([7, 5]);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      let pen = false;
      for (const p of this.horizon) {
        if (!p) {
          pen = false;
          continue;
        }
        if (pen) ctx.lineTo(X(p[0]), Y(p[1]));
        else ctx.moveTo(X(p[0]), Y(p[1]));
        pen = true;
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // The measured path between the picks.
    if (o.measure) {
      const pts = o.picks.filter((p) => p.res && p.res.status !== 'miss');
      if (pts.length > 1) {
        ctx.setLineDash([8, 5]);
        ctx.strokeStyle = c.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach((p, i) => (i ? ctx.lineTo(X(p.u), Y(p.v)) : ctx.moveTo(X(p.u), Y(p.v))));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.restore();

    // The image edge and the principal point.
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(X(-0.5) + 0.5, Y(-0.5) + 0.5, W * v.s - 1, H * v.s - 1);
    const pcx = X(this.cam.cx), pcy = Y(this.cam.cy);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.beginPath();
    ctx.moveTo(pcx - 6, pcy); ctx.lineTo(pcx + 6, pcy);
    ctx.moveTo(pcx, pcy - 6); ctx.lineTo(pcx, pcy + 6);
    ctx.stroke();

    // The picks: a ring with ticks, a dark halo and a number.
    o.picks.forEach((p, i) => {
      const sel = p.id === o.selected;
      this.drawPick(ctx, X(p.u), Y(p.v), sel ? c.selected : c.accent, String(i + 1), sel, p.res?.status === 'miss');
    });

    // The pixel under the mouse.
    if (o.hover) {
      const x = X(o.hover[0]), y = Y(o.hover[1]);
      ctx.strokeStyle = c.hover;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }

  drawPick(ctx, x, y, color, text, selected, missed) {
    const r = selected ? 8 : 6.5;
    const ring = (width, style) => {
      ctx.lineWidth = width;
      ctx.strokeStyle = style;
      ctx.setLineDash(missed ? [3, 3] : []);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.moveTo(x - r - 5, y); ctx.lineTo(x - r + 2, y);
      ctx.moveTo(x + r - 2, y); ctx.lineTo(x + r + 5, y);
      ctx.moveTo(x, y - r - 5); ctx.lineTo(x, y - r + 2);
      ctx.moveTo(x, y + r - 2); ctx.lineTo(x, y + r + 5);
      ctx.stroke();
    };
    ring(4.5, 'rgba(0, 0, 0, 0.55)');
    ring(2, color);
    ctx.setLineDash([]);

    // The number in a pill at the upper right.
    ctx.font = '600 11px system-ui, sans-serif';
    const w = Math.max(16, ctx.measureText(text).width + 9);
    const bx = x + r + 3, by = y - r - 15;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(bx, by, w, 15, 7.5);
    ctx.fill();
    ctx.fillStyle = selected ? '#111' : '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, bx + w / 2, by + 8);
  }

  // ---- Mouse --------------------------------------------------------------------

  pickNear(clientX, clientY, picks) {
    const v = this.view();
    let best = null, bestD = PICK_RADIUS;
    for (const p of picks) {
      const x = v.rect.left + (p.u - v.u0) * v.s;
      const y = v.rect.top + (p.v - v.v0) * v.s;
      const d = Math.hypot(x - clientX, y - clientY);
      if (d < bestD) [best, bestD] = [p, d];
    }
    return best;
  }

  bindEvents() {
    const el = this.container;
    const h = this.handlers;
    el.addEventListener('contextmenu', (e) => e.preventDefault());

    el.addEventListener('pointerdown', (e) => {
      if (!this.cam) return;
      el.setPointerCapture(e.pointerId);
      this.start = [e.clientX, e.clientY];
      this.last = [e.clientX, e.clientY];
      if (e.button === 0) {
        const near = this.pickNear(e.clientX, e.clientY, h.picks());
        this.mode = near ? 'drag' : 'click';
        this.dragId = near?.id;
        if (near) h.onSelect(near.id);
      } else {
        this.mode = 'turn';
      }
      el.classList.toggle('dragging', this.mode !== 'click');
    });

    el.addEventListener('pointermove', (e) => {
      if (!this.cam) return;
      const [u, v] = this.pixelAt(e.clientX, e.clientY);
      const dx = e.clientX - this.last?.[0], dy = e.clientY - this.last?.[1];
      if (this.mode) this.last = [e.clientX, e.clientY];

      if (this.mode === 'drag') {
        const W = this.cam.width, H = this.cam.height;
        h.onDragPick(this.dragId, clamp(u, -0.5, W - 0.5), clamp(v, -0.5, H - 0.5));
      } else if (this.mode === 'turn') {
        const s = this.view().s;
        h.onTurn(dx / s, dy / s);
      } else if (this.mode === 'click' || this.mode === 'pan') {
        const moved = Math.hypot(e.clientX - this.start[0], e.clientY - this.start[1]);
        if (this.mode === 'click' && moved > 4 && this.zoom > 1) this.mode = 'pan';
        if (this.mode === 'pan') {
          const s = this.view().s;
          const c = this.center ?? [this.cam.width / 2 - 0.5, this.cam.height / 2 - 0.5];
          this.center = [c[0] - dx / s, c[1] - dy / s];
          this.clampCenter();
          h.onView();
        }
      } else {
        el.classList.toggle('over-pick', !!this.pickNear(e.clientX, e.clientY, h.picks()));
        h.onHover(this.inImage(u, v) ? [u, v] : null, e.clientX, e.clientY);
      }
    });

    el.addEventListener('pointerup', (e) => {
      if (this.mode === 'click') {
        const [u, v] = this.pixelAt(e.clientX, e.clientY);
        const moved = Math.hypot(e.clientX - this.start[0], e.clientY - this.start[1]);
        if (moved <= 4 && this.inImage(u, v)) h.onClick(u, v);
      }
      if (this.mode === 'drag') h.onDragEnd(this.dragId);
      this.mode = null;
      el.classList.remove('dragging');
    });

    el.addEventListener('pointerleave', () => {
      if (!this.mode) h.onHover(null);
    });

    el.addEventListener('wheel', (e) => {
      if (!this.cam) return;
      e.preventDefault();
      // Shift turns a vertical wheel into a horizontal one on some systems.
      const factor = Math.exp(-(e.deltaY || e.deltaX) * 0.0015);
      if (e.shiftKey) {
        h.onFov(1 / factor);
      } else {
        this.zoomAt(factor, e.clientX, e.clientY);
        h.onView();
      }
    }, { passive: false });
  }
}
