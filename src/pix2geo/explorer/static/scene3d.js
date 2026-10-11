// The 3D scene: the terrain, the camera, its frustum, the rays and the picks.
//
// All the geometry is in the scene frame of the server: east, north and up in
// meters. The root group scales the up axis for the vertical exaggeration of
// the 3D view. The camera view renders the same scene with the scale at 1, and
// it sees only layer 0 (the terrain). Everything else is on layer 1.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

import { POINT_FRAG, POINT_VERT, TERRAIN_FRAG, TERRAIN_VERT } from './shaders.js';
import { ORANGES, lut, terrainLut } from './colormaps.js';
import { pixelToNormalized } from './camera.js';
import { niceStep } from './util.js';

export const LAYER_3D = 1;

const only3d = (obj) => {
  obj.traverse((o) => o.layers.set(LAYER_3D));
  return obj;
};

const color = (css) => new THREE.Color(css || '#888888');

function rampTexture(bytes) {
  const tex = new THREE.DataTexture(bytes, bytes.length / 4, 1, THREE.RGBAFormat);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

// Move the line ends toward the eye in view space, by the factor `pull`. The
// projection does not change, but a line on the terrain then stays in front
// of the drawn mesh, which has fewer posts than the DEM of the ray casts.
function pullLines(material, pull) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uPull = { value: pull };
    shader.vertexShader = shader.vertexShader
      .replace('uniform float linewidth;', 'uniform float linewidth;\nuniform float uPull;')
      .replace(
        'vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );',
        'vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );\nstart.xyz *= uPull;\nend.xyz *= uPull;',
      );
  };
}

// Wide line segments in pixels. `set` takes flat pairs of points.
class Segments {
  constructor(parent, materials, opts) {
    this.material = new LineMaterial({
      color: 0xffffff,
      linewidth: opts.width ?? 2,
      dashed: !!opts.dashed,
      dashSize: opts.dashSize ?? 1,
      gapSize: opts.gapSize ?? 1,
      transparent: (opts.opacity ?? 1) < 1,
      opacity: opts.opacity ?? 1,
      depthTest: opts.depthTest ?? true,
      vertexColors: !!opts.vertexColors,
      worldUnits: false,
    });
    pullLines(this.material, opts.pull ?? 0.998);
    materials.push(this.material);
    this.object = only3d(new LineSegments2(new LineSegmentsGeometry(), this.material));
    this.object.frustumCulled = false;
    this.object.visible = false;
    if (opts.order != null) this.object.renderOrder = opts.order;
    parent.add(this.object);
  }

  set(points, colors = null) {
    this.object.geometry.dispose();
    const geom = new LineSegmentsGeometry();
    if (points.length) {
      geom.setPositions(points);
      if (colors) geom.setColors(colors);
    }
    this.object.geometry = geom;
    if (this.material.dashed && points.length) this.object.computeLineDistances();
    this.object.visible = points.length > 0;
  }
}

// Round markers of a fixed pixel size.
class Markers {
  constructor(parent, ring, opts = {}) {
    this.material = new THREE.ShaderMaterial({
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      uniforms: {
        uScale: { value: window.devicePixelRatio || 1 },
        uPull: { value: opts.pull ?? 0.995 },
        uRingPx: { value: opts.ringPx ?? 2 },
        uRing: { value: color(ring) },
      },
      depthTest: opts.depthTest ?? true,
    });
    this.object = only3d(new THREE.Points(new THREE.BufferGeometry(), this.material));
    this.object.frustumCulled = false;
    this.object.visible = false;
    if (opts.order != null) this.object.renderOrder = opts.order;
    parent.add(this.object);
  }

  // Flat xyz positions, flat rgb colors (0 to 1) and one size for each point.
  set(positions, colors, sizes) {
    this.object.geometry.dispose();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('aColor', new THREE.Float32BufferAttribute(colors, 3));
    geom.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1));
    this.object.geometry = geom;
    this.object.visible = positions.length > 0;
  }
}

export class Scene3D {
  constructor(container) {
    this.container = container;
    this.ve = 1;
    this.lineMaterials = [];
    this.labels = new Map();

    this.scene = new THREE.Scene();
    this.root = new THREE.Group();
    this.scene.add(this.root);

    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 5e6);
    this.camera.up.set(0, 0, 1);
    this.camera.layers.enable(LAYER_3D);
    this.controls = new OrbitControls(this.camera, container);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;

    this.css = new CSS2DRenderer();
    this.css.domElement.style.position = 'absolute';
    this.css.domElement.style.inset = '0';
    this.css.domElement.style.pointerEvents = 'none';
    container.append(this.css.domElement);

    this.raycaster = new THREE.Raycaster();
    this.makeTerrainMaterial();
    this.makeObjects();
  }

  // ---- Objects --------------------------------------------------------------

  makeTerrainMaterial() {
    this.terrainMaterial = new THREE.ShaderMaterial({
      vertexShader: TERRAIN_VERT,
      fragmentShader: TERRAIN_FRAG,
      uniforms: {
        uRamp: { value: rampTexture(terrainLut()) },
        uSlopeRamp: { value: rampTexture(lut(ORANGES)) },
        uHeightRange: { value: new THREE.Vector2(0, 1) },
        uSun: { value: new THREE.Vector3(-0.5, 0.5, 0.7).normalize() },
        uAmbient: { value: 0.42 },
        uColorMode: { value: 0 },
        uContour: { value: 0 },
        uGraticule: { value: 0 },
        uGratOffset: { value: new THREE.Vector2() },
        uPass: { value: 0 },
        uViewshed: { value: 0 },
        uDrape: { value: 0 },
        uImage: { value: null },
        uImageSize: { value: new THREE.Vector2(1, 1) },
        uGridPx: { value: 100 },
        uDrapeOpacity: { value: 0.85 },
        uGridColor: { value: color('#ffffff') },
        uLineColor: { value: color('#ffffff') },
        uFogColor: { value: color('#c9d8e6') },
        uFogDistance: { value: 0 },
      },
    });
  }

  makeObjects() {
    const lm = this.lineMaterials;
    const r = this.root;

    // The fan of rays: thin lines, colored by the ray state, and the hits.
    this.fanGeometry = new THREE.BufferGeometry();
    this.fan = only3d(new THREE.LineSegments(
      this.fanGeometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.35, depthWrite: false }),
    ));
    this.fan.frustumCulled = false;
    r.add(this.fan);
    this.misses = new Segments(r, lm, { width: 1, dashed: true, dashSize: 6, gapSize: 6, opacity: 0.6 });
    this.fanHits = new Markers(r, '#000000', { ringPx: 0 });

    // The camera: a marker, its axes and the line down to the terrain.
    this.cameraAxes = new Segments(r, lm, { width: 3, vertexColors: true });
    this.nadir = new Segments(r, lm, { width: 1.5, dashed: true, dashSize: 5, gapSize: 5 });
    this.cameraMarker = new Markers(r, '#ffffff', { pull: 1, depthTest: false, order: 5 });

    // The frustum icon, its edges to the terrain, the footprint, and the image plane.
    this.frustum = new Segments(r, lm, { width: 2 });
    this.edges = new Segments(r, lm, { width: 1.2, dashed: true, dashSize: 8, gapSize: 6, opacity: 0.8 });
    this.footprint = new Segments(r, lm, { width: 3, pull: 0.994 });
    this.imagePlane = only3d(new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.92, depthWrite: false }),
    ));
    this.imagePlane.frustumCulled = false;
    this.imagePlane.visible = false;
    r.add(this.imagePlane);

    // The picks, the hover ray, the measured path, and the marker of a 3D click.
    this.pickRays = new Segments(r, lm, { width: 3, vertexColors: true, order: 2 });
    this.pickMarkers = new Markers(r, '#ffffff', { pull: 0.99, depthTest: false, order: 4 });
    this.path = new Segments(r, lm, { width: 3, dashed: true, dashSize: 25, gapSize: 15, depthTest: false, order: 3 });
    this.hoverRay = new Segments(r, lm, { width: 1.5 });
    this.hoverMarker = new Markers(r, '#ffffff', { pull: 0.99, order: 4 });
    this.locateMarker = new Markers(r, '#ffffff', { pull: 0.99, depthTest: false, order: 6 });
    this.locateLabel = this.label('', 'label3d info', [0.5, 1.5]);
  }

  // A label next to a point. `center` is the point on the label, in label
  // units from its top-left corner: (-0.3, 1.3) puts the label up and right.
  label(text, className, center = [-0.3, 1.3]) {
    const div = document.createElement('div');
    div.className = className;
    div.textContent = text;
    const obj = new CSS2DObject(div);
    obj.center.set(...center);
    obj.layers.set(LAYER_3D);
    obj.visible = false;
    this.root.add(obj);
    return obj;
  }

  // ---- Colors ---------------------------------------------------------------

  // Colors from the CSS tokens of the theme.
  setTheme(c) {
    this.colors = c;
    this.background = color(c.scene);
    this.misses.material.color.set(c.miss);
    this.nadir.material.color.set(c.text2);
    this.frustum.material.color.set(c.text);
    this.edges.material.color.set(c.text2);
    this.footprint.material.color.set(c.footprint);
    this.path.material.color.set(c.accent);
    this.hoverRay.material.color.set(c.hover);
    for (const m of [this.fanHits, this.cameraMarker, this.pickMarkers, this.hoverMarker, this.locateMarker]) {
      m.material.uniforms.uRing.value.set(c.scene);
    }
  }

  // ---- Terrain --------------------------------------------------------------

  // A mesh from /api/terrain, with its arrays decoded.
  setTerrain(mesh) {
    const { rows, cols } = mesh;
    const n = rows * cols;
    const pos = mesh.positions;

    // Nodata posts are NaN. Move them to a finite point, and leave them out
    // of the triangles.
    const valid = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      valid[i] = Number.isFinite(pos[3 * i]) && Number.isFinite(pos[3 * i + 2]) ? 1 : 0;
      if (!valid[i]) pos[3 * i] = pos[3 * i + 1] = pos[3 * i + 2] = 0;
    }

    // Two triangles for each cell with four valid posts.
    const index = new Uint32Array((rows - 1) * (cols - 1) * 6);
    let k = 0;
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const a = r * cols + c, b = a + 1, d = a + cols, e = d + 1;
        if (valid[a] && valid[b] && valid[d] && valid[e]) {
          index[k++] = a; index[k++] = d; index[k++] = b;
          index[k++] = b; index[k++] = d; index[k++] = e;
        }
      }
    }

    // The triangles must face up. A raster with rows from south to north
    // reverses them.
    if (k >= 3) {
      const p = (i) => new THREE.Vector3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
      const [a, b, c] = [p(index[0]), p(index[1]), p(index[2])];
      const up = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a)).z;
      if (up < 0) {
        for (let i = 0; i < k; i += 3) [index[i + 1], index[i + 2]] = [index[i + 2], index[i + 1]];
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('aHeight', new THREE.BufferAttribute(mesh.height, 1));
    geom.setAttribute('aLonLat', new THREE.BufferAttribute(mesh.lonlat, 2));
    geom.setAttribute('aPix', new THREE.BufferAttribute(new Float32Array(2 * n), 2));
    geom.setAttribute('aVis', new THREE.BufferAttribute(new Float32Array(n), 1));
    geom.setIndex(new THREE.BufferAttribute(index.subarray(0, k), 1));
    geom.computeVertexNormals();
    geom.computeBoundingSphere();

    if (this.terrain) {
      this.terrain.geometry.dispose();
      this.terrain.geometry = geom;
    } else {
      this.terrain = new THREE.Mesh(geom, this.terrainMaterial);
      this.root.add(this.terrain);
    }
    this.terrainBounds = new THREE.Box3().setFromBufferAttribute(geom.attributes.position);
    this.terrainMaterial.uniforms.uHeightRange.value.set(...mesh.height_range);
    this.mesh = mesh;
  }

  // The camera pixel and the visibility of each post, from /api/viewshed.
  setViewshed(uv, visible) {
    if (!this.terrain) return;
    const geom = this.terrain.geometry;
    const pix = geom.attributes.aPix;
    const vis = geom.attributes.aVis;
    for (let i = 0; i < vis.count; i++) {
      const ok = visible[i] && Number.isFinite(uv[2 * i]);
      pix.array[2 * i] = ok ? uv[2 * i] : 0;
      pix.array[2 * i + 1] = ok ? uv[2 * i + 1] : 0;
      vis.array[i] = ok ? 1 : 0;
    }
    pix.needsUpdate = true;
    vis.needsUpdate = true;
  }

  // The display settings of the terrain.
  setDisplay(d) {
    const u = this.terrainMaterial.uniforms;
    const az = d.sunAz * Math.PI / 180;
    const el = d.sunEl * Math.PI / 180;
    u.uSun.value.set(Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el));
    u.uColorMode.value = d.colorMode;
    u.uContour.value = d.contour;
    u.uViewshed.value = d.viewshed ? 0.45 : 0;
    u.uDrape.value = d.drape;
    u.uGridPx.value = d.gridPx;
    u.uDrapeOpacity.value = d.drape === 1 ? 0.9 : 0.85;

    // About eight graticule lines across the drawn terrain, at a round step.
    u.uGraticule.value = 0;
    if (d.graticule && this.mesh) {
      const [w, s, e, n] = this.mesh.bounds;
      const step = niceStep(Math.max(e - w, n - s) / 8);
      const { lat, lon } = this.mesh.origin;
      u.uGraticule.value = step;
      u.uGratOffset.value.set(((lon % step) + step) % step, ((lat % step) + step) % step);
    }
    this.setExaggeration(d.ve);
  }

  // The vertical exaggeration. The orbit target and the eye keep their place
  // relative to the terrain.
  setExaggeration(ve) {
    if (ve === this.ve) return;
    const k = ve / this.ve;
    this.controls.target.z *= k;
    this.camera.position.z *= k;
    this.ve = ve;
    this.root.scale.z = ve;
    this.controls.update();
  }

  setDrapeImage(texture, width, height) {
    const u = this.terrainMaterial.uniforms;
    u.uImage.value = texture;
    u.uImageSize.value.set(width, height);
  }

  // ---- Camera and rays ------------------------------------------------------

  // The response of /api/cast.
  setCast(res, opts) {
    const cam = res.camera;
    const P = cam.position;
    const c = this.colors;
    this.cast = res;
    this.iconDepth = 0.12 * res.miss_length;

    // The camera marker, its axes (red x, green y, blue z) and the nadir line.
    this.cameraMarker.set(P, [...color(c.text).toArray()], [11]);
    const axisLen = 0.6 * this.iconDepth;
    const axisColors = ['#e03131', '#2f9e44', '#1c7ed6'].map((h) => color(h).toArray());
    const ap = [], ac = [];
    cam.axes.forEach((a, i) => {
      ap.push(...P, P[0] + axisLen * a[0], P[1] + axisLen * a[1], P[2] + axisLen * a[2]);
      ac.push(...axisColors[i], ...axisColors[i]);
    });
    this.cameraAxes.set(ap, ac);
    this.nadir.set(cam.ground ? [...P, ...cam.ground] : []);

    this.setFrustum(res, opts);
    this.setFan(res.grid, P, opts);
  }

  setFrustum(res, opts) {
    const P = res.camera.position;
    const fwd = res.camera.axes[2];
    const up = res.camera.axes[1].map((x) => -x);
    const { dirs, ends, status, corners } = res.border;
    const depth = this.iconDepth;

    // The image edge on the plane at the icon depth in front of the camera.
    const rim = dirs.map((d) => {
      const s = depth / (d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2]);
      return [P[0] + s * d[0], P[1] + s * d[1], P[2] + s * d[2]];
    });
    const seg = [];
    for (let i = 0; i + 1 < rim.length; i++) seg.push(...rim[i], ...rim[i + 1]);
    for (const k of corners) seg.push(...P, ...rim[k]);

    // A triangle above the top edge marks the top of the image.
    if (corners.length === 4) {
      const tl = rim[corners[0]], tr = rim[corners[1]];
      const mid = tl.map((x, i) => (x + tr[i]) / 2);
      const half = tl.map((x, i) => 0.15 * (tr[i] - x));
      const width = Math.hypot(...tl.map((x, i) => tr[i] - x));
      const tip = mid.map((x, i) => x + 0.2 * width * up[i]);
      const a = mid.map((x, i) => x - half[i]);
      const b = mid.map((x, i) => x + half[i]);
      seg.push(...a, ...tip, ...tip, ...b, ...b, ...a);
    }
    this.frustum.set(seg);

    // The corner rays to their ends, and the footprint of the landed border.
    const edges = [];
    for (const k of corners) edges.push(...P, ...ends[k]);
    this.edges.set(edges);
    const foot = [];
    for (let i = 0; i + 1 < ends.length; i++) {
      if (status[i] && status[i + 1]) foot.push(...ends[i], ...ends[i + 1]);
    }
    this.footprint.set(foot);

    this.setImagePlane(res, opts);
  }

  // A grid mesh on the icon plane with the camera image. Distortion bends
  // its edges, as it bends the rim.
  setImagePlane(res, opts) {
    const cam = opts.camera;
    if (!opts.frustumImage || !opts.imageTexture) {
      this.imagePlane.visible = false;
      return;
    }
    const P = res.camera.position;
    const [ax, ay, az] = res.camera.axes;
    const depth = this.iconDepth;
    const nu = 16, nv = 10;
    const pos = [], uv = [], index = [];
    for (let j = 0; j <= nv; j++) {
      for (let i = 0; i <= nu; i++) {
        const u = -0.5 + cam.width * i / nu;
        const v = -0.5 + cam.height * j / nv;
        let [x, y] = pixelToNormalized(cam, u, v);
        if (!Number.isFinite(x)) [x, y] = [0, 0];
        for (let k = 0; k < 3; k++) pos.push(P[k] + depth * (x * ax[k] + y * ay[k] + az[k]));
        uv.push(i / nu, 1 - j / nv);
      }
    }
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nu; i++) {
        const a = j * (nu + 1) + i;
        index.push(a, a + nu + 1, a + 1, a + 1, a + nu + 1, a + nu + 2);
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geom.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geom.setIndex(index);
    this.imagePlane.geometry.dispose();
    this.imagePlane.geometry = geom;
    this.imagePlane.material.map = opts.imageTexture;
    this.imagePlane.material.needsUpdate = true;
    this.imagePlane.visible = true;
  }

  setFan(grid, P, opts) {
    const c = this.colors;
    const hit = color(c.ray).toArray();
    const fallback = color(c.fallback).toArray();
    const pos = [], col = [], miss = [], mpos = [], mcol = [], msize = [];
    grid.ends.forEach((e, i) => {
      const s = grid.status[i];
      if (!e) return;
      if (s === 0) {
        miss.push(...P, ...e);
        return;
      }
      const rgb = s === 1 ? hit : fallback;
      pos.push(...P, ...e);
      col.push(...rgb, ...rgb);
      mpos.push(...e);
      mcol.push(...rgb);
      msize.push(4.5);
    });
    this.fanGeometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    this.fanGeometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    this.fanGeometry.computeBoundingSphere();
    this.fan.visible = pos.length > 0;
    this.misses.set(opts.showMisses ? miss : []);
    this.fanHits.set(mpos, mcol, msize);
  }

  // The picks: their rays, their ends and their number labels.
  setPicks(picks, P, selected, measure) {
    const c = this.colors;
    const base = color(c.accent).toArray();
    const sel = color(c.selected).toArray();
    const rays = [], rayCol = [], pos = [], col = [], size = [], path = [];
    const seen = new Set();

    picks.forEach((p, i) => {
      const r = p.res;
      if (!r || !r.end) return;
      const text = String(i + 1);
      const isSel = p.id === selected;
      const rgb = isSel ? sel : base;
      rays.push(...P, ...r.end);
      rayCol.push(...rgb, ...rgb);
      if (r.status !== 'miss') {
        pos.push(...r.end);
        col.push(...rgb);
        size.push(isSel ? 15 : 12);
        path.push(r.end);
      }

      // The label sits at the end of the ray.
      let label = this.labels.get(p.id);
      if (!label) {
        label = this.label(text, 'label3d');
        this.labels.set(p.id, label);
      }
      label.element.textContent = text;
      label.element.className = `label3d${isSel ? ' selected' : ''}`;
      label.position.set(...r.end);
      label.visible = true;
      seen.add(p.id);
    });
    for (const [id, label] of this.labels) {
      if (!seen.has(id)) {
        label.removeFromParent();
        this.labels.delete(id);
      }
    }

    this.pickRays.set(rays, rayCol);
    this.pickMarkers.set(pos, col, size);
    const seg = [];
    if (measure) for (let i = 0; i + 1 < path.length; i++) seg.push(...path[i], ...path[i + 1]);
    this.path.set(seg);
  }

  setHover(P, res) {
    if (!res || !res.end) {
      this.hoverRay.set([]);
      this.hoverMarker.set([], [], []);
      return;
    }
    this.hoverRay.set([...P, ...res.end]);
    const landed = res.status !== 'miss';
    this.hoverMarker.set(landed ? res.end : [], [...color(this.colors.hover).toArray()], [9]);
  }

  // A marker and a label at a terrain point of a 3D click that the camera
  // does not see. Null hides them.
  setLocate(point, text) {
    if (!point) {
      this.locateMarker.set([], [], []);
      this.locateLabel.visible = false;
      return;
    }
    const rgb = color(this.colors.critical).toArray();
    this.locateMarker.set(point, rgb, [12]);
    this.locateLabel.element.textContent = text;
    this.locateLabel.position.set(...point);
    this.locateLabel.visible = true;
  }

  // ---- 3D view ----------------------------------------------------------------

  // The orbit target and eye position, in scene units (before the exaggeration).
  lookFrom(eye, target) {
    const s = new THREE.Vector3(1, 1, this.ve);
    this.controls.target.copy(new THREE.Vector3(...target).multiply(s));
    this.camera.position.copy(new THREE.Vector3(...eye).multiply(s));
    this.controls.update();
  }

  // A view from behind and above the camera that holds the camera and its footprint.
  fit(res, mode = 'fit') {
    const P = new THREE.Vector3(...res.camera.position);
    const pts = [P];
    for (const e of [...res.border.ends, ...res.grid.ends]) if (e) pts.push(new THREE.Vector3(...e));
    if (res.camera.ground) pts.push(new THREE.Vector3(...res.camera.ground));
    const box = new THREE.Box3().setFromPoints(pts);
    const center = box.getCenter(new THREE.Vector3());
    const size = Math.max(box.getSize(new THREE.Vector3()).length(), 200);

    // The view looks the same way as the camera. A nadir camera uses the top
    // of its image for the direction.
    const [, ay, az] = res.camera.axes;
    let look = new THREE.Vector2(az[0], az[1]);
    if (look.length() < 0.2) look = new THREE.Vector2(-ay[0], -ay[1]);
    look.normalize();

    if (mode === 'top') {
      this.lookFrom([center.x - 0.01 * look.x, center.y - 0.01 * look.y, center.z + 1.3 * size], center.toArray());
    } else if (mode === 'chase') {
      const d = 0.35 * size;
      this.lookFrom([P.x - d * look.x, P.y - d * look.y, P.z + 0.25 * d], [P.x + d * look.x, P.y + d * look.y, P.z - 0.2 * d]);
    } else {
      const d = 1.25 * size;
      this.lookFrom([center.x - 0.87 * d * look.x, center.y - 0.87 * d * look.y, center.z + 0.5 * d], center.toArray());
    }
  }

  // The ray under a screen point, in the scene frame without the exaggeration.
  rayAt(clientX, clientY) {
    const r = this.container.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.camera.aspect = r.width / r.height;
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(ndc, this.camera);
    const o = this.raycaster.ray.origin.clone();
    const d = this.raycaster.ray.direction.clone();
    o.z /= this.ve;
    d.z /= this.ve;
    d.normalize();
    return { origin: o.toArray(), direction: d.toArray() };
  }

  resize(width, height) {
    for (const m of this.lineMaterials) m.resolution.set(width, height);
    this.css.setSize(width, height);
  }

  render(renderer, rect, canvasHeight) {
    this.root.scale.z = this.ve;
    this.terrainMaterial.uniforms.uPass.value = 0;
    this.camera.aspect = rect.width / rect.height;
    this.camera.updateProjectionMatrix();

    const y = canvasHeight - rect.bottom;
    renderer.setViewport(rect.left, y, rect.width, rect.height);
    renderer.setScissor(rect.left, y, rect.width, rect.height);
    renderer.setClearColor(this.background, 1);
    renderer.clear();
    renderer.render(this.scene, this.camera);
    this.css.render(this.scene, this.camera);
  }
}

