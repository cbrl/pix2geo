// pix2geo explorer: the page state, the controls, the requests and the render loop.
//
// The page owns the camera, the pose and the picks. Each change sends the new
// state to the server, which runs pix2geo and returns the results. Pumps keep
// one request of each kind in flight, so a slider drag stays smooth.

import * as THREE from 'three';

import {
  $, $$, Pump, api, area, clamp, copyText, cssColor, debounce, download, escapeHtml,
  f32, meters, num, round, toast, u8, wrap180,
} from './util.js';
import { degreesPerMeter, hfov, withHfov, withSize } from './camera.js';
import { Scene3D } from './scene3d.js';
import { PovView } from './pov.js';
import { ProfileChart } from './chart.js';
import { layerImage, layerRequest, legendHtml } from './layers.js';
import { cliCommand, hashString, parseHash, picksCsv, picksGeojson, sessionJson } from './io.js';
import { measureHtml, pickFacts, pickHead, picksTable, profileNote, tipHtml } from './panel.js';

// The colors in the code are the screen colors. No color space conversion.
THREE.ColorManagement.enabled = false;

const DIST_NAMES = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'];
const BARREL = [-0.28, 0.09, 0, 0, -0.012, 0, 0, 0];
const EDGE_POINTS = 24;
const MESH_SIZES = [256, 512, 768, 1024];

// ---- State --------------------------------------------------------------------

const S = {
  info: null,
  start: null,
  pose: null,
  camera: null,
  fallback: { on: false, height: 0 },
  rays: { count: 256, misses: true, dots: true },
  terrain: { extent: 'all', size: 512 },
  display: {
    ve: 2, colorMode: 0, contour: 0, graticule: false, sunAz: 315, sunEl: 40,
    viewshed: false, drape: 0, gridPx: 100, frustumImage: true,
  },
  layer: { name: 'none', step: 4, opacity: 0.75, res: null, data: null },
  photo: null,
  photoOpacity: 0.85,
  showHorizon: true,
  picks: [],
  nextId: 1,
  selected: null,
  measure: false,
  cast: null,
  castMs: 0,
  hover: null,
  hoverPx: null,
  tipAt: null,
  mesh: null,
  fitted: false,
  imageDirty: true,
  overlayDirty: true,
  spin: false,
};

const view = () => ({
  camera: S.camera,
  pose: S.pose,
  fallback_height: S.fallback.on ? S.fallback.height : null,
});

const selectedPick = () => S.picks.find((p) => p.id === S.selected) ?? null;
const pickNumber = (id) => S.picks.findIndex((p) => p.id === id) + 1;

// ---- Renderer and views -------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ canvas: $('#gl'), antialias: true, logarithmicDepthBuffer: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.autoClear = false;

const body3d = $('#body3d');
const povBody = $('#bodypov');
const scene3d = new Scene3D(body3d);
const pov = new PovView(povBody, $('#overlay'), scene3d, {
  picks: () => S.picks,
  onClick: (u, v) => addPick(u, v),
  onSelect: (id) => selectPick(id),
  onDragPick: (id, u, v) => movePick(id, u, v),
  onDragEnd: () => scheduleProfile(),
  onHover: (px, x, y) => setHover(px, x, y),
  onTurn: (du, dv) => turn(du, dv),
  onFov: (factor) => setCamera(withHfov(S.camera, clamp(hfov(S.camera) * factor, 1, 170))),
  onView: () => {
    updateZoom();
    requestRender(true);
  },
});
const chart = new ProfileChart($('#profile'), $('#profileTip'));

let dirty = true;
function requestRender(overlay = false) {
  dirty = true;
  if (overlay) S.overlayDirty = true;
}

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  const r = body3d.getBoundingClientRect();
  scene3d.resize(r.width, r.height);
  requestRender(true);
}
window.addEventListener('resize', resize);
new ResizeObserver(resize).observe(body3d);
new ResizeObserver(resize).observe(povBody);
document.addEventListener('scroll', () => requestRender(true), true);
scene3d.controls.addEventListener('change', () => requestRender());

const shown = (el) => el.offsetParent !== null && el.clientWidth > 0 && el.clientHeight > 0;

// The camera image for the 3D view: the photo, or the render of the camera view.
const imageTexture = () => (S.photo ? S.photo.texture : pov.image.texture);
const needImage = () => !S.photo && (S.display.frustumImage || S.display.drape === 1);

let lastFrame = 0;
function frame(time) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (time - lastFrame) / 1000);
  lastFrame = time;
  if (S.spin) setPose({ yaw: S.pose.yaw + 12 * dt });

  const moving = scene3d.controls.update();
  if (!dirty && !moving) return;
  dirty = false;

  // The page background shows in the gaps between the panels.
  const H = window.innerHeight;
  renderer.setScissorTest(false);
  renderer.setClearColor(colors?.bg ?? '#000000', 1);
  renderer.clear();
  renderer.setScissorTest(true);
  if (needImage() && S.imageDirty) {
    pov.renderImage(renderer);
    S.imageDirty = false;
  }
  if (shown(povBody)) pov.render(renderer, H);
  if (shown(body3d)) scene3d.render(renderer, body3d.getBoundingClientRect(), H);

  if (S.overlayDirty) {
    S.overlayDirty = false;
    pov.drawOverlay({
      picks: S.picks,
      selected: S.selected,
      hover: S.hoverPx,
      grid: S.rays.dots ? S.cast?.grid : null,
      horizon: S.showHorizon,
      measure: S.measure,
      photo: S.photo?.img,
      photoOpacity: S.photoOpacity,
      layer: S.layer.name !== 'none' ? S.layer.data : null,
      layerOpacity: S.layer.opacity,
    });
  }
}

// ---- Theme ----------------------------------------------------------------------

// The CSS tokens that the canvases and WebGL use, as camelCase keys.
const THEME_TOKENS = [
  'bg', 'scene', 'panel', 'text', 'text-2', 'muted', 'grid', 'axis', 'accent', 'selected',
  'ray', 'footprint', 'fallback', 'miss', 'hover', 'terrain-line', 'critical', 'letterbox',
];
let colors = null;

function isDark() {
  const forced = document.documentElement.dataset.theme;
  return forced ? forced === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
}

function applyTheme() {
  const key = (token) => token.replace(/-(\w)/g, (_, c) => c.toUpperCase());
  colors = Object.fromEntries(THEME_TOKENS.map((t) => [key(t), cssColor(`--${t}`)]));
  scene3d.setTheme(colors);
  pov.setTheme(colors);
  chart.setTheme(colors);
  refreshScene();
  drawLayer();
  requestRender(true);
}

$('#btnTheme').addEventListener('click', () => {
  const theme = isDark() ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem('pix2geo-theme', theme);
  } catch {
    // The theme lasts only for this page.
  }
  applyTheme();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

// ---- Errors -----------------------------------------------------------------------

let lastError = { text: '', time: 0 };
function onError(err) {
  const text = err?.message ?? String(err);
  const now = performance.now();
  if (text === lastError.text && now - lastError.time < 3000) return;
  lastError = { text, time: now };
  toast(text, 'error');
}

// ---- Requests ---------------------------------------------------------------------

const castPump = new Pump(
  'cast',
  () => ({
    ...view(),
    pixels: S.picks.map((p) => [p.u, p.v]),
    ids: S.picks.map((p) => p.id),
    grid: S.rays.count,
    edge_points: EDGE_POINTS,
    detail: true,
  }),
  applyCast,
  onError,
);

const probePump = new Pump(
  'cast',
  () => (S.hoverPx ? { ...view(), pixels: [S.hoverPx], ids: [0], grid: 0, edge_points: 0, detail: false } : null),
  applyProbe,
  () => {},
);

const layerPump = new Pump(
  'image',
  () => {
    if (S.layer.name === 'none') return null;
    return { ...view(), step: S.layer.step, layers: layerRequest(S.layer.name), sun: [S.display.sunAz, S.display.sunEl] };
  },
  applyLayer,
  onError,
);

const needViewshed = () => S.display.viewshed || S.display.drape > 0;

const viewshedPump = new Pump(
  'viewshed',
  () => (needViewshed() && S.mesh ? { ...view(), mesh: S.mesh.id } : null),
  applyViewshed,
  onError,
);

const profilePump = new Pump(
  'profile',
  () => {
    const p = selectedPick();
    if (!p) return null;
    return { ...view(), pixel: [p.u, p.v], length: S.cast ? 2 * S.cast.miss_length : 5000, pick: p.id };
  },
  applyProfile,
  onError,
);

const scheduleLayer = debounce(() => layerPump.kick(), 120);
const scheduleViewshed = debounce(() => viewshedPump.kick(), 200);
const scheduleProfile = debounce(() => profilePump.kick(), 100);

// The pose, the camera or the fallback changed.
function viewChanged() {
  castPump.kick();
  if (S.hoverPx) probePump.kick();
  if (S.layer.name !== 'none') scheduleLayer();
  if (needViewshed()) scheduleViewshed();
  if (S.selected != null) scheduleProfile();
  S.imageDirty = true;
  saveHash();
  requestRender(true);
}

// ---- Cast results -------------------------------------------------------------------

// Draw the last cast again, after a change of the picks or the display options.
function refreshScene() {
  if (!S.cast || !colors) return;
  scene3d.setCast(S.cast, {
    camera: S.camera,
    showMisses: S.rays.misses,
    frustumImage: S.display.frustumImage,
    imageTexture: imageTexture(),
  });
  scene3d.setPicks(S.picks, S.cast.camera.position, S.selected, S.measure);
  scene3d.setHover(S.cast.camera.position, S.hoverPx ? S.hover : null);
  requestRender(true);
}

function applyCast(res, ms) {
  S.cast = res;
  S.castMs = ms;
  const byId = new Map(res.picks.map((r) => [r.id, r]));
  for (const p of S.picks) p.res = byId.get(p.id) ?? p.res;

  pov.setState(res.camera);
  refreshScene();
  if (!S.fitted) {
    scene3d.fit(res);
    S.fitted = true;
  }
  S.imageDirty = true;
  updatePerf();
  updateHud();
  updatePicks();
  followView();
  requestRender(true);
}

function applyProbe(res) {
  if (!S.hoverPx) return;
  S.hover = res.picks[0];
  scene3d.setHover(res.camera.position, S.hover);
  updateTip();
  requestRender();
}

// ---- Picks --------------------------------------------------------------------------

// The list of picks changed: cast them, and draw the panel and the views again.
function picksChanged() {
  castPump.kick();
  scheduleProfile();
  updatePicks();
  refreshScene();
  saveHash();
}

function addPick(u, v) {
  const p = { id: S.nextId++, u, v, res: null };
  S.picks.push(p);
  selectPick(p.id);
  picksChanged();
}

function movePick(id, u, v) {
  const p = S.picks.find((q) => q.id === id);
  if (!p) return;
  p.u = u;
  p.v = v;
  castPump.kick();
  saveHash();
  requestRender(true);
}

function removePick(id) {
  S.picks = S.picks.filter((p) => p.id !== id);
  if (S.selected === id) selectPick(S.picks.at(-1)?.id ?? null);
  picksChanged();
}

function clearPicks() {
  S.picks = [];
  selectPick(null);
  picksChanged();
}

function selectPick(id) {
  if (S.selected === id) return;
  S.selected = id;
  chart.set(null);
  $('#profileNote').hidden = true;
  updatePicks();
  refreshScene();
  scheduleProfile();
}

function stepSelection(delta) {
  if (!S.picks.length) return;
  const i = S.picks.findIndex((p) => p.id === S.selected);
  const j = (i + delta + S.picks.length) % S.picks.length;
  selectPick(S.picks[j].id);
}

// ---- Hover ----------------------------------------------------------------------------

function setHover(px, x, y) {
  S.hoverPx = px;
  S.tipAt = px ? [x, y] : null;
  if (px) {
    probePump.kick();
  } else {
    S.hover = null;
    $('#tip').hidden = true;
    if (S.cast) scene3d.setHover(S.cast.camera.position, null);
  }
  requestRender(true);
}

// The tip sits below and right of the mouse, or on the other side near an edge.
function updateTip() {
  const tip = $('#tip');
  if (!S.hover || !S.tipAt) {
    tip.hidden = true;
    return;
  }
  tip.innerHTML = tipHtml(S.hover);
  tip.hidden = false;
  const box = povBody.getBoundingClientRect();
  const x = S.tipAt[0] - box.left;
  const y = S.tipAt[1] - box.top;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${x + 16 + w < box.width ? x + 16 : Math.max(4, x - 16 - w)}px`;
  tip.style.top = `${y + 16 + h < box.height ? y + 16 : Math.max(4, y - 16 - h)}px`;
}

// ---- Pose and camera ----------------------------------------------------------------

function setPose(patch) {
  const p = Object.assign(S.pose, patch);
  p.yaw = wrap180(p.yaw);
  p.roll = wrap180(p.roll);
  p.pitch = clamp(p.pitch, -90, 90);
  p.lat = clamp(p.lat, -89.9999, 89.9999);
  p.lon = wrap180(p.lon);
  syncInputs();
  viewChanged();
}

function setCamera(cam) {
  S.camera = cam;
  pov.setCamera(cam);
  updateDrapeImage();
  syncInputs();
  updateZoom();
  viewChanged();
}

// Change the image size. The picks keep their place in the image.
function resizeCamera(width, height, base = S.camera) {
  if (!(width >= 16 && height >= 16)) return;
  const old = S.camera;
  for (const p of S.picks) {
    p.u = ((p.u + 0.5) * width) / old.width - 0.5;
    p.v = ((p.v + 0.5) * height) / old.height - 0.5;
  }
  setCamera(withSize(base, width, height));
}

// Turn the camera by a drag of (du, dv) image pixels. The scene follows the mouse.
function turn(du, dv) {
  const k = 180 / Math.PI;
  setPose({ yaw: S.pose.yaw - (du / S.camera.fx) * k, pitch: S.pose.pitch + (dv / S.camera.fy) * k });
}

// Move the camera by (forward, right) meters along its heading, and up.
function move(forward, right, up) {
  const yaw = (S.pose.yaw * Math.PI) / 180;
  const north = forward * Math.cos(yaw) - right * Math.sin(yaw);
  const east = forward * Math.sin(yaw) + right * Math.cos(yaw);
  const [dLat, dLon] = degreesPerMeter(S.pose.lat);
  setPose({ lat: S.pose.lat + north * dLat, lon: S.pose.lon + east * dLon, alt: S.pose.alt + up });
}

// A step in meters for the keyboard: a share of the height above the terrain.
function moveStep() {
  const agl = S.cast?.camera.agl;
  return clamp(0.05 * Math.abs(agl ?? 1000), 5, 5000);
}

// The altitude in a new reference, so the camera stays in place.
function convertAltitude(ref) {
  const c = S.cast?.camera;
  if (!c) return S.pose.alt;
  return { ellipsoid: c.alt_ellipsoid, geoid: c.alt_msl, agl: c.agl }[ref] ?? S.pose.alt;
}

function toggleSpin() {
  S.spin = !S.spin;
  $('#btnSpin').classList.toggle('on', S.spin);
}

// ---- Inputs -------------------------------------------------------------------------

const syncers = [];
const syncInputs = () => syncers.forEach((f) => f());

// A range input with a number box. `get` reads the state and `set` writes it.
// The CSS variable --fill colors the track up to the thumb.
function slider(id, get, set, digits = 1) {
  const range = $(`#${id}`);
  const box = $(`[data-for="${id}"]`);
  const fill = () => {
    const share = (range.value - range.min) / (range.max - range.min);
    range.style.setProperty('--fill', `${(100 * share).toFixed(1)}%`);
  };
  range.addEventListener('input', () => {
    box.value = range.value;
    fill();
    set(Number(range.value));
  });
  box.addEventListener('change', () => {
    const x = Number(box.value);
    if (Number.isFinite(x)) set(x);
  });
  syncers.push(() => {
    const x = get();
    if (document.activeElement !== range) range.value = x;
    if (document.activeElement !== box) box.value = round(x, digits);
    fill();
  });
}

// A checkbox, a select or a number box.
function field(id, get, set, digits = 6) {
  const el = $(`#${id}`);
  el.addEventListener('change', () => {
    const x = el.type === 'checkbox' ? el.checked : el.type === 'number' ? Number(el.value) : el.value;
    if (el.type === 'number' && (el.value === '' || !Number.isFinite(x))) return;
    set(x);
  });
  syncers.push(() => {
    if (document.activeElement === el) return;
    const x = get();
    if (el.type === 'checkbox') el.checked = !!x;
    else if (el.type === 'number') el.value = round(x, digits);
    else el.value = String(x);
  });
}

function bindInputs() {
  // Pose.
  field('lat', () => S.pose.lat, (x) => setPose({ lat: x }), 7);
  field('lon', () => S.pose.lon, (x) => setPose({ lon: x }), 7);
  field('alt', () => S.pose.alt, (x) => setPose({ alt: x }), 2);
  field('altRef', () => S.pose.alt_ref, (x) => setPose({ alt: convertAltitude(x), alt_ref: x }));
  slider('yaw', () => S.pose.yaw, (x) => setPose({ yaw: x }));
  slider('pitch', () => S.pose.pitch, (x) => setPose({ pitch: x }));
  slider('roll', () => S.pose.roll, (x) => setPose({ roll: x }));
  $('#btnNadir').addEventListener('click', () => setPose({ pitch: -90, roll: 0 }));
  $('#btnLevel').addEventListener('click', () => setPose({ pitch: 0, roll: 0 }));
  $('#btnSpin').addEventListener('click', toggleSpin);
  $('#btnResetPose').addEventListener('click', () => setPose({ ...S.start.pose }));

  // Camera: the size, the field of view, K and the distortion.
  field('camW', () => S.camera.width, (x) => resizeCamera(Math.round(x), S.camera.height), 0);
  field('camH', () => S.camera.height, (x) => resizeCamera(S.camera.width, Math.round(x)), 0);
  slider('hfov', () => hfov(S.camera), (x) => setCamera(withHfov(S.camera, clamp(x, 1, 170))));
  for (const key of ['fx', 'fy', 'cx', 'cy']) {
    field(key, () => S.camera[key], (x) => {
      if (key[0] === 'f' && !(x > 0)) syncInputs();
      else setCamera({ ...S.camera, [key]: x });
    }, 2);
  }
  DIST_NAMES.forEach((name, i) => {
    const label = document.createElement('label');
    label.className = 'field';
    label.innerHTML = `<span>${name}</span><input type="number" id="dist${i}" step="0.01">`;
    $('#distInputs').append(label);
    field(`dist${i}`, () => S.camera.dist[i], (x) => {
      const dist = [...S.camera.dist];
      dist[i] = x;
      setCamera({ ...S.camera, dist });
    });
  });
  $('#btnDistZero').addEventListener('click', () => setCamera({ ...S.camera, dist: new Array(8).fill(0) }));
  $('#btnDistBarrel').addEventListener('click', () => setCamera({ ...S.camera, dist: [...BARREL] }));
  $('#btnResetCam').addEventListener('click', () => {
    resizeCamera(S.start.camera.width, S.start.camera.height, { ...S.start.camera });
  });

  // Rays.
  slider('rayCount', () => S.rays.count, (x) => {
    S.rays.count = Math.round(clamp(x, 0, 5000));
    castPump.kick();
  }, 0);
  field('showMisses', () => S.rays.misses, (x) => {
    S.rays.misses = x;
    refreshScene();
  });
  field('showDots', () => S.rays.dots, (x) => {
    S.rays.dots = x;
    requestRender(true);
  });
  field('fallbackOn', () => S.fallback.on, (x) => {
    S.fallback.on = x;
    viewChanged();
  });
  field('fallbackH', () => S.fallback.height, (x) => {
    S.fallback.height = x;
    if (S.fallback.on) viewChanged();
  }, 1);

  // Terrain.
  field('extent', () => S.terrain.extent, (x) => {
    S.terrain.extent = x;
    loadTerrain();
  });
  field('meshSize', () => S.terrain.size, (x) => {
    S.terrain.size = Number(x);
    loadTerrain();
  });
  $('#btnReloadTerrain').addEventListener('click', () => loadTerrain());
  const display = (key, cast = (x) => x) => (x) => {
    S.display[key] = cast(x);
    scene3d.setDisplay(S.display);
    S.imageDirty = true;
    syncInputs();
    requestRender();
    saveHash();
  };
  slider('ve', () => S.display.ve, display('ve', (x) => clamp(x, 0.1, 50)));
  field('colorMode', () => S.display.colorMode, display('colorMode', Number));
  field('contour', () => S.display.contour, display('contour', Number));
  field('graticule', () => S.display.graticule, display('graticule'));
  const sun = (key) => (x) => {
    display(key)(x);
    if (S.layer.name === 'render') scheduleLayer();
  };
  slider('sunAz', () => S.display.sunAz, sun('sunAz'), 0);
  slider('sunEl', () => S.display.sunEl, sun('sunEl'), 0);

  // Camera view: the pix2geo layer, the photo and the horizon.
  field('layer', () => S.layer.name, (x) => {
    S.layer.name = x;
    S.layer.res = null;
    S.layer.data = null;
    $('#layerInfo').textContent = '';
    if (x !== 'none') layerPump.kick();
    syncInputs();
    updateLegend();
    requestRender(true);
    saveHash();
  });
  field('layerStep', () => S.layer.step, (x) => {
    S.layer.step = Number(x);
    layerPump.kick();
  });
  slider('layerOpacity', () => S.layer.opacity, (x) => {
    S.layer.opacity = clamp(x, 0, 1);
    requestRender(true);
  }, 2);
  slider('photoOpacity', () => S.photoOpacity, (x) => {
    S.photoOpacity = clamp(x, 0, 1);
    requestRender(true);
  }, 2);
  field('showHorizon', () => S.showHorizon, (x) => {
    S.showHorizon = x;
    requestRender(true);
  });
  $('#btnPhoto').addEventListener('click', () => $('#fileInput').click());
  $('#btnPhotoClear').addEventListener('click', () => setPhoto(null));
  $('#btnPhotoSize').addEventListener('click', () => {
    if (S.photo) resizeCamera(S.photo.img.naturalWidth, S.photo.img.naturalHeight);
  });

  // Coverage on the terrain.
  const coverage = (key, cast) => (x) => {
    display(key, cast)(x);
    if (needViewshed()) scheduleViewshed();
    else $('#coverageInfo').textContent = '';
    updateDrapeImage();
    refreshScene();
  };
  field('viewshed', () => S.display.viewshed, coverage('viewshed'));
  field('drape', () => S.display.drape, coverage('drape', Number));
  field('gridPx', () => S.display.gridPx, coverage('gridPx', Number));
  field('frustumImage', () => S.display.frustumImage, coverage('frustumImage'));

  // Inputs that do nothing in the current state.
  syncers.push(() => {
    const noLayer = S.layer.name === 'none';
    for (const el of [$('#layerStep'), $('#layerOpacity'), $('[data-for="layerOpacity"]')]) el.disabled = noLayer;
    $('#gridPx').disabled = S.display.drape !== 2;
    $('#fallbackH').disabled = !S.fallback.on;
    $('#btnMeasure').classList.toggle('on', S.measure);
  });

  // Engine.
  const settings = async (patch) => {
    try {
      S.info.settings = await api('settings', patch);
      castPump.kick();
      if (S.layer.name !== 'none') scheduleLayer();
    } catch (err) {
      onError(err);
    }
    syncInputs();
  };
  field('backend', () => S.info.settings.backend, (x) => settings({ backend: x }));
  field('segment', () => S.info.settings.max_segment_length, (x) => settings({ max_segment_length: x }), 1);
  const rangeBox = $('#maxRange');
  rangeBox.addEventListener('change', () => {
    const km = rangeBox.value === '' ? 0 : Number(rangeBox.value);
    settings({ max_range: km > 0 ? km * 1000 : null });
  });
  syncers.push(() => {
    if (document.activeElement === rangeBox) return;
    const r = S.info.settings.max_range;
    rangeBox.value = r ? round(r / 1000, 3) : '';
  });
  $('#btnBench').addEventListener('click', benchmark);

  // Session.
  $('#btnSave').addEventListener('click', () => {
    download('pix2geo-session.json', JSON.stringify(sessionJson(S), null, 2));
  });
  $('#btnLoad').addEventListener('click', () => $('#fileInput').click());
  $('#fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (file) openFile(file);
  });

  // Picks panel.
  $('#btnMeasure').addEventListener('click', () => {
    S.measure = !S.measure;
    syncInputs();
    updatePicks();
    refreshScene();
  });
  $('#btnCsv').addEventListener('click', () => download('pix2geo-picks.csv', picksCsv(S), 'text/csv'));
  $('#btnGeojson').addEventListener('click', () => {
    download('pix2geo-picks.geojson', picksGeojson(S), 'application/geo+json');
  });
  $('#btnClear').addEventListener('click', clearPicks);

  // Top bar and view heads.
  $('#btnLink').addEventListener('click', () => {
    saveHashNow();
    copyText(location.href);
    toast('Link copied.');
  });
  $('#btnCli').addEventListener('click', () => {
    copyText(cliCommand(S));
    toast(S.info.cli.dem.length ? 'Command copied.' : 'Command copied. Put the DEM path in place of DEM.tif.');
  });
  $('#btnHelp').addEventListener('click', () => $('#helpDlg').showModal());
  $('#btnZoomFit').addEventListener('click', () => {
    pov.fit();
    updateZoom();
    requestRender(true);
  });
  $$('[data-view3d]').forEach((b) => b.addEventListener('click', () => {
    if (S.cast) scene3d.fit(S.cast, b.dataset.view3d);
  }));
  $$('[data-max]').forEach((b) => b.addEventListener('click', () => {
    const cls = `max-${b.dataset.max}`;
    const on = !document.body.classList.contains(cls);
    document.body.classList.remove('max-3d', 'max-pov');
    if (on) document.body.classList.add(cls);
    for (const x of $$('[data-max]')) x.classList.toggle('on', document.body.classList.contains(`max-${x.dataset.max}`));
    resize();
  }));
}

// ---- Keyboard -----------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
  if (e.target.closest?.('input, select, textarea, dialog') || e.ctrlKey || e.metaKey || e.altKey) return;
  if (!S.pose || (e.key === ' ' && e.target.closest?.('button'))) return;
  const step = moveStep() * (e.shiftKey ? 5 : 1);
  const angle = e.shiftKey ? 10 : 2;
  const actions = {
    w: () => move(step, 0, 0),
    s: () => move(-step, 0, 0),
    a: () => move(0, -step, 0),
    d: () => move(0, step, 0),
    r: () => move(0, 0, step),
    f: () => move(0, 0, -step),
    arrowleft: () => setPose({ yaw: S.pose.yaw - angle }),
    arrowright: () => setPose({ yaw: S.pose.yaw + angle }),
    arrowup: () => setPose({ pitch: S.pose.pitch + angle }),
    arrowdown: () => setPose({ pitch: S.pose.pitch - angle }),
    q: () => setPose({ roll: S.pose.roll - angle }),
    e: () => setPose({ roll: S.pose.roll + angle }),
    ' ': toggleSpin,
    delete: () => S.selected != null && removePick(S.selected),
    backspace: () => S.selected != null && removePick(S.selected),
    '[': () => stepSelection(-1),
    ']': () => stepSelection(1),
    '?': () => $('#helpDlg').showModal(),
    escape: () => selectPick(null),
  };
  const action = actions[e.key.toLowerCase()];
  if (action) {
    e.preventDefault();
    action();
  }
});

// ---- 3D clicks ------------------------------------------------------------------------

// A click on the 3D terrain finds the point in the camera. A visible point
// becomes a pick. Alt aims the camera at the point, and Shift moves the
// camera above it.
let down3d = null;
let locateTimer = 0;
body3d.addEventListener('pointerdown', (e) => {
  down3d = e.button === 0 ? { x: e.clientX, y: e.clientY, t: performance.now() } : null;
});
body3d.addEventListener('pointerup', async (e) => {
  const d = down3d;
  down3d = null;
  if (!d || e.button !== 0 || !S.camera) return;
  if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4 || performance.now() - d.t > 600) return;
  const mode = e.altKey ? 'aim' : e.shiftKey ? 'move' : 'locate';
  try {
    const res = await api('locate', { ...view(), ...scene3d.rayAt(e.clientX, e.clientY) });
    await onLocate(res, mode);
  } catch (err) {
    onError(err);
  }
});

async function onLocate(res, mode) {
  clearTimeout(locateTimer);
  scene3d.setLocate(null);
  requestRender();
  if (!res.hit) return;

  if (mode === 'aim') {
    setPose(await api('look_at', { ...view(), target: [res.lat, res.lon, res.alt] }));
  } else if (mode === 'move') {
    setPose({ lat: res.lat, lon: res.lon });
  } else if (res.visible) {
    addPick(res.pixel[0], res.pixel[1]);
  } else {
    const why = !res.in_front ? 'behind the camera' : !res.in_image ? 'outside the image' : 'hidden by the terrain';
    scene3d.setLocate(res.enu, `Not in view: ${why}`);
    locateTimer = setTimeout(() => {
      scene3d.setLocate(null);
      requestRender();
    }, 4000);
  }
}

// ---- Terrain, layers and coverage -------------------------------------------------------

let loadingTerrain = false;
async function loadTerrain() {
  loadingTerrain = true;
  try {
    $('#meshInfo').textContent = 'Loading…';
    const res = await api('terrain', { ...view(), extent: S.terrain.extent, size: S.terrain.size });
    S.mesh = {
      ...res,
      positions: f32(res.positions),
      height: f32(res.height),
      lonlat: f32(res.lonlat),
      origin: S.info.origin,
    };
    scene3d.setTerrain(S.mesh);
    scene3d.setDisplay(S.display);
    if (needViewshed()) scheduleViewshed();
    $('#meshInfo').textContent = `${res.rows} × ${res.cols} posts, ${meters(res.spacing)} apart`;
    S.imageDirty = true;
    requestRender(true);
  } catch (err) {
    onError(err);
  } finally {
    loadingTerrain = false;
  }
}

// True when a border ray hits the terrain outside the drawn terrain.
function footprintOutside() {
  const m = S.mesh, b = S.cast?.border;
  if (!m || !b) return false;
  const [w, s, e, n] = m.bounds;
  return b.lonlat.some(([lon, lat], i) => b.status[i] === 1 && (lon < w || lon > e || lat < s || lat > n));
}

// Around the view, the drawn terrain follows the footprint of the camera.
const followView = debounce(() => {
  if (S.terrain.extent === 'view' && !loadingTerrain && footprintOutside()) loadTerrain();
}, 600);

function applyViewshed(res) {
  if (!S.mesh || res.mesh !== S.mesh.id) return;
  scene3d.setViewshed(f32(res.uv), u8(res.visible));
  $('#coverageInfo').textContent = `The camera sees ${area(res.area)} of terrain.`;
  requestRender();
}

function updateDrapeImage() {
  const texture = S.display.drape === 1 && S.photo ? S.photo.texture : pov.image.texture;
  scene3d.setDrapeImage(texture, S.camera.width, S.camera.height);
  S.imageDirty = true;
}

function applyLayer(res, ms, req) {
  // A response for a layer that the page no longer shows does not count.
  if (S.layer.name === 'none' || req.layers.join() !== layerRequest(S.layer.name).join()) return;
  S.layer.res = res;
  drawLayer();
  $('#layerInfo').textContent = `${num(res.rays, 0)} rays in ${num(res.ms, 0)} ms`;
}

// The layer image from the last response, in the colors of the theme.
function drawLayer() {
  if (S.layer.res && colors) {
    S.layer.data = layerImage(S.layer.name, S.layer.res, colors, S.mesh?.height_range);
  }
  updateLegend();
  requestRender(true);
}

function updateLegend() {
  const box = $('#legend');
  box.hidden = S.layer.name === 'none' || !colors;
  if (!box.hidden) box.innerHTML = legendHtml(S.layer.name, S.layer.data?.scale, colors);
}

// ---- Photo and files ----------------------------------------------------------------------

async function setPhoto(file) {
  if (S.photo) URL.revokeObjectURL(S.photo.img.src);
  S.photo = null;
  if (file) {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    try {
      await img.decode();
    } catch {
      onError(new Error(`Cannot read the image ${file.name}.`));
      return;
    }
    const texture = new THREE.Texture(img);
    texture.needsUpdate = true;
    S.photo = { img, texture, name: file.name };
    const { naturalWidth: w, naturalHeight: h } = img;
    if (w !== S.camera.width || h !== S.camera.height) {
      toast(`The photo is ${w} × ${h} px and the camera ${S.camera.width} × ${S.camera.height} px, so the photo is stretched.`, 'warning', 7000);
    }
  }
  for (const id of ['#btnPhotoSize', '#btnPhotoClear', '#photoRow']) $(id).hidden = !S.photo;
  updateDrapeImage();
  refreshScene();
}

async function openFile(file) {
  if (file.type.startsWith('image/')) {
    setPhoto(file);
  } else if (file.name.endsWith('.json') || file.type === 'application/json') {
    try {
      applySession(JSON.parse(await file.text()));
      toast(`Opened ${file.name}.`);
    } catch (err) {
      onError(new Error(`Cannot open ${file.name}: ${err.message}`));
    }
  } else {
    toast(`Cannot open ${file.name}. Use an image or a session file.`, 'warning');
  }
}

let dragDepth = 0;
const showDrop = (on) => {
  $('#dropHint').hidden = !on;
};
povBody.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth++;
  showDrop(true);
});
povBody.addEventListener('dragover', (e) => e.preventDefault());
povBody.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) showDrop(false);
});
povBody.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  showDrop(false);
  const file = e.dataTransfer.files[0];
  if (file) openFile(file);
});
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => e.preventDefault());

// ---- Panels -------------------------------------------------------------------------------

function updatePerf() {
  const r = S.cast;
  const perf = $('#perf');
  perf.textContent = `${num(r.rays, 0)} rays · ${num(r.ms, 1)} ms · ${r.backend.replace('auto: ', '')}`;
  perf.title = `The time of the last ray cast on the server. The round trip took ${num(S.castMs, 0)} ms.`;
}

// The camera-view summary: the values that the controls do not show.
function updateHud() {
  const c = S.cast.camera;
  const gsd = S.cast.center?.gsd;
  const parts = [`AGL ${meters(c.agl)}`];
  if (gsd) parts.push(`GSD ${num(gsd[0], 2)} × ${num(gsd[1], 2)} m`);
  if (S.cast.border.area != null) parts.push(`footprint ${area(S.cast.border.area)}`);
  const hud = $('#hud');
  hud.textContent = parts.join(' · ');
  hud.classList.toggle('warn', c.agl != null && c.agl < 0);
}

function updateZoom() {
  const zoomed = pov.zoom > 1.001;
  $('#zoomInfo').hidden = !zoomed;
  $('#btnZoomFit').hidden = !zoomed;
  $('#zoomInfo').textContent = `${num(pov.zoom, 1)}×`;
}

function updatePicks() {
  const n = S.picks.length;
  $('#pickCount').textContent = n ? String(n) : '';
  $('#pickTable').innerHTML = n
    ? picksTable(S.picks, S.selected, S.info.terrain.geoid ? 'Height MSL (m)' : 'Height (m)')
    : '<p class="empty">Click the camera view to pick a pixel and cast its ray.</p>';
  $('#measureInfo').innerHTML = S.measure && n ? measureHtml(S.cast?.measure, S.picks) : '';

  const p = selectedPick();
  $('#pickDetail').hidden = !p;
  if (p) {
    $('#pickHead').innerHTML = pickHead(p, pickNumber(p.id));
    $('#pickFacts').innerHTML = pickFacts(p.res);
  }
}

$('#pickTable').addEventListener('click', (e) => {
  const del = e.target.closest('[data-del]');
  if (del) {
    removePick(Number(del.dataset.del));
    return;
  }
  const row = e.target.closest('tr[data-id]');
  if (row) selectPick(Number(row.dataset.id));
});

$('#pickHead').addEventListener('click', async (e) => {
  const act = e.target.closest('[data-act]')?.dataset.act;
  const r = selectedPick()?.res;
  if (!act || !r) return;
  if (act === 'copy') {
    copyText(`${r.lat.toFixed(8)}, ${r.lon.toFixed(8)}`);
    toast('Latitude and longitude copied.');
  } else if (act === 'aim') {
    try {
      setPose(await api('look_at', { ...view(), target: [r.lat, r.lon, r.alt] }));
    } catch (err) {
      onError(err);
    }
  }
});

function applyProfile(res, ms, req) {
  if (req.pick !== S.selected) return;
  chart.set(res);
  const note = profileNote(res);
  $('#profileNote').textContent = note;
  $('#profileNote').hidden = !note;
}

// ---- Sessions and links ---------------------------------------------------------------------

function applySession(j, { quiet = false } = {}) {
  if (j.pose) S.pose = { ...S.pose, ...j.pose };
  if (j.camera) {
    S.camera = { ...S.camera, ...j.camera };
    pov.setCamera(S.camera);
  }
  if (j.fallback) S.fallback = { ...S.fallback, ...j.fallback };
  if (j.rays) S.rays = { ...S.rays, ...j.rays };
  if (j.display) S.display = { ...S.display, ...j.display };
  if (j.layer) S.layer = { ...S.layer, ...j.layer, res: null, data: null };
  if (j.showHorizon != null) S.showHorizon = j.showHorizon;
  if (j.measure != null) S.measure = j.measure;
  if (j.picks) {
    S.picks = j.picks.map(([u, v]) => ({ id: S.nextId++, u, v, res: null }));
    S.selected = S.picks.at(-1)?.id ?? null;
  }
  if (quiet) return;
  S.fitted = false;
  chart.set(null);
  scene3d.setDisplay(S.display);
  syncInputs();
  updatePicks();
  updateLegend();
  if (S.layer.name !== 'none') layerPump.kick();
  viewChanged();
}

function saveHashNow() {
  if (S.pose && S.camera) history.replaceState(null, '', `#${hashString(S)}`);
}
const saveHash = debounce(saveHashNow, 400);

function readHash() {
  if (!location.hash) return;
  try {
    applySession(parseHash(location.hash), { quiet: true });
  } catch (err) {
    toast(`Cannot read the link: ${err.message}`, 'warning');
  }
}

// ---- Benchmark ------------------------------------------------------------------------------

const rate = (x) => (x >= 1e6 ? `${num(x / 1e6, 2)} M/s` : `${num(x / 1e3, 1)} k/s`);

async function benchmark() {
  const dlg = $('#benchDlg');
  $('#benchBody').innerHTML = '<p class="muted">Running… The first run of a backend can compile its kernel.</p>';
  dlg.showModal();
  try {
    const res = await api('benchmark', view());
    const best = Math.max(...res.results.map((r) => r.rays_per_second));
    const rows = res.results.map((r) => `<tr>
      <td>${r.backend}</td><td>${num(r.rays, 0)}${r.step > 1 ? ` <span class="muted">(step ${r.step})</span>` : ''}</td>
      <td>${num(r.seconds * 1000, 1)} ms</td><td>${rate(r.rays_per_second)}</td>
      <td class="barcell"><div class="bar" style="width:${(100 * r.rays_per_second) / best}%"></div></td>
      <td>${num(100 * r.hits, 1)} %</td></tr>`);
    $('#benchBody').innerHTML = `<table class="bench"><thead><tr><th>Backend</th><th>Rays</th><th>Time</th>
      <th>Rate</th><th></th><th>Hits</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
  } catch (err) {
    $('#benchBody').innerHTML = `<p>${escapeHtml(err.message)}</p>`;
  }
}

// ---- Start ----------------------------------------------------------------------------------

async function init() {
  let info;
  try {
    info = await api('session');
  } catch (err) {
    $('#terrainInfo').textContent = `Cannot reach the pix2geo server: ${err.message}`;
    return;
  }
  S.info = info;
  S.start = { pose: { ...info.pose }, camera: { ...info.camera, dist: [...info.camera.dist] } };
  S.pose = { ...info.pose };
  S.camera = { ...info.camera, dist: [...info.camera.dist] };
  S.fallback = { on: info.fallback_height != null, height: Math.round(info.fallback_height ?? info.terrain.h_min) };
  S.rays.count = info.max_rays;

  // The nearest mesh size, and the whole DEM when it is not much larger.
  const near = (a, b) => (Math.abs(b - info.terrain_size) < Math.abs(a - info.terrain_size) ? b : a);
  S.terrain.size = MESH_SIZES.reduce(near);
  S.terrain.extent = Math.max(...info.terrain.shape) <= 2 * S.terrain.size ? 'all' : 'view';

  for (const name of info.backends) $('#backend').add(new Option(name, name));
  if (!info.terrain.geoid) $('#altRef option[value="geoid"]').disabled = true;
  const t = info.terrain;
  $('#terrainInfo').textContent = [
    `${t.shape[0]} × ${t.shape[1]} posts`,
    t.crs,
    `${num(t.h_min, 0)} to ${num(t.h_max, 0)} m`,
    t.geoid ? `geoid ${t.geoid}` : '',
  ].filter(Boolean).join(' · ');
  $('#terrainInfo').title = t.summary;

  readHash();
  pov.setCamera(S.camera);
  bindInputs();
  syncInputs();
  applyTheme();
  updatePicks();
  resize();

  await loadTerrain();
  viewChanged();
  if (S.layer.name !== 'none') layerPump.kick();
  requestAnimationFrame(frame);
}

// A handle for the browser console and for tests.
window.__pix2geo = {
  S, scene3d, pov, renderer, api, addPick, setPose, setCamera, openFile,
  cliCommand: () => cliCommand(S),
  sessionJson: () => sessionJson(S),
};
init();
