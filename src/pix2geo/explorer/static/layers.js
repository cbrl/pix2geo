// The pix2geo layers of the camera view: their colors, their images and
// their legends. The server sends one value for each cell of `step` pixels.

import { BLUES, ORANGES, gradient, hexRgb, lut, terrainGradient, terrainLut } from './colormaps.js';
import { clamp, escapeHtml, f32, num, u8 } from './util.js';

// Log scales suit the values that span decades.
export const LAYERS = {
  render: { title: 'Ray-cast render' },
  range: { title: 'Slant range', ramp: BLUES, unit: 'm' },
  height: { title: 'Hit height', ramp: null, unit: 'm' },
  gsd: { title: 'Ground sample distance', ramp: BLUES, unit: 'm/px', log: true },
  sensitivity: { title: 'Hit shift for 1 m of DEM error', ramp: ORANGES, unit: 'm', log: true },
  status: { title: 'Misses and fallback hits' },
};

// The layers that the server must compute for a layer of the page.
export const layerRequest = (name) => (name === 'render' ? ['height', 'shade'] : [name]);

// An image of one layer from /api/image, one canvas pixel for each cell.
// `heightRange` is the height range of the drawn terrain, for the render.
export function layerImage(name, res, colors, heightRange) {
  const { rows, cols } = res;
  const canvas = document.createElement('canvas');
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  const px = img.data;
  let scale = null;

  if (name === 'status') {
    const status = u8(res.layers.status.data);
    const miss = hexRgb(colors.miss), fallback = hexRgb(colors.fallback);
    for (let i = 0; i < status.length; i++) {
      const rgb = status[i] === 0 ? miss : status[i] === 2 ? fallback : null;
      if (rgb) px.set([...rgb, 220], 4 * i);
    }
  } else if (name === 'render') {
    // The hit height in the terrain colors, lit by the sun.
    const height = f32(res.layers.height.data);
    const shade = f32(res.layers.shade.data);
    const table = terrainLut();
    const [lo, hi] = heightRange ?? [res.layers.height.min, res.layers.height.max];
    for (let i = 0; i < height.length; i++) {
      if (!Number.isFinite(height[i])) continue;
      const k = 4 * Math.round(255 * clamp((height[i] - lo) / Math.max(hi - lo, 1e-3), 0, 1));
      const light = 0.42 + 0.58 * (Number.isFinite(shade[i]) ? shade[i] : 0.5);
      px.set([table[k] * light, table[k + 1] * light, table[k + 2] * light, 255], 4 * i);
    }
  } else {
    // The 2nd to 98th percentile of the values spans the ramp.
    const layer = res.layers[name];
    const values = f32(layer.data);
    const spec = LAYERS[name];
    const table = spec.ramp ? lut(spec.ramp) : terrainLut();
    let lo = layer.low ?? 0, hi = layer.high ?? 1;
    if (spec.log) {
      lo = Math.max(lo, 1e-3);
      hi = Math.max(hi, lo * 1.01);
    } else if (hi <= lo) {
      hi = lo + 1;
    }
    const f = spec.log
      ? (x) => (Math.log(x) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))
      : (x) => (x - lo) / (hi - lo);
    for (let i = 0; i < values.length; i++) {
      if (!Number.isFinite(values[i])) continue;
      const k = 4 * Math.round(255 * clamp(f(values[i]), 0, 1));
      px.set([table[k], table[k + 1], table[k + 2], 255], 4 * i);
    }
    scale = { lo, hi, unit: spec.unit, log: spec.log, ramp: spec.ramp };
  }

  ctx.putImageData(img, 0, 0);
  return { canvas, rows, cols, step: res.step, scale };
}

// The HTML of the legend of a layer. `scale` comes from layerImage.
export function legendHtml(name, scale, colors) {
  const spec = LAYERS[name];
  const fmt = (x) => num(x, Math.abs(x) >= 100 ? 0 : Math.abs(x) >= 10 ? 1 : 2);
  const swatch = (color, text) => `<span class="swatch" style="background:${color}"></span>${text}`;
  let html = `<div>${escapeHtml(spec.title)}</div>`;

  if (name === 'status') {
    html += `<div class="keys">${swatch(colors.miss, 'miss')}${swatch(colors.fallback, 'fallback')}</div>`;
  } else if (name === 'render') {
    html += `<div class="bar" style="background:${terrainGradient()}"></div>`;
  } else if (scale) {
    const mid = scale.log ? Math.sqrt(scale.lo * scale.hi) : (scale.lo + scale.hi) / 2;
    const bg = scale.ramp ? gradient(scale.ramp) : terrainGradient();
    html += `<div class="bar" style="background:${bg}"></div>`;
    html += `<div class="ticks"><span>${fmt(scale.lo)}</span><span>${fmt(mid)}</span><span>${fmt(scale.hi)} ${scale.unit}</span></div>`;
    if (scale.log) html += '<div class="muted">log scale</div>';
  }
  return html;
}
