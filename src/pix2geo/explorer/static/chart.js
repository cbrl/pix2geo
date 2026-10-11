// The terrain profile along the ray of the selected pick: the ray height and
// the terrain height against the distance. It has a legend, a hit marker, the
// closest earlier approach, and a crosshair with a tooltip. The ray has the
// color of the selected pick.

import { escapeHtml, meters, niceStep, num } from './util.js';

const PAD = { left: 54, right: 12, top: 24, bottom: 30 };

const clampIndex = (i, hi) => Math.max(0, Math.min(hi, i));

function ticks(lo, hi, count) {
  const step = niceStep((hi - lo) / count);
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(t);
  return out;
}

export class ProfileChart {
  constructor(canvas, tip) {
    this.canvas = canvas;
    this.tip = tip;
    this.data = null;
    this.hoverIndex = null;
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerleave', () => {
      this.hoverIndex = null;
      this.tip.hidden = true;
      this.draw();
    });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  setTheme(c) {
    this.colors = c;
    this.draw();
  }

  // The response of /api/profile.
  set(data) {
    this.data = data;
    this.hoverIndex = null;
    this.draw();
  }

  // The x values: the horizontal distance, or the slant range for a steep ray.
  layout() {
    const d = this.data;
    const rect = this.canvas.getBoundingClientRect();
    const n = d.t.length;
    const groundSpan = d.ground[n - 1] ?? 0;
    const heights = [...d.ray, ...d.terrain].filter((h) => h != null);
    const heightSpan = Math.max(...heights) - Math.min(...heights);
    const useGround = groundSpan > 0.25 * heightSpan;
    const xs = useGround ? d.ground : d.t;
    const xLabel = useGround ? 'Horizontal distance from the camera' : 'Distance along the ray';

    let y0 = Math.min(...heights);
    let y1 = Math.max(...heights);
    const pad = 0.06 * Math.max(y1 - y0, 10);
    y0 -= pad;
    y1 += pad;
    const x1 = Math.max(xs[n - 1] ?? 1, 1);

    const w = rect.width - PAD.left - PAD.right;
    const h = rect.height - PAD.top - PAD.bottom;
    return {
      rect, xs, xLabel, x1, y0, y1,
      X: (x) => PAD.left + (x / x1) * w,
      Y: (y) => PAD.top + (1 - (y - y0) / (y1 - y0)) * h,
      w, h,
    };
  }

  draw() {
    const canvas = this.canvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!this.data || !this.colors) return;

    const c = this.colors;
    const d = this.data;
    const L = this.layout();
    const { X, Y, xs } = L;
    ctx.font = '11px system-ui, sans-serif';

    // Hairline grid and the axis labels.
    ctx.strokeStyle = c.grid;
    ctx.fillStyle = c.muted;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of ticks(L.y0, L.y1, 4)) {
      const y = Math.round(Y(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + L.w, y); ctx.stroke();
      ctx.fillText(num(t, 0), PAD.left - 6, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of ticks(0, L.x1, 5)) {
      const x = Math.round(X(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + L.h); ctx.stroke();
      ctx.fillText(L.x1 >= 5000 ? `${num(t / 1000, t % 1000 ? 1 : 0)} km` : `${num(t, 0)} m`, x, PAD.top + L.h + 5);
    }
    ctx.fillText(L.xLabel, PAD.left + L.w / 2, PAD.top + L.h + 17);
    ctx.save();
    ctx.translate(11, PAD.top + L.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`${d.label} (m)`, 0, -2);
    ctx.restore();

    // The terrain: a 10 % wash under a 2 px line.
    const path = (values) => {
      ctx.beginPath();
      let pen = false;
      values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        if (pen) ctx.lineTo(X(xs[i]), Y(v)); else ctx.moveTo(X(xs[i]), Y(v));
        pen = true;
      });
    };
    ctx.save();
    ctx.globalAlpha = 0.1;
    ctx.fillStyle = c.terrainLine;
    ctx.beginPath();
    ctx.moveTo(X(xs[0]), Y(L.y0));
    d.terrain.forEach((v, i) => ctx.lineTo(X(xs[i]), Y(v ?? L.y0)));
    ctx.lineTo(X(xs[xs.length - 1]), Y(L.y0));
    ctx.fill();
    ctx.restore();
    ctx.lineJoin = ctx.lineCap = 'round';
    ctx.lineWidth = 2;
    ctx.strokeStyle = c.terrainLine;
    path(d.terrain);
    ctx.stroke();

    // The ray: solid up to the hit, dashed after it.
    const tHit = d.hit.range;
    const split = tHit == null ? d.t.length : d.t.findIndex((t) => t > tHit);
    const end = split < 0 ? d.t.length : split;
    ctx.strokeStyle = c.selected;
    path(d.ray.map((v, i) => (i <= end ? v : null)));
    ctx.stroke();
    ctx.setLineDash([4, 4]);
    path(d.ray.map((v, i) => (i >= end - 1 ? v : null)));
    ctx.stroke();
    ctx.setLineDash([]);

    // The closest earlier approach to the terrain. A small clearance is a risk.
    if (d.clearance) {
      const i = d.t.findIndex((t) => t >= d.clearance.t);
      if (i >= 0 && d.ray[i] != null && d.terrain[i] != null) {
        const x = X(xs[i]);
        const risk = d.clearance.clearance < 20 ? c.critical : c.text2;
        ctx.strokeStyle = risk;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, Y(d.ray[i])); ctx.lineTo(x, Y(d.terrain[i])); ctx.stroke();
        ctx.fillStyle = risk;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(`closest ${meters(d.clearance.clearance)}`, x + 5, Y(d.ray[i]) - 2);
      }
    }

    // The hit marker: an 8 px dot with a 2 px surface ring.
    if (tHit != null && d.hit.height != null) {
      // Interpolate the x value of the hit between the two samples around it.
      const i = clampIndex(end - 1, xs.length - 2);
      const f = (tHit - d.t[i]) / Math.max(d.t[i + 1] - d.t[i], 1e-9);
      const x = X(xs[i] + f * (xs[i + 1] - xs[i]));
      const y = Y(d.hit.height);
      ctx.fillStyle = c.panel;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = d.hit.terrain ? c.selected : c.fallback;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill();
    }

    // The legend: a line key for each series.
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    let lx = PAD.left;
    for (const [name, col] of [['Terrain', c.terrainLine], ['Ray', c.selected]]) {
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(lx, 10); ctx.lineTo(lx + 14, 10); ctx.stroke();
      ctx.fillStyle = c.text2;
      ctx.fillText(name, lx + 19, 10);
      lx += 28 + ctx.measureText(name).width;
    }

    // The crosshair.
    if (this.hoverIndex != null) {
      const i = this.hoverIndex;
      const x = X(xs[i]);
      ctx.strokeStyle = c.axis;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + L.h); ctx.stroke();
      for (const [v, col] of [[d.terrain[i], c.terrainLine], [d.ray[i], c.selected]]) {
        if (v == null) continue;
        ctx.fillStyle = c.panel;
        ctx.beginPath(); ctx.arc(x, Y(v), 5, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(x, Y(v), 3.5, 0, 2 * Math.PI); ctx.fill();
      }
    }
  }

  onMove(e) {
    if (!this.data) return;
    const L = this.layout();
    const x = ((e.clientX - L.rect.left - PAD.left) / L.w) * L.x1;
    let best = 0;
    for (let i = 1; i < L.xs.length; i++) if (Math.abs(L.xs[i] - x) < Math.abs(L.xs[best] - x)) best = i;
    this.hoverIndex = best;
    this.draw();

    const d = this.data;
    const ray = d.ray[best], ter = d.terrain[best];
    this.tip.innerHTML = [
      `<b>${escapeHtml(L.xLabel)}</b> ${meters(L.xs[best])}`,
      `Range ${meters(d.t[best])}`,
      `Ray ${num(ray, 1)} m`,
      `Terrain ${num(ter, 1)} m`,
      `Clearance ${ray != null && ter != null ? meters(ray - ter) : '–'}`,
    ].join('<br>');
    this.tip.hidden = false;
    const box = this.canvas.parentElement.getBoundingClientRect();
    const left = e.clientX - box.left + 14;
    this.tip.style.left = `${Math.min(left, box.width - this.tip.offsetWidth - 4)}px`;
    this.tip.style.top = `${Math.max(4, e.clientY - box.top - this.tip.offsetHeight - 8)}px`;
  }
}
