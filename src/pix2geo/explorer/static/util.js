// Small helpers: DOM, the API, request pumps, number formats and messages.

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

// ---- API --------------------------------------------------------------------

// Call the server. A body makes a POST request. An error response throws its message.
export async function api(name, body) {
  const init = body === undefined
    ? {}
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const res = await fetch(`/api/${name}`, init);
  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(`${name}: ${res.status} ${res.statusText}`);
  }
  if (!res.ok) throw new Error(data.error || `${name}: ${res.status}`);
  return data;
}

function bytes(b64) {
  const text = atob(b64);
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
  return out;
}

// Base64 text of little-endian float32 or uint8 values.
export const f32 = (b64) => new Float32Array(bytes(b64).buffer);
export const u8 = (b64) => bytes(b64);

// A pump sends one request at a time. A kick during a request sends one more
// request after it, with the state of that time. Thus a fast slider makes
// at most one request in flight, and the last state always goes out.
export class Pump {
  constructor(name, build, apply, onError) {
    this.name = name;
    this.build = build;
    this.apply = apply;
    this.onError = onError;
    this.pending = false;
    this.busy = false;
  }

  kick() {
    this.pending = true;
    if (!this.busy) this.run();
  }

  async run() {
    this.busy = true;
    while (this.pending) {
      this.pending = false;
      const req = this.build();
      if (req == null) continue;
      const start = performance.now();
      try {
        const res = await api(this.name, req);
        this.apply(res, performance.now() - start, req);
      } catch (err) {
        this.onError(err);
      }
    }
    this.busy = false;
  }
}

export function debounce(fn, ms) {
  let timer = 0;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

// ---- Formats ----------------------------------------------------------------

const DASH = '–';

export function num(x, digits = 1) {
  if (x == null || !Number.isFinite(x)) return DASH;
  return x.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function meters(x, digits) {
  if (x == null || !Number.isFinite(x)) return DASH;
  const a = Math.abs(x);
  if (a >= 10000) return `${num(x / 1000, 1)} km`;
  if (a >= 1000) return `${num(x / 1000, 2)} km`;
  return `${num(x, digits ?? (a < 10 ? 2 : 1))} m`;
}

export function area(m2) {
  if (m2 == null || !Number.isFinite(m2)) return DASH;
  if (m2 >= 1e6) return `${num(m2 / 1e6, 2)} km²`;
  if (m2 >= 1e4) return `${num(m2 / 1e4, 2)} ha`;
  return `${num(m2, 0)} m²`;
}

export const deg = (x, digits = 7) => (x == null ? DASH : `${num(x, digits)}°`);

export function dms(value, pos, neg) {
  if (value == null) return DASH;
  const a = Math.abs(value);
  const d = Math.floor(a);
  const m = Math.floor((a - d) * 60);
  const s = (a - d - m / 60) * 3600;
  return `${d}° ${String(m).padStart(2, '0')}′ ${s.toFixed(2).padStart(5, '0')}″ ${value >= 0 ? pos : neg}`;
}

export const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---- Messages and files -----------------------------------------------------

export function toast(message, kind = 'info', ms = 4500) {
  const box = document.createElement('div');
  box.className = `toast ${kind}`;
  box.textContent = message;
  $('#toasts').append(box);
  setTimeout(() => box.remove(), ms);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // The clipboard API needs a secure context. A text area works everywhere.
    const area = document.createElement('textarea');
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }
}

export function download(name, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// The color of a CSS custom property as #rrggbb, for WebGL and the 2D
// canvases. A hidden probe element resolves light-dark() for the theme.
let probe = null;
export function cssColor(name) {
  if (!probe) {
    probe = document.createElement('i');
    probe.hidden = true;
    document.body.append(probe);
  }
  probe.style.color = `var(${name})`;
  const rgb = (getComputedStyle(probe).color.match(/[\d.]+/g) ?? []).slice(0, 3);
  return `#${rgb.map((x) => Math.round(Number(x)).toString(16).padStart(2, '0')).join('')}`;
}

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// A number rounded to `digits` decimals, without trailing zeros.
export const round = (x, digits) => Number(Number(x).toFixed(digits));

// A round step near `x`: 1, 2 or 5 times a power of ten.
export function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x));
  const f = x / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}

// An angle in degrees in [-180, 180).
export const wrap180 = (a) => ((((a + 180) % 360) + 360) % 360) - 180;
