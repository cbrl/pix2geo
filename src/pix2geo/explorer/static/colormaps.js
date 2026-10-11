// Color ramps. Height uses hypsometric tints: green lowland, tan slopes, brown
// ridges and white peaks. Magnitudes use one hue, light to dark: blue for the
// first, orange for the second.

export const TERRAIN = ['#4a7c4e', '#9bb87a', '#d8c99b', '#9c7a5b', '#f4f1ec'];
const TERRAIN_STOPS = [0, 0.3, 0.55, 0.8, 1];

export const BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
export const ORANGES = ['#fde3d5', '#f8bb98', '#f28e5e', '#eb6834', '#c44f20', '#953b16', '#66280e'];

// A #rrggbb color as [r, g, b] bytes.
export function hexRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// A lookup table of `n` RGBA colors, as bytes.
export function lut(colors, n = 256, stops = null) {
  const pos = stops ?? colors.map((_, i) => i / (colors.length - 1));
  const rgbs = colors.map(hexRgb);
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    let k = 0;
    while (k < pos.length - 2 && t > pos[k + 1]) k++;
    const f = Math.min(1, Math.max(0, (t - pos[k]) / (pos[k + 1] - pos[k])));
    for (let c = 0; c < 3; c++) out[4 * i + c] = Math.round(rgbs[k][c] + f * (rgbs[k + 1][c] - rgbs[k][c]));
    out[4 * i + 3] = 255;
  }
  return out;
}

export const terrainLut = () => lut(TERRAIN, 256, TERRAIN_STOPS);

// A CSS gradient of a ramp, for the legends.
export function gradient(colors, stops = null) {
  const pos = stops ?? colors.map((_, i) => i / (colors.length - 1));
  return `linear-gradient(90deg, ${colors.map((c, i) => `${c} ${(pos[i] * 100).toFixed(0)}%`).join(', ')})`;
}

export const terrainGradient = () => gradient(TERRAIN, TERRAIN_STOPS);
