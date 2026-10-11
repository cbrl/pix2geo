// Exports, links and session files. Each function reads the page state `S`
// of main.js, or makes a patch of it.

import { round } from './util.js';

// ---- Picks ------------------------------------------------------------------

export function picksCsv(S) {
  const head = 'n,u,v,lat,lon,alt_ellipsoid,alt_msl,range,status';
  const rows = S.picks.map((p, i) => {
    const r = p.res ?? {};
    return [i + 1, p.u, p.v, r.lat, r.lon, r.alt, r.alt_msl, r.range, r.status].map((x) => x ?? '').join(',');
  });
  return [head, ...rows].join('\n');
}

// The picks that land, and the image footprint when every border ray lands.
export function picksGeojson(S) {
  const features = [];
  S.picks.forEach((p, i) => {
    const r = p.res;
    if (!r || r.lat == null) return;
    features.push({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [r.lon, r.lat, r.alt] },
      properties: { n: i + 1, u: p.u, v: p.v, alt_msl: r.alt_msl, range: r.range, status: r.status },
    });
  });

  const border = S.cast?.border.lonlat;
  if (border && border.every(([lon]) => lon != null)) {
    features.push({
      type: 'Feature',
      geometry: { type: 'Polygon', coordinates: [border] },
      properties: { kind: 'footprint', area_m2: S.cast.border.area },
    });
  }
  return JSON.stringify({ type: 'FeatureCollection', features }, null, 2);
}

// The pix2geo command that casts the picks, or the image center.
export function cliCommand(S) {
  const p = S.pose, c = S.camera, cli = S.info.cli;
  const pixels = S.picks.length ? S.picks : [{ u: c.cx, v: c.cy }];
  return [
    'pix2geo',
    cli.dem.length ? cli.dem.map((d) => `--dem "${d}"`).join(' ') : '--dem DEM.tif',
    cli.geoid != null ? `--geoid ${cli.geoid}` : '',
    `--lat ${p.lat.toFixed(8)} --lon ${p.lon.toFixed(8)} --alt ${round(p.alt, 3)} --alt-ref ${p.alt_ref}`,
    `--yaw ${round(p.yaw, 4)} --pitch ${round(p.pitch, 4)} --roll ${round(p.roll, 4)}`,
    `--size ${c.width} ${c.height} --K ${round(c.fx, 4)} ${round(c.fy, 4)} ${round(c.cx, 4)} ${round(c.cy, 4)}`,
    c.dist.some((x) => x) ? `--dist ${c.dist.map((x) => round(x, 8)).join(' ')}` : '',
    S.fallback.on ? `--fallback-height ${S.fallback.height}` : '',
    ...pixels.map((q) => `--pixel ${round(q.u, 3)} ${round(q.v, 3)}`),
  ].filter(Boolean).join(' ');
}

// ---- Sessions ---------------------------------------------------------------

export function sessionJson(S) {
  return {
    version: 1,
    pose: S.pose,
    camera: S.camera,
    fallback: S.fallback,
    rays: S.rays,
    display: S.display,
    layer: { name: S.layer.name, step: S.layer.step, opacity: S.layer.opacity },
    showHorizon: S.showHorizon,
    measure: S.measure,
    picks: S.picks.map((p) => [p.u, p.v]),
  };
}

// The URL hash: the pose, the camera, the picks and a few settings.
export function hashString(S) {
  const p = S.pose, c = S.camera;
  const parts = [
    `pose=${[round(p.lat, 8), round(p.lon, 8), round(p.alt, 3), p.alt_ref, round(p.yaw, 4), round(p.pitch, 4), round(p.roll, 4)].join(',')}`,
    `cam=${[c.width, c.height, round(c.fx, 4), round(c.fy, 4), round(c.cx, 4), round(c.cy, 4), round(c.skew, 6), ...c.dist.map((x) => round(x, 8))].join(',')}`,
  ];
  if (S.picks.length) parts.push(`picks=${S.picks.map((q) => `${round(q.u, 2)}:${round(q.v, 2)}`).join(';')}`);
  if (S.fallback.on) parts.push(`fallback=${S.fallback.height}`);
  if (S.layer.name !== 'none') parts.push(`layer=${S.layer.name}:${S.layer.step}`);
  parts.push(`ve=${S.display.ve}`);
  return parts.join('&');
}

// A session patch from the URL hash. A bad value throws.
export function parseHash(hash) {
  const h = new URLSearchParams(hash.replace(/^#/, ''));
  const nums = (s) => s.split(',').map(Number);
  const j = {};
  if (h.get('pose')) {
    const v = h.get('pose').split(',');
    j.pose = { lat: +v[0], lon: +v[1], alt: +v[2], alt_ref: v[3], yaw: +v[4], pitch: +v[5], roll: +v[6] };
  }
  if (h.get('cam')) {
    const v = nums(h.get('cam'));
    const dist = v.slice(7, 15);
    while (dist.length < 8) dist.push(0);
    j.camera = { width: v[0], height: v[1], fx: v[2], fy: v[3], cx: v[4], cy: v[5], skew: v[6], dist };
  }
  if (h.get('picks')) j.picks = h.get('picks').split(';').map((s) => s.split(':').map(Number));
  if (h.get('fallback')) j.fallback = { on: true, height: Number(h.get('fallback')) };
  if (h.get('layer')) {
    const [name, step] = h.get('layer').split(':');
    j.layer = { name, step: Number(step) || 4 };
  }
  if (h.get('ve')) j.display = { ve: Number(h.get('ve')) };
  return j;
}
