// The HTML of the picks panel and of the hover tip. A pick is
// {id, u, v, res}, where `res` is its row of /api/cast. The number of a pick
// is its place in the list, from 1.

import { area, deg, dms, meters, num } from './util.js';

const WAIT = '<span class="muted">…</span>';

// The height that the page shows: above the geoid when the terrain has one.
export const shownHeight = (r) => r.alt_msl ?? r.alt;

// The range of a ray, or its state when it does not hit the terrain.
function rangeHtml(r) {
  if (r.status === 'miss') return '<span class="tag">miss</span>';
  const range = meters(r.range);
  return r.status === 'fallback' ? `${range} <span class="tag fallback">fallback</span>` : range;
}

// ---- Hover tip --------------------------------------------------------------

export function tipHtml(r) {
  const pixel = `<span class="muted">${num(r.u, 1)}, ${num(r.v, 1)}</span>`;
  if (r.status === 'miss') return `${pixel}<br>The ray misses the terrain`;
  return [
    pixel,
    `${deg(r.lat, 6)}, ${deg(r.lon, 6)}`,
    `Height ${num(shownHeight(r), 1)} m · range ${rangeHtml(r)}`,
  ].join('<br>');
}

// ---- Table ------------------------------------------------------------------

export function picksTable(picks, selected, heightLabel) {
  const rows = picks.map((p, i) => {
    const r = p.res;
    const cells = r
      ? [deg(r.lat, 6), deg(r.lon, 6), num(shownHeight(r), 1), rangeHtml(r)]
      : [WAIT, WAIT, WAIT, WAIT];
    return `<tr data-id="${p.id}"${p.id === selected ? ' class="selected"' : ''}>
      <td><span class="pill">${i + 1}</span></td>
      <td>${num(p.u, 1)}, ${num(p.v, 1)}</td>
      ${cells.map((c) => `<td>${c}</td>`).join('')}
      <td class="del" data-del="${p.id}" title="Remove (Delete)">✕</td>
    </tr>`;
  });
  return `<table class="picks"><thead><tr>
    <th>#</th><th>Pixel</th><th>Latitude</th><th>Longitude</th><th>${heightLabel}</th><th>Range</th><th></th>
  </tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

// The geodesic path through the picks that land, and the area inside it.
export function measureHtml(m, picks) {
  if (!m || !m.segments.length) return '<span class="muted">Pick two or more points on the terrain.</span>';
  const n = new Map(picks.map((p, i) => [p.id, i + 1]));
  const total = [`Path ${meters(m.length)}`];
  if (m.area != null) total.push(`area ${area(m.area)}`, `perimeter ${meters(m.perimeter)}`);
  const legs = m.segments.map((s) => `${n.get(s.from)}→${n.get(s.to)} ${meters(s.distance)} at ${num(s.azimuth, 0)}°`);
  return `<div>${total.join(' · ')}</div><div class="muted">${legs.join(' · ')}</div>`;
}

// ---- Selected pick ----------------------------------------------------------

export function pickHead(p, n) {
  const r = p.res;
  const tools = r && r.lat != null
    ? `<button data-act="aim" title="Point the camera at this point">Aim</button>
      <button data-act="copy" title="Copy the latitude and longitude">Copy</button>
      <a class="button" href="https://www.openstreetmap.org/?mlat=${r.lat}&mlon=${r.lon}#map=15/${r.lat}/${r.lon}"
        target="_blank" rel="noopener" title="Open the point in OpenStreetMap">Map ↗</a>`
    : '';
  return `<b>Pick</b><span class="pill">${n}</span><span class="spacer"></span>${tools}`;
}

// The values of the selected pick that the table does not show.
export function pickFacts(r) {
  if (!r) return '';
  if (r.status === 'miss') {
    return fact('Ray', 'The ray misses the terrain. The fallback surface (Rays) gives it a point.', '', true);
  }
  const facts = [
    fact('Position', `${dms(r.lat, 'N', 'S')}, ${dms(r.lon, 'E', 'W')}`),
    fact('UTM', r.utm ? `${r.utm.zone} ${num(r.utm.easting, 2)} E ${num(r.utm.northing, 2)} N` : '–'),
  ];
  if (r.alt_msl != null) facts.push(fact('Ellipsoid h', `${num(r.alt, 2)} m`));
  facts.push(
    fact('ECEF', r.ecef.map((x) => num(x, 2)).join(', ')),
    fact('From camera', r.ground_distance != null ? `${meters(r.ground_distance)} at ${num(r.azimuth, 1)}°` : '–'),
    fact('Depression', r.depression != null ? `${num(r.depression, 2)}°` : '–'),
    fact('Slope', r.slope != null ? `${num(r.slope, 1)}° · grazing ${num(r.grazing, 1)}°` : '–'),
    fact('GSD', r.gsd ? `${num(r.gsd[0], 2)} × ${num(r.gsd[1], 2)} m` : '–'),
    fact('DEM error', r.sensitivity != null ? `${meters(r.sensitivity)} shift per 1 m` : '–',
      'The horizontal shift of the hit for 1 m of DEM height error'),
  );
  return facts.join('');
}

// One label and value. A wide fact spans the two columns.
function fact(label, value, title = '', wide = false) {
  const attrs = `${title ? ` title="${title}"` : ''}${wide ? ' class="wide"' : ''}`;
  return `<div${attrs}><dt>${label}</dt><dd>${value}</dd></div>`;
}

// A warning for a ray that passes close to the terrain before its hit.
export function profileNote(res) {
  const c = res.clearance;
  if (!c || c.clearance >= 20) return '';
  const effect = res.hit.range != null ? 'move the hit far' : 'make the ray hit';
  return `The ray passes ${meters(c.clearance)} above the terrain at ${meters(c.t)}. A small error of the pose or the DEM can ${effect}.`;
}
