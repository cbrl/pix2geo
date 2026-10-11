// GLSL of the terrain, the markers and the camera-view remap.
// three.js compiles these as GLSL ES 3.0 and adds its standard uniforms.

// ---- Terrain ----------------------------------------------------------------
// One material for both views. uPass is 0 in the 3D view and 1 in the camera
// view. The coverage overlays show only in the 3D view, and the haze only in
// the camera view.

export const TERRAIN_VERT = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>

attribute float aHeight;
attribute vec2 aLonLat;
attribute vec2 aPix;
attribute float aVis;

varying float vHeight;
varying vec2 vLonLat;
varying vec2 vPix;
varying float vVis;
varying vec3 vNormal;
varying float vUp;
varying vec3 vWorld;

void main() {
  vHeight = aHeight;
  vLonLat = aLonLat;
  vPix = aPix;
  vVis = aVis;

  // The view-space normal is right under the vertical scale of the 3D view.
  // The slope uses the true normal.
  vNormal = normalize(normalMatrix * normal);
  vUp = normal.z;

  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;

  #include <logdepthbuf_vertex>
}
`;

export const TERRAIN_FRAG = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>

uniform sampler2D uRamp;
uniform sampler2D uSlopeRamp;
uniform vec2 uHeightRange;
uniform vec3 uSun;
uniform float uAmbient;
uniform int uColorMode;
uniform float uContour;
uniform float uGraticule;
uniform vec2 uGratOffset;
uniform int uPass;
uniform float uViewshed;
uniform int uDrape;
uniform sampler2D uImage;
uniform vec2 uImageSize;
uniform float uGridPx;
uniform float uDrapeOpacity;
uniform vec3 uGridColor;
uniform vec3 uLineColor;
uniform vec3 uFogColor;
uniform float uFogDistance;

varying float vHeight;
varying vec2 vLonLat;
varying vec2 vPix;
varying float vVis;
varying vec3 vNormal;
varying float vUp;
varying vec3 vWorld;

// 1 on the lines of a unit grid of a value, 0 away from them, about 1 px wide.
float gridLine(float value) {
  float d = abs(fract(value - 0.5) - 0.5) / max(fwidth(value), 1e-6);
  return 1.0 - min(d, 1.0);
}

void main() {
  #include <logdepthbuf_fragment>

  // Lambert light from the sun, plus ambient light.
  vec3 n = normalize(vNormal);
  vec3 sun = normalize((viewMatrix * vec4(uSun, 0.0)).xyz);
  float light = uAmbient + (1.0 - uAmbient) * max(dot(n, sun), 0.0);

  vec3 base;
  if (uColorMode == 1) {
    base = vec3(0.88, 0.87, 0.84);
  } else if (uColorMode == 2) {
    float slope = acos(clamp(vUp, -1.0, 1.0)) / radians(50.0);
    base = texture2D(uSlopeRamp, vec2(clamp(slope, 0.0, 1.0), 0.5)).rgb;
  } else {
    float t = (vHeight - uHeightRange.x) / max(uHeightRange.y - uHeightRange.x, 1e-3);
    base = texture2D(uRamp, vec2(clamp(t, 0.0, 1.0), 0.5)).rgb;
  }
  vec3 color = base * light;

  // The coverage of the camera, in the 3D view only.
  if (uPass == 0) {
    if (uDrape == 1 && vVis > 0.5) {
      vec2 st = vec2((vPix.x + 0.5) / uImageSize.x, 1.0 - (vPix.y + 0.5) / uImageSize.y);
      color = mix(color, texture2D(uImage, st).rgb, uDrapeOpacity);
    } else if (uDrape == 2 && vVis > 0.5) {
      vec2 g = vPix / uGridPx;
      float line = max(gridLine(g.x), gridLine(g.y));
      color = mix(color, uGridColor, line * uDrapeOpacity);
    }
    if (uViewshed > 0.0) {
      float seen = smoothstep(0.35, 0.65, vVis);
      color = mix(mix(color, vec3(dot(color, vec3(0.33))), 0.5) * (1.0 - uViewshed), color, seen);
    }
  }

  if (uContour > 0.0) {
    color = mix(color, color * 0.55, gridLine(vHeight / uContour));
  }

  if (uGraticule > 0.0) {
    vec2 g = (vLonLat + uGratOffset) / uGraticule;
    color = mix(color, uLineColor, 0.85 * max(gridLine(g.x), gridLine(g.y)));
  }

  // Haze with distance, in the camera view only.
  if (uPass == 1 && uFogDistance > 0.0) {
    float d = length(vWorld - cameraPosition);
    color = mix(color, uFogColor, 1.0 - exp(-d / uFogDistance));
  }

  gl_FragColor = vec4(color, 1.0);
}
`;

// ---- Markers ----------------------------------------------------------------
// Round points of a fixed size in pixels, with a ring of uRingPx pixels.
// uPull moves each point toward the eye in view space. The projection does
// not change, but the point then clears the terrain around it.

export const POINT_VERT = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>

attribute vec3 aColor;
attribute float aSize;
uniform float uScale;
uniform float uPull;
uniform float uRingPx;
varying vec3 vColor;
varying float vRing;

void main() {
  vColor = aColor;
  float vSize = aSize * uScale;
  vRing = 1.0 - 2.0 * uRingPx * uScale / max(vSize, 1.0);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  mv.xyz *= uPull;
  gl_Position = projectionMatrix * mv;
  gl_PointSize = vSize;
  #include <logdepthbuf_vertex>
}
`;

export const POINT_FRAG = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>

uniform vec3 uRing;
varying vec3 vColor;
varying float vRing;

void main() {
  #include <logdepthbuf_fragment>
  float r = length(gl_PointCoord * 2.0 - 1.0);
  if (r > 1.0) discard;
  gl_FragColor = vec4(r > vRing ? uRing : vColor, 1.0);
}
`;

// ---- Camera-view remap ------------------------------------------------------
// The camera view first renders the terrain with a pinhole projection of a
// box of normalized coordinates. This pass then finds, for each screen pixel,
// the image pixel, its normalized coordinates (with the lens distortion
// undone by Newton steps), and the texel of the pinhole render. The sky fills
// the pixels with no terrain: the ellipsoid horizon splits it from the ground.

export const REMAP_VERT = /* glsl */ `
varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const REMAP_FRAG = /* glsl */ `
uniform sampler2D uTex;
uniform vec4 uRegion;
uniform vec4 uBox;
uniform vec4 uK;
uniform float uSkew;
uniform float uDist[8];
uniform int uHasDist;
uniform vec2 uSize;
uniform vec3 uUp;
uniform float uSinDip;
uniform vec3 uSkyTop;
uniform vec3 uSkyLow;
uniform vec3 uGround;
uniform vec3 uLetterbox;

varying vec2 vUv;

// Distort normalized coordinates, and give the symmetric Jacobian (j11, j12, j22).
vec2 distortJ(vec2 p, out vec3 J) {
  float k1 = uDist[0], k2 = uDist[1], p1 = uDist[2], p2 = uDist[3];
  float k3 = uDist[4], k4 = uDist[5], k5 = uDist[6], k6 = uDist[7];
  float x = p.x, y = p.y, xy = p.x * p.y;
  float r2 = x * x + y * y;
  float num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
  float den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6));
  float radial = num / den;
  float dRadial = (k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2) - radial * (k4 + r2 * (2.0 * k5 + 3.0 * k6 * r2))) / den;
  J = vec3(
    radial + 2.0 * x * x * dRadial + 2.0 * p1 * y + 6.0 * p2 * x,
    2.0 * xy * dRadial + 2.0 * p1 * x + 2.0 * p2 * y,
    radial + 2.0 * y * y * dRadial + 6.0 * p1 * y + 2.0 * p2 * x
  );
  return vec2(
    x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x),
    y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy
  );
}

vec2 undistort(vec2 pd, out bool ok) {
  vec2 p = pd;
  vec3 J;
  for (int i = 0; i < 20; i++) {
    vec2 e = distortJ(p, J) - pd;
    float det = J.x * J.z - J.y * J.y;
    det = abs(det) < 1e-12 ? 1e-12 : det;
    p -= vec2(J.z * e.x - J.y * e.y, J.x * e.y - J.y * e.x) / det;
  }
  ok = length(distortJ(p, J) - pd) < 1e-4;
  return p;
}

void main() {
  // The image pixel under this fragment. The region gives the pixel edges
  // at the left, top, right and bottom of the viewport.
  float u = mix(uRegion.x, uRegion.z, vUv.x);
  float v = mix(uRegion.y, uRegion.w, 1.0 - vUv.y);
  if (u < -0.5 || v < -0.5 || u > uSize.x - 0.5 || v > uSize.y - 0.5) {
    gl_FragColor = vec4(uLetterbox, 1.0);
    return;
  }

  // Invert K, then the distortion.
  float yn = (v - uK.w) / uK.y;
  vec2 n = vec2((u - uK.z - uSkew * yn) / uK.x, yn);
  if (uHasDist == 1) {
    bool ok;
    n = undistort(n, ok);
    if (!ok) {
      gl_FragColor = vec4(uLetterbox * 0.6, 1.0);
      return;
    }
  }

  // The sky above the horizon of the ellipsoid, and the ground below it.
  vec3 dir = normalize(vec3(n, 1.0));
  float e = dot(dir, uUp) + uSinDip;
  vec3 sky = e > 0.0 ? mix(uSkyLow, uSkyTop, smoothstep(0.0, 0.5, e)) : uGround;

  // The pinhole render holds premultiplied color with the terrain coverage in alpha.
  vec2 st = vec2((n.x - uBox.x) / (uBox.z - uBox.x), 1.0 - (n.y - uBox.y) / (uBox.w - uBox.y));
  vec4 c = texture2D(uTex, st);
  gl_FragColor = vec4(c.rgb + sky * (1.0 - c.a), 1.0);
}
`;
