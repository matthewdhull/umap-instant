// WebGPU renderer for the Jetstream demo: dots straight from the engine's position buffer, recent-arrival
// emphasis by stream time, topic/era/local-fit colouring, stranded markers, a growth density layer and
// click-to-inspect highlights. No ground-truth layout exists here, so there is no blink comparison.
const UNI = /* wgsl */`
struct Uniforms { center: vec2<f32>, viewport: vec2<f32>, scale: f32, pointPx: f32, mode: u32, nBase: u32, style: u32, flushStart: u32,
                  dimBase: f32, curTime: f32, tMin: f32, tMax: f32, recent: f32, settle: f32, thr: f32, window: f32, density: f32, fitMid: f32, focus: i32, ghost: f32, boost: f32, animNow: f32,
                  rate: f32, regPal: f32, prevPal: f32, rmix: f32, heatOn: f32, heatPal: f32, heatPrevPal: f32, pad3: f32 };
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> start: array<f32>;
@group(0) @binding(3) var<storage, read> tsec: array<f32>;
fn worldOf(ii: u32) -> vec2<f32> {
  let p = vec2<f32>(pos[2u * ii], pos[2u * ii + 1u]);
  if (ii >= U.flushStart && U.settle < 1.0) { let s = vec2<f32>(start[2u * ii], start[2u * ii + 1u]); return mix(s, p, U.settle); }
  return p;
}
fn clipOf(world: vec2<f32>, corner: vec2<f32>, size: f32) -> vec4<f32> {
  let px = (world - U.center) * U.scale;
  return vec4<f32>((px.x + corner.x * size * 0.5) / (U.viewport.x * 0.5), (px.y + corner.y * size * 0.5) / (U.viewport.y * 0.5), 0.0, 1.0);
}`;

const POINTS = UNI + /* wgsl */`
@group(0) @binding(4) var<storage, read> attr: array<u32>;      // label | reply << 16
@group(0) @binding(5) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> score: array<f32>;
@group(0) @binding(7) var<storage, read> region: array<i32>;   // packed: (new region + 1) | (previous region + 1) << 16, 0 = none
fn regionNew(ii: u32) -> i32 { return i32(u32(region[ii]) & 0xffffu) - 1; }
fn regionPrev(ii: u32) -> i32 { return i32((u32(region[ii]) >> 16u) & 0xffffu) - 1; }
@group(0) @binding(8) var<storage, read> anim: array<f32>;   // per row: [wall-clock seconds when placed (base rows: long ago), seconds when retired (0 = living)]
// no ninth storage buffer here (WebGPU allows eight per stage): the heat colour of each region lives in a third block of the palette
const GROW_S: f32 = 0.3;   // a new dot grows from nothing to full size over this many real seconds
fn growOf(ii: u32) -> f32 { let g = clamp((U.animNow - anim[2u * ii]) / GROW_S, 0.0, 1.0); return g * g * (3.0 - 2.0 * g); }   // ease in, ease out
struct VSOut { @builtin(position) p: vec4<f32>, @location(0) color: vec4<f32>, @location(1) uv: vec2<f32>, @location(2) @interpolate(flat) ring: u32 };
fn era(t: f32) -> vec3<f32> { let c0 = vec3<f32>(0.22, 0.20, 0.55); let c1 = vec3<f32>(0.13, 0.60, 0.62); let c2 = vec3<f32>(0.98, 0.86, 0.25); return select(mix(c1, c2, (t - 0.5) * 2.0), mix(c0, c1, t * 2.0), t < 0.5); }
fn fitColor(s: f32) -> vec3<f32> { let bad = vec3<f32>(0.85, 0.35, 0.19); let mid = vec3<f32>(0.62, 0.64, 0.68); let good = vec3<f32>(0.11, 0.62, 0.46); let m = max(U.fitMid, 0.05); return select(mix(mid, good, clamp((s - m) / max(1.0 - m, 0.05), 0.0, 1.0)), mix(bad, mid, clamp(s / m, 0.0, 1.0)), s < m); }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var corners = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  let a = attr[ii]; let label = a & 0xffffu; let reply = a >> 16u; let t = tsec[ii]; let sc = score[ii];
  let age = (U.animNow - anim[2u * ii]) * U.rate;   // emphasis age in stream seconds, from real time since the row appeared (the stream clock can jump when live)
  var size = U.pointPx; var rgb: vec3<f32>; var alpha: f32 = 1.0; var ring: u32 = 0u; var o: VSOut; var fade: f32 = 1.0;
  let rt = anim[2u * ii + 1u];
  if (rt > 0.0) { let f = clamp((U.animNow - rt) / 2.0, 0.0, 1.0); if (f >= 1.0 || U.style == 2u) { o.p = vec4<f32>(2.0, 2.0, 0.0, 1.0); o.color = vec4<f32>(0.0); o.uv = vec2<f32>(0.0); o.ring = 1u; return o; } fade = 1.0 - f; }
  if (U.style == 2u) {   // stranded marker pass: only recent arrivals below the threshold get a ring
    if (sc >= U.thr || ii < U.nBase || age >= U.recent || (U.focus >= 0 && regionNew(ii) != U.focus)) { o.p = vec4<f32>(2.0, 2.0, 0.0, 1.0); o.color = vec4<f32>(0.0); o.uv = vec2<f32>(0.0); o.ring = 1u; return o; }
    size = U.pointPx * 3.2; rgb = vec3<f32>(0.96, 0.45, 0.25); alpha = 0.95; ring = 1u;
  } else {
    if (U.mode == 0u) { rgb = palette[label].rgb; }
    else if (U.mode == 1u) { rgb = era(1.0 - clamp((U.curTime - t) / max(1.0, U.tMax - U.tMin), 0.0, 1.0)); }   // age over the loaded stream's length: newest yellow, oldest and base rows blue
    else if (U.mode == 2u) { rgb = select(vec3<f32>(1.0, 0.70, 0.30), vec3<f32>(0.55, 0.60, 0.68), ii < U.nBase); }
    else if (U.mode == 3u) { rgb = fitColor(sc); }
    else if (U.mode == 5u) { rgb = select(vec3<f32>(0.95, 0.75, 0.30), vec3<f32>(0.35, 0.65, 0.95), reply == 1u); }
    else if (U.mode == 6u) {   // region colour, crossfading from the previous membership while the regions change
      let rn = regionNew(ii); let rp = regionPrev(ii); let none = vec3<f32>(0.40, 0.44, 0.50);
      let cn = select(none, palette[u32(U.regPal) + u32(max(rn, 0))].rgb, rn >= 0); let cp = select(none, palette[u32(U.prevPal) + u32(max(rp, 0))].rgb, rp >= 0);
      rgb = mix(cp, cn, U.rmix);
      if (U.heatOn > 0.5) { let hcn = select(none, palette[u32(U.heatPal) + u32(max(rn, 0))].rgb, rn >= 0); let hcp = select(none, palette[u32(U.heatPrevPal) + u32(max(rp, 0))].rgb, rp >= 0); rgb = mix(hcp, hcn, U.rmix); } }
    else { rgb = select(vec3<f32>(0.40, 0.44, 0.50), vec3<f32>(1.0, 0.85, 0.40), ii >= U.flushStart); }
    // everything settles to the base-map dimming level; an arrival fades from full brightness to it over the recent window
    alpha = U.dimBase;
    var fresh: bool = ii >= U.nBase && age < U.recent;
    if (U.focus >= 0) {   // region focus: members at a mid level, everything else ghosted; arrivals are handled below so they stay on top
      if (regionNew(ii) == U.focus) { alpha = max(alpha, 0.4); size = max(size, U.pointPx * 1.2); }
      else { alpha = U.ghost; size = U.pointPx * 0.8; fresh = false; }
    }
    if (fresh) {   // arrival emphasis: bigger, whiter and fully opaque, easing back to the settled look over the recent window
      let f = 1.0 - age / U.recent; let gs = growOf(ii); size = U.pointPx * (1.0 + 0.6 * f) * gs; alpha = mix(U.dimBase, 1.0, f) * gs;
    }
  }
  o.p = clipOf(worldOf(ii), corners[vi], size * (0.6 + 0.4 * fade)); o.color = vec4<f32>(rgb, alpha * fade); o.uv = corners[vi]; o.ring = ring; return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv); if (r2 > 1.0) { discard; }
  if (in.ring == 1u && r2 < 0.6) { discard; }
  return vec4<f32>(in.color.rgb * in.color.a, in.color.a);
}`;

const SPLAT = UNI + /* wgsl */`
@group(0) @binding(4) var<storage, read> anim: array<f32>;
struct VSOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var corners = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  let age = U.curTime - tsec[ii]; var o: VSOut;
  if (ii < U.nBase || age >= U.window || age < 0.0 || anim[2u * ii + 1u] > 0.0) { o.p = vec4<f32>(2.0, 2.0, 0.0, 1.0); o.uv = vec2<f32>(0.0); return o; }
  o.p = clipOf(worldOf(ii), corners[vi], 28.0); o.uv = corners[vi]; return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> { let r2 = dot(in.uv, in.uv); if (r2 > 1.0) { discard; } let w = (1.0 - r2) * (1.0 - r2); return vec4<f32>(w, 0.0, 0.0, 1.0); }`;

const BLUR = /* wgsl */`
struct UB { dx: f32, dy: f32, sigma: f32, pad: f32 };
@group(0) @binding(0) var<uniform> B: UB;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct VSOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut { var pts = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0)); var o: VSOut; o.p = vec4<f32>(pts[vi], 0.0, 1.0); o.uv = vec2<f32>((pts[vi].x + 1.0) * 0.5, 1.0 - (pts[vi].y + 1.0) * 0.5); return o; }
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let r = i32(ceil(3.0 * B.sigma)); var acc = vec4<f32>(0.0); var wsum = 0.0;
  for (var k: i32 = -r; k <= r; k = k + 1) { let w = exp(-f32(k * k) / (2.0 * B.sigma * B.sigma)); acc = acc + w * textureSample(tex, samp, in.uv + f32(k) * vec2<f32>(B.dx, B.dy)); wsum = wsum + w; }
  return acc / wsum;
}`;

const COMPOSITE = /* wgsl */`
struct U2 { density: f32, pad: f32, pad2: f32, pad3: f32 };
@group(0) @binding(0) var<uniform> U: U2;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct VSOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut { var pts = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0)); var o: VSOut; o.p = vec4<f32>(pts[vi], 0.0, 1.0); o.uv = vec2<f32>((pts[vi].x + 1.0) * 0.5, 1.0 - (pts[vi].y + 1.0) * 0.5); return o; }
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let v = textureSample(tex, samp, in.uv).r; let t = clamp(v / U.density, 0.0, 1.0);
  let cold = vec3<f32>(0.98, 0.70, 0.20); let hot = vec3<f32>(1.0, 0.95, 0.75); let rgb = mix(cold, hot, t); let a = 0.85 * smoothstep(0.0, 0.25, t);
  return vec4<f32>(rgb * a, a);
}`;

const HIGHLIGHT = UNI + /* wgsl */`
@group(0) @binding(4) var<storage, read> ids: array<u32>;
@group(0) @binding(5) var<storage, read> kinds: array<u32>;   // 0 selected, 1 forward neighbour, 2 adopter, 3 both
struct VSOut { @builtin(position) p: vec4<f32>, @location(0) color: vec4<f32>, @location(1) uv: vec2<f32>, @location(2) @interpolate(flat) ring: u32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var corners = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  let id = ids[ii]; let kind = kinds[ii]; let world = vec2<f32>(pos[2u * id], pos[2u * id + 1u]);
  var size = U.pointPx * 3.0; var rgb = vec3<f32>(1.0, 1.0, 1.0); var ring: u32 = 1u;
  if (kind == 0u) { size = U.pointPx * 4.5; ring = 0u; }
  else if (kind == 1u) { rgb = vec3<f32>(0.35, 0.90, 0.78); }
  else if (kind == 2u) { rgb = vec3<f32>(0.98, 0.50, 0.78); }
  else { rgb = vec3<f32>(1.0, 0.85, 0.35); }
  var o: VSOut; o.p = clipOf(world, corners[vi], size); o.color = vec4<f32>(rgb, 1.0); o.uv = corners[vi]; o.ring = ring; return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> { let r2 = dot(in.uv, in.uv); if (r2 > 1.0) { discard; } if (in.ring == 1u && r2 < 0.5) { discard; } return in.color; }`;


// Soft region tints, recomputed every frame from the base rows themselves: each base row splats its region's colour with a
// Gaussian footprint into a grid texture (additive, premultiplied), so the tint field follows the rows wherever worldOf()
// puts them, including mid-morph after a quiet rebuild. The composite divides out the weight for colour and turns the
// accumulated density into opacity.
const TINT = UNI + /* wgsl */`
struct RU { ex0: f32, ey0: f32, spanX: f32, spanY: f32, alpha: f32, half: f32, lo: f32, hi: f32 };
@group(0) @binding(4) var<storage, read> region: array<i32>;
@group(0) @binding(5) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> R: RU;
@group(0) @binding(7) var<storage, read> heat: array<f32>;   // per region (new regions), 0 = no heat
fn regionNew(ii: u32) -> i32 { return i32(u32(region[ii]) & 0xffffu) - 1; }
fn regionPrev(ii: u32) -> i32 { return i32((u32(region[ii]) >> 16u) & 0xffffu) - 1; }
struct VSOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec3<f32>, @location(2) wt: f32 };
// drawn indexed (0 1 2, 0 2 3): the same two triangles as six listed corners, with the vertex work done four times per row instead of six
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var corners = array<vec2<f32>, 4>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  var o: VSOut; let rn = regionNew(ii); let rp = regionPrev(ii);
  // membership crossfade: a row's old region fades out and its new one fades in; a row in no region at either end contributes nothing then
  let wn = select(0.0, 1.0, rn >= 0); let wp = select(0.0, 1.0, rp >= 0); let wt = mix(wp, wn, U.rmix);
  if (ii >= U.nBase || wt <= 0.0) { o.p = vec4<f32>(2.0, 2.0, 0.0, 1.0); o.uv = vec2<f32>(0.0); o.color = vec3<f32>(0.0); o.wt = 0.0; return o; }
  let pn = palette[u32(U.regPal) + u32(max(rn, 0))]; let pp = palette[u32(U.prevPal) + u32(max(rp, 0))];   // rgb = region colour, a = its peak density in splat units
  let sn = select(0.0, wn / max(pn.a, 1e-6), rn >= 0); let sp = select(0.0, wp / max(pp.a, 1e-6), rp >= 0); let sw = mix(sp, sn, U.rmix);
  let w = worldOf(ii); let g = vec2<f32>((w.x - R.ex0) / R.spanX * 2.0 - 1.0, (w.y - R.ey0) / R.spanY * 2.0 - 1.0);
  o.p = vec4<f32>(g + corners[vi] * R.half, 0.0, 1.0); o.uv = corners[vi]; o.color = (pp.rgb * sp * (1.0 - U.rmix) + pn.rgb * sn * U.rmix) / max(sw, 1e-9); o.wt = sw;
  // heat colours come ready-made from the palette's heat blocks (the chosen palette), as for the dots
  if (U.heatOn > 0.5) {   // heat: every region on one ramp, the hottest a little stronger; the previous regions' heat (second half of the buffer) blends into the new along the morph
    let hn = select(0.0, heat[u32(max(rn, 0))], rn >= 0); let hp = select(0.0, heat[1024u + u32(max(rp, 0))], rp >= 0);
    let hm = mix(hp, hn, U.rmix); o.color = mix(palette[u32(U.heatPrevPal) + u32(max(rp, 0))].rgb, palette[u32(U.heatPal) + u32(max(rn, 0))].rgb, U.rmix); o.wt = o.wt * (0.35 + 1.0 * hm); }   // cold regions faint, hot ones strong
  return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  // outside the disc: add zero rather than discard (the same sum, bit for bit; discard costs this blended splat about a tenth of its time)
  let r2 = dot(in.uv, in.uv);
  let w = select(0.0, exp(-4.5 * r2) * in.wt, r2 <= 1.0);   // uv = 1 at three sigma
  return vec4<f32>(in.color * w, w);
}`;
const REGION = /* wgsl */`
struct Uniforms { center: vec2<f32>, viewport: vec2<f32>, scale: f32, pointPx: f32, mode: u32, nBase: u32, style: u32, flushStart: u32,
                  dimBase: f32, curTime: f32, tMin: f32, tMax: f32, recent: f32, settle: f32, thr: f32, window: f32, density: f32, fitMid: f32, focus: i32, ghost: f32, boost: f32, animNow: f32,
                  rate: f32, regPal: f32, prevPal: f32, rmix: f32, heatOn: f32, heatPal: f32, heatPrevPal: f32, pad3: f32 };
struct RU { ex0: f32, ey0: f32, spanX: f32, spanY: f32, alpha: f32, half: f32, lo: f32, hi: f32 };
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<uniform> R: RU;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
struct VSOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var c = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
  let world = vec2<f32>(R.ex0 + c[vi].x * R.spanX, R.ey0 + c[vi].y * R.spanY); let px = (world - U.center) * U.scale;
  var o: VSOut; o.p = vec4<f32>(px.x / (U.viewport.x * 0.5), px.y / (U.viewport.y * 0.5), 0.0, 1.0); o.uv = vec2<f32>(c[vi].x, 1.0 - c[vi].y); return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let t = textureSample(tex, samp, in.uv); if (t.a < 1e-4) { discard; }
  let col = t.rgb / t.a; let d = t.a;   // density relative to the region's own peak
  let a = R.alpha * smoothstep(R.lo, R.hi, d) * (0.45 + 0.55 * min(1.0, d));   // regions.js's rule: nothing below lo of the peak, soft edge up to hi, density-weighted inside
  return vec4<f32>(col * a, a);
}`;

const ARRIVAL = UNI + /* wgsl */`
@group(0) @binding(4) var<storage, read> region: array<i32>;
@group(0) @binding(5) var<storage, read> anim: array<f32>;
fn regionNew(ii: u32) -> i32 { return i32(u32(region[ii]) & 0xffffu) - 1; }
const GROW_S: f32 = 0.3;
fn growOf(ii: u32) -> f32 { let g = clamp((U.animNow - anim[2u * ii]) / GROW_S, 0.0, 1.0); return g * g * (3.0 - 2.0 * g); }   // ease in, ease out
struct VSOut { @builtin(position) p: vec4<f32>, @location(0) color: vec4<f32>, @location(1) uv: vec2<f32>, @location(2) @interpolate(flat) core: u32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var corners = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  let age = (U.animNow - anim[2u * ii]) * U.rate; var o: VSOut; o.core = select(0u, 1u, U.style == 4u);
  if (age < 0.0 || age >= U.recent || (U.focus >= 0 && regionNew(ii) != U.focus)) { o.p = vec4<f32>(2.0, 2.0, 0.0, 1.0); o.color = vec4<f32>(0.0); o.uv = vec2<f32>(0.0); return o; }
  let f = 1.0 - age / U.recent; let e = f; let gs = growOf(ii);   // linear fall-off over the recent window; grow-in over the first real seconds
  if (U.style == 4u) {   // solid core: white, full opacity, shrinking with age
    let size = U.pointPx * (1.6 + 1.6 * e * min(U.boost, 2.0)) * gs;
    o.p = clipOf(worldOf(ii), corners[vi], size); o.color = vec4<f32>(1.0, 1.0, 1.0, f * gs);   // linear to nothing at the end of the window
  } else {   // additive halo, several point sizes wide, fading with age
    let size = U.pointPx * (4.0 + 7.0 * U.boost) * (0.35 + 0.65 * e) * gs;
    o.p = clipOf(worldOf(ii), corners[vi], size); o.color = vec4<f32>(1.0, 0.86, 0.45, e * min(1.0, 0.4 * U.boost) * gs);   // linear to nothing at the end of the window
  }
  o.uv = corners[vi]; return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv); if (r2 > 1.0) { discard; }
  if (in.core == 1u) { return vec4<f32>(in.color.rgb * in.color.a, in.color.a); }
  let w = (1.0 - r2) * (1.0 - r2); return vec4<f32>(in.color.rgb * in.color.a * w, 0.0);   // additive: alpha 0 leaves the destination alpha alone
}`;

// Heat palettes: sequential ramps, dark to bright, five stops each. The chosen one colours every region's heat (tints, dots,
// chips); the shaders read the ready-made colours from the palette's heat blocks, so switching needs no shader change.
export const HEAT_PALETTES = {
  ember:   [[0.22, 0.14, 0.06], [0.46, 0.27, 0.08], [0.76, 0.46, 0.12], [0.96, 0.70, 0.26], [1.00, 0.95, 0.72]],
  inferno: [[0.16, 0.04, 0.30], [0.45, 0.10, 0.43], [0.77, 0.24, 0.33], [0.97, 0.55, 0.13], [0.99, 0.93, 0.56]],
  magma:   [[0.12, 0.07, 0.28], [0.38, 0.10, 0.50], [0.71, 0.21, 0.48], [0.98, 0.52, 0.38], [0.99, 0.96, 0.76]],
  viridis: [[0.27, 0.05, 0.36], [0.23, 0.32, 0.55], [0.13, 0.57, 0.55], [0.37, 0.79, 0.38], [0.99, 0.91, 0.14]],
  ice:     [[0.07, 0.12, 0.25], [0.11, 0.26, 0.48], [0.21, 0.47, 0.76], [0.48, 0.73, 0.95], [0.88, 0.96, 1.00]],
  rose:    [[0.22, 0.05, 0.12], [0.46, 0.09, 0.22], [0.76, 0.19, 0.36], [0.96, 0.47, 0.56], [1.00, 0.87, 0.89]],
  mono:    [[0.17, 0.19, 0.23], [0.36, 0.39, 0.44], [0.58, 0.61, 0.66], [0.80, 0.82, 0.86], [1.00, 1.00, 1.00]],
};
let HEAT_STOPS = HEAT_PALETTES.ember;
export function setHeatStops(name) { HEAT_STOPS = HEAT_PALETTES[name] || HEAT_PALETTES.ember; }
export function heatRgb(h) { const n = HEAT_STOPS.length - 1, t = Math.max(0, Math.min(1, h)) * n, i = Math.min(n - 1, Math.floor(t)), f = t - i, a = HEAT_STOPS[i], b = HEAT_STOPS[i + 1]; return [0, 1, 2].map(k => a[k] + (b[k] - a[k]) * f); }
const HEAT_TWEEN_S = 0.8;   // real seconds for the shown heat to cover about two thirds of a change
const FRAMES_AHEAD = 2;   // frames the page may have queued on the GPU before further draws are skipped
const BLEND = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
const ADD = { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } };

export class PointRenderer {
  constructor(device, canvas, engine, data) {
    this.device = device; this.canvas = canvas; this.data = data; this.engine = engine;
    this.ctx = canvas.getContext('webgpu'); this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    const N = data.N, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.attrBuf = device.createBuffer({ size: N * 4, usage: S }); this.tBuf = device.createBuffer({ size: N * 4, usage: S });
    this.appearS = new Float32Array(N).fill(-1e9); this.retiredS = new Float32Array(N); this.lastAppear = -1e9; this.lastRetire = -1e9; this.animNow = 0; this.rate = 1;   // rate = stream seconds per real second, for the arrival emphasis
    this.animBuf = device.createBuffer({ size: N * 8, usage: S }); this.uploadAnim();
    this.updateLabels(0, data.N); this.updateTimes(0, data.n);
    this.REG_MAX = 1024; this.regPalOff = Math.max(1, data.labels.length); this.prevPalOff = this.regPalOff + this.REG_MAX; this.heatPalOff = this.prevPalOff + this.REG_MAX; this.heatPrevPalOff = this.heatPalOff + this.REG_MAX; this.rmix = 1;
    const pal = new Float32Array((this.heatPrevPalOff + this.REG_MAX) * 4);   // label colours | region colours | previous region colours | heat colours per region | previous regions' heat colours
    data.labels.forEach((l, i) => { const c = data.labelColors[l] || '#999999'; pal[4 * i] = parseInt(c.slice(1, 3), 16) / 255; pal[4 * i + 1] = parseInt(c.slice(3, 5), 16) / 255; pal[4 * i + 2] = parseInt(c.slice(5, 7), 16) / 255; pal[4 * i + 3] = 1; });
    this.palBuf = device.createBuffer({ size: pal.byteLength, usage: S }); device.queue.writeBuffer(this.palBuf, 0, pal);
    this.scoreBuf = device.createBuffer({ size: N * 4, usage: S });
    this.regionBuf = device.createBuffer({ size: N * 4, usage: S }); this.focusRegion = -1; this.ghostAlpha = 0.05; this.arrivalBoost = 1.5;
    this.heatBuf = device.createBuffer({ size: 2 * 1024 * 4, usage: S }); this.heatOn = false; this.heatArr = new Float32Array(1024); this.heatPalArr = new Float32Array(1024 * 4);   // [new regions | previous regions]
    this.uni = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniRing = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniComp = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const mkPipe = (code, label, blend, format) => { const module = device.createShaderModule({ code, label }); return device.createRenderPipeline({ label, layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format, blend }] }, primitive: { topology: 'triangle-list' } }); };
    this.pPoints = mkPipe(POINTS, 'points', BLEND, this.format);
    this.pSplat = mkPipe(SPLAT, 'density splat', ADD, 'rgba16float');
    this.pComp = mkPipe(COMPOSITE, 'density composite', BLEND, this.format);
    this.pHighlight = mkPipe(HIGHLIGHT, 'highlight', BLEND, this.format);
    this.pTint = mkPipe(TINT, 'region tint splat', ADD, 'rgba16float'); this.pRegion = mkPipe(REGION, 'soft regions', BLEND, this.format);
    this.pBlur = mkPipe(BLUR, 'tint blur', undefined, 'rgba16float'); this.uniBlurH = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.uniBlurV = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pGlow = mkPipe(ARRIVAL, 'arrival glow', ADD, this.format); this.pCore = mkPipe(ARRIVAL, 'arrival core', BLEND, this.format);
    this.uniGlow = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.uniCore = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.quadIdx = device.createBuffer({ size: 12, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(this.quadIdx, 0, new Uint16Array([0, 1, 2, 0, 2, 3]));
    this.uniRegion = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.regionAlpha = 0.7; this.regions = null; this.tintRef = 4; this.tintScale = 0.15; this.tintG = 512;
    this.tintTex = device.createTexture({ size: [this.tintG, this.tintG], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.tintTex2 = device.createTexture({ size: [this.tintG, this.tintG], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }); this.tintSoft = 6;
    this.hlIds = device.createBuffer({ size: 65536 * 4, usage: S }); this.hlKinds = device.createBuffer({ size: 65536 * 4, usage: S }); this.hlCount = 0;
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.viewOn = false; this.viewPosBuf = null; this.viewStartBuf = null; this.bg = {}; this.rebuildBindGroups();
    this.view = { cx: 0, cy: 0, scale: 40 };
    // Queue guard: at most FRAMES_AHEAD frames submitted and not yet finished on the GPU. A draw asked for beyond that is
    // skipped and, once the queue drains, replaced by one redraw (onRedraw, set by the page) so the last state still shows.
    this.inFlight = 0; this.skipped = 0; this.redrawDue = false; this.onRedraw = null;
    // Optional GPU timing (jet.renderer.gpuTiming = true; needs the timestamp-query feature): the last 600 frames land in
    // gpuStats as { pass: ms, total, span }. span (first pass's start to last pass's end) is the frame's real GPU time: on
    // Apple GPUs the passes overlap, so total (the passes added up) overcounts. Off by default; a frame without a free
    // readback buffer goes untimed.
    this.gpuTiming = false; this.gpuStats = []; this.tsFree = [];
    if (device.features.has('timestamp-query')) { this.tsSet = device.createQuerySet({ type: 'timestamp', count: 16 }); this.tsResolve = device.createBuffer({ size: 16 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }); }
    this.mode = 0; this.pointPx = 2.0; this.dimBase = 0.35; this.recentWindow = 15; this.settle = 1; this.showStranded = false; this.threshold = 0.2;
    this.growth = false; this.growthWindow = 120; this.densityScale = 6; this.fitMid = 0.4; this.curTime = 0; this.tMin = 0; this.tMax = 1; this.flushStart = data.nBase;
    this.fitTo(data.baseXY);
  }
  rebuildBindGroups() {
    const d = this.device, e = this.engine, ent = (list) => list.map((buffer, binding) => ({ binding, resource: { buffer } }));
    const common = (uni) => [uni, this.srcPos, this.srcStart, this.tBuf];
    this.bg.points = d.createBindGroup({ layout: this.pPoints.getBindGroupLayout(0), entries: ent([...common(this.uni), this.attrBuf, this.palBuf, this.scoreBuf, this.regionBuf, this.animBuf]) });
    this.bg.ring = d.createBindGroup({ layout: this.pPoints.getBindGroupLayout(0), entries: ent([...common(this.uniRing), this.attrBuf, this.palBuf, this.scoreBuf, this.regionBuf, this.animBuf]) });
    this.bg.splat = d.createBindGroup({ layout: this.pSplat.getBindGroupLayout(0), entries: ent([...common(this.uni), this.animBuf]) });
    // the arrival shader ages rows by their appear time, not tsec, so binding 3 is absent from its layout
    const arrival = (uni) => [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: this.srcPos } }, { binding: 2, resource: { buffer: this.srcStart } }, { binding: 4, resource: { buffer: this.regionBuf } }, { binding: 5, resource: { buffer: this.animBuf } }];
    this.bg.glow = d.createBindGroup({ layout: this.pGlow.getBindGroupLayout(0), entries: arrival(this.uniGlow) });
    this.bg.core = d.createBindGroup({ layout: this.pCore.getBindGroupLayout(0), entries: arrival(this.uniCore) });
    this.bg.hl = d.createBindGroup({ layout: this.pHighlight.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uni } }, { binding: 1, resource: { buffer: this.srcPos } }, { binding: 4, resource: { buffer: this.hlIds } }, { binding: 5, resource: { buffer: this.hlKinds } }] });   // the highlight shader reads neither start nor tsec
  }
  // View buffers: a second pair of position buffers (start, pos) that the rewind and the time-lapse draw from, so scrubbing
  // through old layouts never touches the engine's own buffers; every shader reads whichever pair is current.
  get srcPos() { return this.viewOn ? this.viewPosBuf : this.posOverride || this.engine.posBuf; }
  // tune mode draws the map from its own buffer while it follows a computation (tunemode.js); null = the engine's positions
  setPosOverride(buf) { if ((this.posOverride || null) === (buf || null)) return; this.posOverride = buf || null; this.rebuildBindGroups(); }
  get srcStart() { return this.viewOn ? this.viewStartBuf : this.engine.startBuf; }
  setViewBuffers(on) {
    if (on && !this.viewPosBuf) { const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST; this.viewPosBuf = this.device.createBuffer({ label: 'view positions', size: this.data.N * 8, usage: S }); this.viewStartBuf = this.device.createBuffer({ label: 'view start positions', size: this.data.N * 8, usage: S }); }
    if (!!this.viewOn === !!on) return; this.viewOn = !!on; this.rebuildBindGroups();
  }
  uploadView(startArr, posArr, row0, rows) { if (rows <= 0) return; this.device.queue.writeBuffer(this.viewStartBuf, row0 * 8, startArr.subarray(row0 * 2, (row0 + rows) * 2)); this.device.queue.writeBuffer(this.viewPosBuf, row0 * 8, posArr.subarray(row0 * 2, (row0 + rows) * 2)); }
  updateLabels(start, count) { const a = new Uint32Array(count), L = this.data.labelS, R = this.data.replyS; for (let i = 0; i < count; i++) a[i] = L[start + i] | (R[start + i] << 16); this.device.queue.writeBuffer(this.attrBuf, start * 4, a); }
  // Rows placed just now: record the wall-clock second so the shaders can grow them in.
  uploadAnim(start = 0, count = this.data.N) { const a = new Float32Array(count * 2); for (let i = 0; i < count; i++) { a[2 * i] = this.appearS[start + i]; a[2 * i + 1] = this.retiredS[start + i]; } this.device.queue.writeBuffer(this.animBuf, start * 8, a); }
  // isFresh(row) false = a backlog row (its stream time is already outside the emphasis window): it appears settled, without grow-in or halo
  updateAppear(start, count, isFresh = null) { if (count <= 0) return; const now = performance.now() / 1000; for (let r = start; r < start + count; r++) this.appearS[r] = (!isFresh || isFresh(r)) ? now : -1e9; this.uploadAnim(start, count); this.lastAppear = now; }
  markRetired(rows) { const now = performance.now() / 1000; for (const r of rows) { this.retiredS[r] = now; this.uploadAnim(r, 1); } this.lastRetire = now; }
  // After the engine compacted its rows: gather the per-row animation arrays by the old-to-new map and re-upload everything per row.
  remap(map, m, regionS, scores) {
    const ap = new Float32Array(this.appearS.length).fill(-1e9); for (let i = 0; i < map.length; i++) { const j = map[i]; if (j >= 0) ap[j] = this.appearS[i]; }
    this.appearS = ap; this.retiredS.fill(0); this.uploadAnim();
    this.updateLabels(0, this.data.N); this.updateTimes(0, m); this.updateRegionIds(regionS, 0, m); this.setScores(scores);
  }
  resetAppear() { this.appearS.fill(-1e9); this.retiredS.fill(0); this.uploadAnim(); this.lastAppear = -1e9; }
  updateTimes(start, count) { if (count > 0) this.device.queue.writeBuffer(this.tBuf, start * 4, this.data.tS.subarray(start, start + count)); }
  // Upload a soft-region field (g x g premultiplied RGBA) covering the given world extent.
  // Regions: membership and colours come from regions.js (density clustering of the base rows); the tint field itself is
  // splatted from the base rows every frame. sigmaGrid = Gaussian sigma in grid cells (grain and edge softness combined).
  // Regions: membership, names and colours come from regions.js (density clustering of the base rows at `sigma` grid cells);
  // the tint field is splatted from the base rows every frame with that same footprint, each row normalised by its region's
  // peak so every region's core reaches 1; `soft` widens the fade at the region's edge.
  setRegions(reg, sigma = 4, soft = 6, prevReg = null) {
    this.regions = reg; if (!reg) { this.bg.region = null; this.bg.tint = null; return; }
    const e = reg.extent, half = 3 * sigma / this.tintG * 2, lo = 0.1, hi = 0.16;   // regions.js truncates at 0.1 of the peak; the edge is then softened by the blur
    this.tintSoft = soft;
    this.device.queue.writeBuffer(this.uniRegion, 0, new Float32Array([e.ex0, e.ey0, e.spanX, e.spanY, this.regionAlpha, half, lo, hi]));
    // palette entry per region: rgb = colour, a = the splat weight its smoothed-count peak corresponds to at the grain it was clustered with
    const colours = (R, sg) => { const unit = 2 * Math.PI * sg * sg, pal = new Float32Array(this.REG_MAX * 4); R.slice(0, this.REG_MAX).forEach((r, k) => { const c = r.color || '#999999'; pal[4 * k] = parseInt(c.slice(1, 3), 16) / 255; pal[4 * k + 1] = parseInt(c.slice(3, 5), 16) / 255; pal[4 * k + 2] = parseInt(c.slice(5, 7), 16) / 255; pal[4 * k + 3] = Math.max(1e-6, (r.peakTint || r.peak || 0) * unit); }); return pal; };
    // the splat is painted at `sigma` (the page caps it below the clustering grain on sparse maps), so the peak reference uses the same width
    this.device.queue.writeBuffer(this.palBuf, this.regPalOff * 16, colours(reg.regions, sigma));
    const pr = prevReg || reg; this.device.queue.writeBuffer(this.palBuf, this.prevPalOff * 16, colours(pr.regions, sigma));
    const E = this.engine, px = 1 / this.tintG;
    this.device.queue.writeBuffer(this.uniBlurH, 0, new Float32Array([px, 0, Math.max(0.5, soft), 0])); this.device.queue.writeBuffer(this.uniBlurV, 0, new Float32Array([0, px, Math.max(0.5, soft), 0]));
    this.bg.tint = this.device.createBindGroup({ layout: this.pTint.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uni } }, { binding: 1, resource: { buffer: this.srcPos } }, { binding: 2, resource: { buffer: this.srcStart } }, { binding: 4, resource: { buffer: this.regionBuf } }, { binding: 5, resource: { buffer: this.palBuf } }, { binding: 6, resource: { buffer: this.uniRegion } }, { binding: 7, resource: { buffer: this.heatBuf } }] });
    this.bg.blurH = this.device.createBindGroup({ layout: this.pBlur.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniBlurH } }, { binding: 1, resource: this.tintTex.createView() }, { binding: 2, resource: this.sampler }] });
    this.bg.blurV = this.device.createBindGroup({ layout: this.pBlur.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniBlurV } }, { binding: 1, resource: this.tintTex2.createView() }, { binding: 2, resource: this.sampler }] });
    this.bg.region = this.device.createBindGroup({ layout: this.pRegion.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uni } }, { binding: 1, resource: { buffer: this.uniRegion } }, { binding: 2, resource: this.tintTex.createView() }, { binding: 3, resource: this.sampler }] });
  }
  // Per-region heat (0 to 1, indexed like the regions list); null clears it.
  // Per-region heat (0 to 1, indexed like the regions list); null clears it. The tint shader reads the scalar from heatBuf; the
  // dots read a ready-made colour per region from the palette's heat block (one storage buffer fewer in the point shader).
  setHeat(h, immediate = false) {
    const target = new Float32Array(this.REG_MAX); if (h) target.set(h.subarray(0, Math.min(h.length, this.REG_MAX)));
    const wasOn = this.heatOn; this.heatOn = !!h; this.heatTarget = target;
    if (immediate || !wasOn || !h) this.applyHeat(target); else this.tweenHeat(0);   // switching on or off jumps; a change while on eases over HEAT_TWEEN_S
  }
  applyHeat(a) { const pal = new Float32Array(this.REG_MAX * 4); for (let k = 0; k < this.REG_MAX; k++) { if (a[k] <= 0 && !this.heatOn) continue; const c = heatRgb(a[k]); pal[4 * k] = c[0]; pal[4 * k + 1] = c[1]; pal[4 * k + 2] = c[2]; pal[4 * k + 3] = a[k]; } this.heatArr = a; this.heatPalArr = pal; this.device.queue.writeBuffer(this.heatBuf, 0, a); this.device.queue.writeBuffer(this.palBuf, this.heatPalOff * 16, pal); }
  // ease the shown heat towards the target by the real time elapsed; returns true while still moving
  tweenHeat(dt) { if (!this.heatOn || !this.heatTarget) return false; const cur = this.heatArr, tg = this.heatTarget, k = 1 - Math.exp(-dt / HEAT_TWEEN_S); let maxd = 0; const next = new Float32Array(this.REG_MAX); for (let i = 0; i < this.REG_MAX; i++) { const d = tg[i] - cur[i]; next[i] = Math.abs(d) < 0.002 ? tg[i] : cur[i] + d * k; if (Math.abs(d) > maxd) maxd = Math.abs(d); } if (maxd < 0.002) return false; this.applyHeat(next); return maxd >= 0.002; }
  // Before the regions are recomputed: the current regions' heat becomes the previous regions' heat, so a morph blends heat to heat.
  rotateHeat() { this.heatPrevArr = this.heatArr; this.device.queue.writeBuffer(this.heatBuf, this.REG_MAX * 4, this.heatArr); this.device.queue.writeBuffer(this.palBuf, this.heatPrevPalOff * 16, this.heatPalArr); }
  // A new palette: repaint both heat blocks (current and previous regions) from the scalars already held.
  setHeatPalette(name) { setHeatStops(name); const cur = this.heatArr; this.applyHeat(cur); const prev = this.heatPrevArr; if (!prev) return; const pal = new Float32Array(this.REG_MAX * 4); for (let k = 0; k < this.REG_MAX; k++) { if (prev[k] <= 0 && !this.heatOn) continue; const c = heatRgb(prev[k]); pal.set([c[0], c[1], c[2], prev[k]], 4 * k); } this.device.queue.writeBuffer(this.palBuf, this.heatPrevPalOff * 16, pal); }
  updateRegionIds(regionS, start, count, prevS = null) { if (count <= 0) return; const a = new Int32Array(count); for (let i = 0; i < count; i++) { const n = regionS[start + i] + 1, p = (prevS ? prevS[start + i] : regionS[start + i]) + 1; a[i] = n | (p << 16); } this.device.queue.writeBuffer(this.regionBuf, start * 4, a); }
  setScores(scores) { this.device.queue.writeBuffer(this.scoreBuf, 0, scores); }
  setHighlight(ids, kinds) { this.hlCount = Math.min(ids.length, 65536); if (this.hlCount) { this.device.queue.writeBuffer(this.hlIds, 0, Uint32Array.from(ids.slice(0, this.hlCount))); this.device.queue.writeBuffer(this.hlKinds, 0, Uint32Array.from(kinds.slice(0, this.hlCount))); } }
  fitTo(xy) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < xy.length; i += 2) { x0 = Math.min(x0, xy[i]); x1 = Math.max(x1, xy[i]); y0 = Math.min(y0, xy[i + 1]); y1 = Math.max(y1, xy[i + 1]); }
    this.view.cx = (x0 + x1) / 2; this.view.cy = (y0 + y1) / 2; this.fitExtent = Math.max(x1 - x0, y1 - y0); this.resize();
    this.view.scale = 0.9 * Math.min(this.canvas.width, this.canvas.height) / this.fitExtent;
  }
  resize() {
    const dpr = window.devicePixelRatio || 1, w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; this.densityTex = null; }
    if (!this.densityTex) {
      this.densityTex = this.device.createTexture({ size: [Math.max(1, w >> 1), Math.max(1, h >> 1)], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.bg.comp = this.device.createBindGroup({ layout: this.pComp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniComp } }, { binding: 1, resource: this.densityTex.createView() }, { binding: 2, resource: this.sampler }] });
    }
  }
  writeUniforms(buf, style) {
    const dv = new DataView(new ArrayBuffer(128)), v = this.view, dpr = window.devicePixelRatio || 1;
    dv.setFloat32(0, v.cx, true); dv.setFloat32(4, v.cy, true); dv.setFloat32(8, this.canvas.width, true); dv.setFloat32(12, this.canvas.height, true);
    dv.setFloat32(16, v.scale, true); dv.setFloat32(20, this.pointPx * dpr, true); dv.setUint32(24, this.mode, true); dv.setUint32(28, this.data.nBase, true);
    dv.setUint32(32, style, true); dv.setUint32(36, this.flushStart, true); dv.setFloat32(40, this.dimBase, true); dv.setFloat32(44, this.curTime, true);
    dv.setFloat32(48, this.tMin, true); dv.setFloat32(52, this.tMax, true); dv.setFloat32(56, this.recentWindow, true); dv.setFloat32(60, this.settle, true);
    dv.setFloat32(64, this.threshold, true); dv.setFloat32(68, this.growthWindow, true); dv.setFloat32(72, this.densityScale, true); dv.setFloat32(76, this.fitMid, true); dv.setInt32(80, this.focusRegion, true); dv.setFloat32(84, this.ghostAlpha, true); dv.setFloat32(88, this.arrivalBoost, true); dv.setFloat32(92, this.animNow, true); dv.setFloat32(96, this.rate, true); dv.setFloat32(100, this.regPalOff, true); dv.setFloat32(104, this.prevPalOff, true); dv.setFloat32(108, this.rmix, true); dv.setFloat32(112, this.heatOn ? 1 : 0, true); dv.setFloat32(116, this.heatPalOff, true); dv.setFloat32(120, this.heatPrevPalOff, true);
    this.device.queue.writeBuffer(buf, 0, dv.buffer);
  }
  // true while the GPU is FRAMES_AHEAD frames behind: callers that queue their own GPU work per frame (tune mode) wait too
  get gpuBusy() { return this.inFlight >= FRAMES_AHEAD; }
  draw(nHist, firstRecent = nHist) {
    if (this.gpuBusy) { this.skipped++; this.redrawDue = true; return; }
    this.resize(); const nowS = performance.now() / 1000; this.heatMoving = this.tweenHeat(this.animNow ? Math.min(0.5, nowS - this.animNow) : 0); this.animNow = nowS; this.writeUniforms(this.uni, 0); this.writeUniforms(this.uniRing, 2); this.writeUniforms(this.uniGlow, 3); this.writeUniforms(this.uniCore, 4);
    this.device.queue.writeBuffer(this.uniComp, 0, new Float32Array([this.densityScale, 0, 0, 0]));
    const enc = this.device.createCommandEncoder();
    const tsNames = [], tsRead = this.gpuTiming && this.tsSet ? (this.tsFree.pop() || this.device.createBuffer({ size: 16 * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })) : null;
    const tw = (name) => { if (!tsRead) return undefined; const i = tsNames.length; tsNames.push(name); return { querySet: this.tsSet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 }; };
    if (this.growth) {
      const p = enc.beginRenderPass({ timestampWrites: tw('growth splat'), colorAttachments: [{ view: this.densityTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
      p.setPipeline(this.pSplat); p.setBindGroup(0, this.bg.splat); p.draw(6, nHist); p.end();
    }
    const tints = this.regions && this.bg.tint && this.regionAlpha > 0;
    if (tints) { this.device.queue.writeBuffer(this.uniRegion, 16, new Float32Array([this.regionAlpha]));
      const p = enc.beginRenderPass({ timestampWrites: tw('tint splat'), colorAttachments: [{ view: this.tintTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
      p.setPipeline(this.pTint); p.setBindGroup(0, this.bg.tint); p.setIndexBuffer(this.quadIdx, 'uint16'); p.drawIndexed(6, this.data.nBase); p.end();
      if (this.tintSoft >= 1) {   // edge softness: separable Gaussian blur of the field, sigma = softness in grid texels
        const h = enc.beginRenderPass({ timestampWrites: tw('blur across'), colorAttachments: [{ view: this.tintTex2.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] }); h.setPipeline(this.pBlur); h.setBindGroup(0, this.bg.blurH); h.draw(3); h.end();
        const v = enc.beginRenderPass({ timestampWrites: tw('blur down'), colorAttachments: [{ view: this.tintTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] }); v.setPipeline(this.pBlur); v.setBindGroup(0, this.bg.blurV); v.draw(3); v.end(); } }
    const pass = enc.beginRenderPass({ timestampWrites: tw('screen'), colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), clearValue: { r: 0.043, g: 0.055, b: 0.075, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    if (tints) { pass.setPipeline(this.pRegion); pass.setBindGroup(0, this.bg.region); pass.draw(6); }
    pass.setPipeline(this.pPoints); pass.setBindGroup(0, this.bg.points); pass.draw(6, nHist);
    if (this.growth) { pass.setPipeline(this.pComp); pass.setBindGroup(0, this.bg.comp); pass.draw(3); }
    if (this.showStranded) { pass.setPipeline(this.pPoints); pass.setBindGroup(0, this.bg.ring); pass.draw(6, nHist); }
    const nFresh = nHist - firstRecent;
    if (nFresh > 0 && this.arrivalBoost > 0) { pass.setPipeline(this.pGlow); pass.setBindGroup(0, this.bg.glow); pass.draw(6, nFresh, 0, firstRecent); pass.setPipeline(this.pCore); pass.setBindGroup(0, this.bg.core); pass.draw(6, nFresh, 0, firstRecent); }
    if (this.hlCount) { pass.setPipeline(this.pHighlight); pass.setBindGroup(0, this.bg.hl); pass.draw(6, this.hlCount); }
    pass.end();
    if (tsRead) { const m = tsNames.length * 2; enc.resolveQuerySet(this.tsSet, 0, m, this.tsResolve, 0); enc.copyBufferToBuffer(this.tsResolve, 0, tsRead, 0, m * 8); }
    this.device.queue.submit([enc.finish()]); this.inFlight++;
    this.device.queue.onSubmittedWorkDone().then(() => { this.inFlight--; if (!this.redrawDue || this.gpuBusy) return; this.redrawDue = false;
      let done = false; const go = () => { if (!done) { done = true; if (this.onRedraw) this.onRedraw(); } }; requestAnimationFrame(go); setTimeout(go, 50); });
    if (tsRead) tsRead.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigUint64Array(tsRead.getMappedRange()), rec = { total: 0 };
      tsNames.forEach((name, i) => { const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6; rec[name] = ms; rec.total += ms; });
      rec.span = Number(t[2 * tsNames.length - 1] - t[0]) / 1e6; tsRead.unmap(); this.tsFree.push(tsRead); this.gpuStats.push(rec); if (this.gpuStats.length > 600) this.gpuStats.shift();
    }).catch(() => {});
  }
}
