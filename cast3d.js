// 3D view of the live map ("3D" in the header).
//
// Entering is not a jump: every row keeps its 2D x and y exactly and only its depth is fitted (a UMAP layout in which x and
// y are held fixed: attraction and repulsion move rows along z only), so seen from above the 3D map is the 2D map. Depth is
// shown as the camera tilts (it fades in over the first 12 degrees of tilt), and a short nudge tilts and returns once to
// show it is there. Arrivals take a depth the way Batched RevAdopt places them in 2D (the fuzzy-weighted mean of their 15
// history neighbours, Jacobi rounds with the base fixed), on the GPU, again whenever rows arrive.
//
// A static dataset's map (every row fitted at once) gets a full 3D UMAP fit by default: all three coordinates of every row,
// from the 2D map plus a little depth, no leash. ?lift=depth gives the depth-only lift instead.
// Opt-in elsewhere, ?lift=free (or cast.liftMode = 'free' before entering): all three coordinates are fitted, each row's x and y
// leashed to its 2D place (a pull of LEASH x the offset per epoch, scaled by the learning rate), so the top view stays
// close to the 2D map but not identical; groups the 2D map laid on top of each other can then move apart sideways as well
// as in depth. Arrivals keep their own 2D place plus the mean shift of their history neighbours (as tune mode carries
// them), and take their neighbours' mean depth. Leaving 3D from the top view returns to the 2D map unchanged.
//
// Tuning in 3D (tune mode while in 3D) changes all three coordinates: each change is computed from the 3D map tuning opened
// on (or, for a refit, from the flat 2D map), never from the previous result, and shown by a glide; arrivals follow.
//
// Leaving keeps what was done in 3D: the map collapses along the line of sight onto the plane through the camera's target,
// and those projected positions become the live map's 2D positions, at the same place and scale on screen. From the
// starting top view with no 3D tuning the projection is the original 2D map exactly, and nothing changes.
import { LiveFit, findAB } from './livefit.js';

const K = 15, LIFT_EPOCHS = 200, Z_SHARE = 0.02, FADE_DEG = 12, LEASH = 0.2;
const LIFT_PARAM = { free: 'free', depth: 'depth' }[new URLSearchParams(location.search).get('lift')] || null;
const TOP = Math.PI / 2 - 1e-4, YAW0 = -Math.PI / 2;
const WGSL_HASH = /* wgsl */`
fn pcg(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
fn rand01(h: u32) -> f32 { return f32(pcg(h) >> 8u) / 16777216.0; }`;
// livefit.js's per-row SGD (umap-learn's rule, edges in neighbour-row order, each pull twice) in three dimensions;
// zOnly = 1 moves rows along z only (the lift)
const WGSL_SGD3 = WGSL_HASH + /* wgsl */`
struct P { n: u32, seed: u32, zOnly: u32, pad1: u32, epoch: f32, alpha: f32, a: f32, b: f32, gamma: f32, negRate: f32, wmax: f32, leash: f32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(7) var<storage, read> anchor: array<f32>;   // the flat 2D map (x, y, small depth): the leash's anchor
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> nextS: array<f32>;
@group(0) @binding(5) var<storage, read> ycur: array<f32>;
@group(0) @binding(6) var<storage, read_write> ynext: array<f32>;
fn clip4(v: vec3<f32>) -> vec3<f32> { return clamp(v, vec3<f32>(-4.0), vec3<f32>(4.0)); }
fn at(i: u32) -> vec3<f32> { return vec3<f32>(ycur[3u * i], ycur[3u * i + 1u], ycur[3u * i + 2u]); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  let msk = select(vec3<f32>(1.0), vec3<f32>(0.0, 0.0, 1.0), U.zOnly == 1u);
  var yi = at(i);
  for (var e: u32 = off[i]; e < off[i + 1u]; e = e + 1u) {
    if (nextS[e] > U.epoch) { continue; }
    nextS[e] = max(nextS[e], U.epoch) + U.wmax / w[e];
    let yj = at(idx[e]);
    for (var rep: u32 = 0u; rep < 2u; rep = rep + 1u) {
      let d = yi - yj; let d2 = dot(d, d);
      if (d2 > 0.0) { let c = -2.0 * U.a * U.b * pow(d2, U.b - 1.0) / (U.a * pow(d2, U.b) + 1.0); yi = yi + msk * clip4(c * d) * U.alpha; }
    }
    let h0 = pcg(e ^ pcg(U.seed + bitcast<u32>(U.epoch)));
    var nNeg = u32(floor(U.negRate)); if (rand01(h0) < fract(U.negRate)) { nNeg = nNeg + 1u; }
    for (var p: u32 = 0u; p < nNeg; p = p + 1u) {
      let k = pcg(h0 + 0x9e3779b9u * (p + 1u)) % U.n; if (k == i) { continue; }
      let dk = yi - at(k); let dk2 = dot(dk, dk);
      if (dk2 > 0.0) { let c = 2.0 * U.gamma * U.b / ((0.001 + dk2) * (U.a * pow(dk2, U.b) + 1.0)); yi = yi + msk * clip4(c * dk) * U.alpha; }
    }
  }
  if (U.leash > 0.0) { let a2 = vec2<f32>(anchor[3u * i], anchor[3u * i + 1u]); let pull = clamp(U.leash * (a2 - yi.xy), vec2<f32>(-4.0), vec2<f32>(4.0)) * U.alpha; yi = vec3<f32>(yi.xy + pull, yi.z); }
  ynext[3u * i] = yi.x; ynext[3u * i + 1u] = yi.y; ynext[3u * i + 2u] = yi.z;
}`;
// arrivals: one Jacobi round of the fuzzy-weighted mean of their history neighbours; zOnly = 1 sets only their depth;
// zOnly = 2 (free lift) sets depth to that mean and x, y to the row's own 2D place plus its neighbours' mean shift from theirs
const WGSL_PLACE3 = /* wgsl */`
const K: u32 = ${K}u;
struct P { nArr: u32, zOnly: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> rows: array<u32>;
@group(0) @binding(2) var<storage, read> nbr: array<i32>;
@group(0) @binding(3) var<storage, read> wts: array<f32>;
@group(0) @binding(4) var<storage, read> ycur: array<f32>;
@group(0) @binding(5) var<storage, read_write> ynext: array<f32>;
@group(0) @binding(6) var<storage, read> flat: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let a = gid.x; if (a >= U.nArr) { return; }
  let r = rows[a]; var s = vec3<f32>(0.0); var sh = vec2<f32>(0.0); var sw: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) { let j = nbr[a * K + k]; if (j < 0) { continue; } let wv = wts[a * K + k]; let jj = u32(j); let yj = vec3<f32>(ycur[3u * jj], ycur[3u * jj + 1u], ycur[3u * jj + 2u]); s = s + wv * yj; sh = sh + wv * (yj.xy - vec2<f32>(flat[3u * jj], flat[3u * jj + 1u])); sw = sw + wv; }
  if (sw > 0.0) {
    let y = s / sw; ynext[3u * r + 2u] = y.z;
    if (U.zOnly == 0u) { ynext[3u * r] = y.x; ynext[3u * r + 1u] = y.y; }
    if (U.zOnly == 2u) { ynext[3u * r] = flat[3u * r] + sh.x / sw; ynext[3u * r + 1u] = flat[3u * r + 1u] + sh.y / sw; }
  }
}`;
// the flatten: each row onto the view plane (right, up), straight into the live map's 2D position buffer
const WGSL_PROJECT = /* wgsl */`
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32, r: vec4<f32>, u: vec4<f32> };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> p3: array<f32>;
@group(0) @binding(2) var<storage, read_write> p2: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  let p = vec3<f32>(p3[3u * i], p3[3u * i + 1u], p3[3u * i + 2u]);
  p2[2u * i] = dot(p, U.r.xyz); p2[2u * i + 1u] = dot(p, U.u.xyz);
}`;
const WGSL_DOTS3 = /* wgsl */`
struct U { vp: mat4x4<f32>, viewport: vec2<f32>, pointPx: f32, flat: f32, glide: f32, collapse: f32, nBase: u32, nRows: f32, fwd: vec4<f32>, tgt: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> prev: array<f32>;
@group(0) @binding(3) var<storage, read> lab: array<u32>;
@group(0) @binding(4) var<storage, read> pal: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> alive: array<u32>;
struct O { @builtin(position) p: vec4<f32>, @location(0) c: vec4<f32>, @location(1) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> O {
  var cs = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  var o: O; if (alive[ii] == 0u) { o.p = vec4<f32>(2.0, 2.0, 2.0, 1.0); return o; }
  let c = cs[vi];
  var w = mix(vec3<f32>(prev[3u * ii], prev[3u * ii + 1u], prev[3u * ii + 2u]), vec3<f32>(pos[3u * ii], pos[3u * ii + 1u], pos[3u * ii + 2u]), u.glide);
  w.z = w.z * (1.0 - u.flat);                                     // depth shown only as the camera tilts
  w = w - u.collapse * dot(w - u.tgt.xyz, u.fwd.xyz) * u.fwd.xyz;   // the flatten: depth along the line of sight goes to zero
  var cp = u.vp * vec4<f32>(w, 1.0);
  // which dot covers which: from the top view the 2D map's order (later rows on top, as the 2D renderer paints them), so
  // nothing swaps when 3D opens; true depth takes over as depth fades in with the tilt
  let ord = 1.0 - (f32(ii) + 0.5) / max(1.0, u.nRows);
  cp.z = mix(ord, clamp(cp.z / cp.w, 0.0, 1.0), 1.0 - u.flat) * cp.w;
  let size = select(u.pointPx * 1.25, u.pointPx, ii < u.nBase);
  cp = vec4<f32>(cp.xy + c * size / u.viewport * cp.w, cp.z, cp.w);
  let col = pal[min(lab[ii], arrayLength(&pal) - 1u)];
  o.p = cp; o.c = vec4<f32>(col.rgb * select(0.85, 1.0, ii >= u.nBase), 1.0); o.uv = c; return o;
}
@fragment fn fs(i: O) -> @location(0) vec4<f32> { let r = length(i.uv); if (r > 1.0) { discard; } let edge = clamp((1.0 - r) * 4.0, 0.0, 1.0); return vec4<f32>(i.c.rgb * (0.75 + 0.25 * (1.0 - r * r)) * edge, edge); }`;

const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
const ease = (t) => t * t * (3 - 2 * t);
function perspective(fovy, aspect, near, far) { const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far); return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0]); }
function lookAt(eye, at, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], norm = (a) => { const l = Math.hypot(...a) || 1; return a.map((x) => x / l); }, dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = norm(sub(eye, at)), x = norm(cross(up, z)), y = cross(z, x);
  return { m: new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]), right: x, up: y, fwd: z.map((v) => -v) };
}
function mul(a, b) { const o = new Float32Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; }

export class Cast3D {
  // ctx: { device, engine, data, renderer (the 2D one: view, point size), state, canvas (the 3D canvas), gizmo (div), status }
  constructor(ctx) {
    this.ctx = ctx; this.on = false; this.fit = null; this.glide = 1; this.collapse = 0; this.changed = false; this.liftMode = LIFT_PARAM || (ctx.staticMap ? 'free' : 'depth'); this.leash = !LIFT_PARAM && ctx.staticMap ? 0 : LEASH;   // a static map: a full 3D fit of every row (no leash); ?lift= chooses
    const d = ctx.device, mk = (code) => d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: 'main' } });
    this.pSgd = mk(WGSL_SGD3); this.pPlace = mk(WGSL_PLACE3); this.pProject = mk(WGSL_PROJECT);
    const UB = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST; this.uSgd = d.createBuffer({ size: 48, usage: UB }); this.uPlace = d.createBuffer({ size: 16, usage: UB }); this.uProject = d.createBuffer({ size: 48, usage: UB }); this.uDraw = d.createBuffer({ size: 128, usage: UB });
    const N = ctx.data.N, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, b = (bytes) => d.createBuffer({ size: bytes, usage: S });
    this.posA = b(N * 12); this.posB = b(N * 12); this.posW = b(N * 12); this.prev = b(N * 12); this.ref = b(N * 12); this.flatStart = b(N * 12); this.labBuf = b(N * 4);
    this.gpu = ctx.canvas.getContext('webgpu'); this.format = navigator.gpu.getPreferredCanvasFormat(); this.gpu.configure({ device: d, format: this.format, alphaMode: 'premultiplied' });
    const mod = d.createShaderModule({ code: WGSL_DOTS3 });
    this.pDraw = d.createRenderPipeline({ layout: 'auto', vertex: { module: mod, entryPoint: 'vs' }, fragment: { module: mod, entryPoint: 'fs', targets: [{ format: this.format, blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] }, primitive: { topology: 'triangle-list' }, depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' } });
    this.wire(); this.gizmo();
  }
  copy(src, dst, bytes) { const enc = this.ctx.device.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, dst, 0, bytes); this.ctx.device.queue.submit([enc.finish()]); }
  palette() {
    const D = this.ctx.data, cols = D.labels.map((nm) => D.labelColors[nm] || '#888888'), pal = new Float32Array((cols.length + 1) * 4);
    cols.forEach((h, i) => pal.set([...hex(h), 1], i * 4)); pal.set([0.55, 0.58, 0.62, 1], cols.length * 4);
    if (!this.palBuf || this.palBuf.size < pal.byteLength) { this.palBuf?.destroy(); this.palBuf = this.ctx.device.createBuffer({ size: pal.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); this.gDraw = null; }
    this.ctx.device.queue.writeBuffer(this.palBuf, 0, pal); this.nLabels = cols.length;
  }
  uploadLabels() { const D = this.ctx.data, n = this.ctx.engine.n, a = new Uint32Array(n); for (let i = 0; i < n; i++) { const l = D.labelS[i]; a[i] = l < this.nLabels ? l : this.nLabels; } this.ctx.device.queue.writeBuffer(this.labBuf, 0, a); }

  // one SGD run over the base rows: from `start` (all rows) into posA's base rows; zOnly for the lift
  async runFit(start, s, { zOnly = false, lrScale = 1, leash = 0 } = {}) {
    const { device, data } = this.ctx, nb = data.nBase, G = this.fit.graph, ab = findAB(Math.min(s.minDist, s.spread), s.spread);
    this.copy(start, this.posW, this.ctx.engine.n * 12); this.fit.rewind();
    const g = device.createBindGroup({ layout: this.pSgd.getBindGroupLayout(0), entries: [this.uSgd, G.off, G.idx, G.w, G.next, this.posW, this.posB, this.flatStart].map((x, i) => ({ binding: i, resource: { buffer: x } })) });
    for (let ep = 0; ep < s.epochs; ep++) {
      const buf = new ArrayBuffer(48), u = new Uint32Array(buf), f = new Float32Array(buf), alpha = s.lr * lrScale * (s.anneal ? 1 - ep / s.epochs : 1);
      u[0] = nb; u[1] = 1 + ep; u[2] = zOnly ? 1 : 0; f[4] = ep; f[5] = alpha; f[6] = ab.a; f[7] = ab.b; f[8] = s.gamma; f[9] = s.negRate; f[10] = G.wmax; f[11] = leash;
      device.queue.writeBuffer(this.uSgd, 0, buf);
      const enc = device.createCommandEncoder(), p = enc.beginComputePass(); p.setPipeline(this.pSgd); p.setBindGroup(0, g); p.dispatchWorkgroups(Math.ceil(nb / 64)); p.end();
      enc.copyBufferToBuffer(this.posB, 0, this.posW, 0, nb * 12); device.queue.submit([enc.finish()]);
      if (ep % 50 === 49) await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();
  }
  // ---- entering: the 2D map, given depth, identical from above; a nudge shows the depth
  async enter() {
    const { device, engine, data, status } = this.ctx, nb = data.nBase, n = engine.n;
    this.on = true; this.changed = false; this.dragged = false; this.palette(); this.uploadLabels();
    const y0 = new Float32Array(data.N * 3), P = engine.pos; let x0 = Infinity, x1 = -Infinity;
    for (let i = 0; i < nb; i++) { x0 = Math.min(x0, P[2 * i]); x1 = Math.max(x1, P[2 * i]); }
    let h = 12345; const zn = (x1 - x0) * Z_SHARE;
    for (let i = 0; i < n; i++) { h = Math.imul(h ^ (i + 1), 0x85ebca6b) >>> 0; h ^= h >>> 13; y0[3 * i] = P[2 * i]; y0[3 * i + 1] = P[2 * i + 1]; y0[3 * i + 2] = ((h % 10000) / 10000 - 0.5) * zn; }
    for (const b of [this.posA, this.prev, this.flatStart]) device.queue.writeBuffer(b, 0, y0);
    this.ctx.canvas.hidden = false; this.ctx.gizmoEl.hidden = false;   // shown first: the camera is sized from the canvas
    this.cam = this.viewCam(); this.glide = 1; this.collapse = 0; this.draw();   // identical to the 2D view
    const t0 = performance.now();
    if (!this.fit) {   // a static map lends its exact 15-neighbour lists (an all-pairs search of 182,000 rows takes a minute)
      this.fit = new LiveFit(device, data.featS.subarray(0, nb * data.dim), nb, data.dim);
      if (!(this.ctx.staticMap && this.fit.useLists(engine.knnIdx, engine.knnDist, K))) { status('3D: neighbour lists of the base map…'); await this.fit.neighbours(); }
    }
    this.liftSettings = { k: K, minDist: 0.03, spread: 1.2, lr: 1, anneal: true, epochs: LIFT_EPOCHS, gamma: 1, negRate: 5 };
    await this.fit.setK(K, LIFT_EPOCHS);
    const free = this.liftMode === 'free'; this.liftFree = free;
    await this.runFit(this.flatStart, this.liftSettings, free ? { leash: this.leash } : { zOnly: true });
    this.copy(this.posW, this.posA, nb * 12); this.placeArrivals(20, free ? 'carry' : true);
    this.liftMs = performance.now() - t0; this.placedN = n; this.compN = engine.compactions;
    if (free) {   // the top view moves a little: glide from the flat map to the fit
      this.copy(this.flatStart, this.prev, n * 12); const g0 = performance.now(); this.glide = 0; this.loop();
      await new Promise((res) => { const tick = () => { const t = Math.min(1, (performance.now() - g0) / 900); this.glide = ease(t); if (t < 1) requestAnimationFrame(tick); else res(); }; requestAnimationFrame(tick); setTimeout(() => { this.glide = 1; res(); }, 3000); });
      this.glide = 1; this.copy(this.posA, this.prev, n * 12);
      status(this.leash ? `3D (free fit, x and y leashed to the 2D map, ×${this.leash}): seen from above it is close to the 2D map · drag to turn, shift-drag to move, wheel to zoom` : `3D: a full 3D fit of all ${nb.toLocaleString()} rows, from the 2D map (${(this.liftMs / 1000).toFixed(1)} s) · drag to turn, shift-drag to move, wheel to zoom`);
      await this.nudge(); return;
    }
    this.copy(this.posA, this.prev, n * 12);
    status('3D: seen from above it is the 2D map · drag to turn, shift-drag to move, wheel to zoom');
    this.loop(); await this.nudge();
  }
  async nudge() {   // tilt 20 degrees with a little turn, and back
    const c = this.cam, p0 = c.pitch, y0 = c.yaw, t0 = performance.now(), T = 1600;
    await new Promise((res) => { const tick = () => { if (!this.on || this.dragged) return res(); const t = Math.min(1, (performance.now() - t0) / T), s = Math.sin(Math.PI * t); c.pitch = p0 - 0.35 * s; c.yaw = y0 + 0.25 * s; if (t < 1) requestAnimationFrame(tick); else { c.pitch = p0; c.yaw = y0; res(); } }; requestAnimationFrame(tick); });
  }
  // arrivals: Jacobi rounds of the fuzzy-weighted mean of their history neighbours (the live map's lists, rho and sigma)
  // mode: false = x, y and depth from the neighbours; true = depth only; 'carry' = the free lift's rule (WGSL_PLACE3, zOnly 2)
  placeArrivals(rounds, zOnly = false) {
    const { device, engine, data } = this.ctx, nb = data.nBase, n = engine.n, rows = []; for (let r = nb; r < n; r++) if (engine.alive[r]) rows.push(r);
    if (!rows.length) return; const na = rows.length;
    if (this.placeN !== n) {
      const nbr = new Int32Array(na * K), wts = new Float32Array(na * K);
      rows.forEach((r, a) => { const rho = engine.rho[r], sg = engine.sigma[r] || 1; for (let k = 0; k < K; k++) { const j = engine.knnIdx[r * K + k]; nbr[a * K + k] = j >= 0 && j < n && engine.alive[j] ? j : -1; wts[a * K + k] = Math.exp(-Math.max(engine.knnDist[r * K + k] - rho, 0) / sg); } });
      const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mk = (a) => { const b = device.createBuffer({ size: Math.max(16, a.byteLength), usage: S }); device.queue.writeBuffer(b, 0, a); return b; };
      for (const b of this.placeBufs || []) b.destroy();
      this.placeBufs = [mk(Uint32Array.from(rows)), mk(nbr), mk(wts)]; this.na = na; this.placeN = n; this.uploadLabels();
      this.gPlace = device.createBindGroup({ layout: this.pPlace.getBindGroupLayout(0), entries: [this.uPlace, ...this.placeBufs, this.posA, this.posB, this.flatStart].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    }
    device.queue.writeBuffer(this.uPlace, 0, new Uint32Array([this.na, zOnly === 'carry' ? 2 : zOnly ? 1 : 0, 0, 0]));
    const enc = device.createCommandEncoder();
    for (let it = 0; it < rounds; it++) { enc.copyBufferToBuffer(this.posA, 0, this.posB, 0, n * 12); const p = enc.beginComputePass(); p.setPipeline(this.pPlace); p.setBindGroup(0, this.gPlace); p.dispatchWorkgroups(Math.ceil(this.na / 64)); p.end(); enc.copyBufferToBuffer(this.posB, 0, this.posA, 0, n * 12); }
    device.queue.submit([enc.finish()]);
  }
  // rows from..to start at their 2D x and y (depth 0) before they are placed: untuned, placement then sets only their depth
  seedRows(from, to) {
    if (to <= from) return; const P = this.ctx.engine.pos, y = new Float32Array((to - from) * 3);
    for (let i = from; i < to; i++) { y[3 * (i - from)] = P[2 * i]; y[3 * (i - from) + 1] = P[2 * i + 1]; }
    this.ctx.device.queue.writeBuffer(this.posA, from * 12, y); this.ctx.device.queue.writeBuffer(this.flatStart, from * 12, y);   // the free lift carries rows from their 2D place
  }
  loop() {
    if (!this.on) return;
    try {
      const st = this.ctx.state, E = this.ctx.engine, nb = this.ctx.data.nBase;
      if ((E.n !== this.placedN || E.compactions !== this.compN) && !st?.rebuilding && !this.busy) {
        // rows arrived, or retired rows were compacted away (row numbers moved): those arrivals take a place in 3D too,
        // and prev follows so they do not glide in
        const n = E.n, from = E.compactions !== this.compN ? nb : Math.min(this.placedN, n);
        if (from === nb) this.placeN = -1;
        this.seedRows(from, n); this.placeArrivals(20, this.changed ? false : this.liftFree ? 'carry' : true); this.placedN = n; this.compN = E.compactions;
        if (n > from) { const enc = this.ctx.device.createCommandEncoder(); enc.copyBufferToBuffer(this.posA, from * 12, this.prev, from * 12, (n - from) * 12); this.ctx.device.queue.submit([enc.finish()]); }
      }
      this.draw();
    } catch (e) { console.error('[3D] frame', e); }
    requestAnimationFrame(() => this.loop());
  }

  // ---- tuning in 3D: from the 3D map tuning opened on (move) or the flat 2D map (refit), shown by a glide
  markRef() { this.copy(this.posA, this.ref, this.ctx.engine.n * 12); this.refN = this.ctx.engine.n; }
  async retune(s, kind, exact) {
    const { engine, data } = this.ctx, n = engine.n, nb = data.nBase; this.busy = true;
    try {
      this.copy(this.posA, this.prev, n * 12);
      if (exact) { this.copy(this.ref, this.posA, this.refN * 12); }
      else {
        await this.runFit(kind === 'refit' ? this.flatStart : this.ref, s, { lrScale: kind === 'move' ? s.restart : 1 });
        this.copy(this.posW, this.posA, nb * 12);
      }
      this.placeArrivals(20, false); this.changed = true; this.placedN = engine.n;
      const t0 = performance.now(); this.glide = 0;
      await new Promise((res) => { const tick = () => { const t = Math.min(1, (performance.now() - t0) / 320); this.glide = 1 - (1 - t) ** 3; if (t < 1) requestAnimationFrame(tick); else res(); }; requestAnimationFrame(tick); setTimeout(() => { this.glide = 1; res(); }, 1500); });
    } finally { this.glide = 1; this.busy = false; }
  }
  async setK(k, epochs) { await this.fit.ensureK(k, () => this.ctx.status('3D: neighbour lists for a larger k…')); await this.fit.setK(k, epochs); }

  // ---- leaving: collapse along the line of sight; the projection becomes the 2D map (unless nothing changed from the top view)
  atTopView() { const c = this.cam; return Math.abs(c.pitch - TOP) < 1e-3 && Math.abs(((c.yaw - YAW0 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI) < 1e-3; }
  async exit() {
    if (!this.on) return { changed: false };
    const { device, engine, renderer } = this.ctx, c = this.cam, cv = this.ctx.canvas, keep = this.changed || !this.atTopView();
    const scale = (cv.height / 2) / (c.dist * Math.tan(c.fov / 2)), basis = this.basis();
    if (!keep && this.liftFree) {   // free lift, top view: glide back to the 2D map, which is left unchanged
      const n = engine.n, t0 = performance.now(); this.copy(this.posA, this.prev, n * 12); this.copy(this.flatStart, this.posA, n * 12); this.glide = 0;
      await new Promise((res) => { const tick = () => { const t = Math.min(1, (performance.now() - t0) / 500); this.glide = ease(t); this.draw(); if (t < 1) requestAnimationFrame(tick); else res(); }; requestAnimationFrame(tick); setTimeout(res, 2000); });
      this.glide = 1;
    }
    if (keep) {   // the map flattens along the line of sight, onto the plane through the target
      const t0 = performance.now();
      await new Promise((res) => { const tick = () => { const t = Math.min(1, (performance.now() - t0) / 700); this.collapse = ease(t); this.draw(); if (t < 1) requestAnimationFrame(tick); else res(); }; requestAnimationFrame(tick); setTimeout(res, 3000); });
      const m = engine.n, buf = new ArrayBuffer(48), u = new Uint32Array(buf), f = new Float32Array(buf); u[0] = m; f.set([...basis.right, 0], 4); f.set([...basis.up, 0], 8);
      device.queue.writeBuffer(this.uProject, 0, buf);
      const enc = device.createCommandEncoder(), p = enc.beginComputePass(); p.setPipeline(this.pProject); p.setBindGroup(0, device.createBindGroup({ layout: this.pProject.getBindGroupLayout(0), entries: [this.uProject, this.posA, engine.posBuf].map((x, i) => ({ binding: i, resource: { buffer: x } })) })); p.dispatchWorkgroups(Math.ceil(m / 64)); p.end();
      enc.copyBufferToBuffer(engine.posBuf, 0, engine.startBuf, 0, m * 8); device.queue.submit([enc.finish()]);
      const xy = await this.fit.read(engine.posBuf, m * 8); engine.pos.set(xy); engine.start.set(xy);
      for (let i = 0; i < m; i++) if (engine.anchored[i]) { engine.anchor[2 * i] = xy[2 * i]; engine.anchor[2 * i + 1] = xy[2 * i + 1]; }
      this.ctx.data.baseXY.set(xy.subarray(0, this.ctx.data.nBase * 2));
    }
    const T = c.target, dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    renderer.view.cx = keep ? dot(T, basis.right) : T[0]; renderer.view.cy = keep ? dot(T, basis.up) : T[1]; renderer.view.scale = scale;   // the same place and scale on screen
    this.on = false; this.collapse = 0; cv.hidden = true; this.ctx.gizmoEl.hidden = true;
    return { changed: keep };
  }

  // ---- camera: the 2D view as a top-down perspective camera (same centre, same pixels per unit at z = 0)
  viewCam() { this.resize(); const v = this.ctx.renderer.view, cv = this.ctx.canvas, fov = 40 * Math.PI / 180; return { yaw: YAW0, pitch: TOP, dist: (cv.height / 2 / v.scale) / Math.tan(fov / 2), target: [v.cx, v.cy, 0], fov }; }
  basis() { const c = this.cam, eye = [c.target[0] + c.dist * Math.cos(c.pitch) * Math.cos(c.yaw), c.target[1] + c.dist * Math.cos(c.pitch) * Math.sin(c.yaw), c.target[2] + c.dist * Math.sin(c.pitch)]; return { eye, ...lookAt(eye, c.target, [0, 0, 1]) }; }
  async topView() {   // back to the starting orientation (the 2D map from above), keeping the zoom and centre
    const c = this.cam, from = { pitch: c.pitch, yaw: c.yaw }, toYaw = YAW0 + 2 * Math.PI * Math.round((c.yaw - YAW0) / (2 * Math.PI)), t0 = performance.now();
    await new Promise((res) => { const tick = () => { const t = Math.min(1, (performance.now() - t0) / 600), e = ease(t); c.pitch = from.pitch + (TOP - from.pitch) * e; c.yaw = from.yaw + (toYaw - from.yaw) * e; if (t < 1) requestAnimationFrame(tick); else { c.yaw = YAW0; c.target[2] = 0; res(); } }; requestAnimationFrame(tick); });
  }
  resize() { const cv = this.ctx.canvas, dpr = devicePixelRatio || 1, r = cv.getBoundingClientRect(), w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr)); if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; this.depth?.destroy(); this.depth = null; } if (!this.depth) this.depth = this.ctx.device.createTexture({ size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT }); }
  draw() {
    const { device, engine, data, renderer } = this.ctx, cv = this.ctx.canvas; this.resize(); const c = this.cam, B = this.basis();
    const vp = mul(perspective(c.fov, cv.width / cv.height, c.dist * 0.01, c.dist * 20), B.m);
    const tilt = (TOP - c.pitch) * 180 / Math.PI, flat = 1 - ease(Math.min(1, Math.max(0, tilt / FADE_DEG)));   // depth fades in over the first degrees of tilt
    const buf = new ArrayBuffer(128), f = new Float32Array(buf), u = new Uint32Array(buf); f.set(vp, 0); f[16] = cv.width; f[17] = cv.height; f[18] = Math.max(2, renderer.pointPx * (devicePixelRatio || 1)); f[19] = flat; f[20] = this.glide; f[21] = this.collapse; u[22] = data.nBase; f[23] = engine.n;
    f.set([...B.fwd, 0], 24); f.set([...c.target, 0], 28);
    device.queue.writeBuffer(this.uDraw, 0, buf);
    if (!this.gDraw) this.gDraw = device.createBindGroup({ layout: this.pDraw.getBindGroupLayout(0), entries: [this.uDraw, this.posA, this.prev, this.labBuf, this.palBuf, engine.aliveBuf].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const enc = device.createCommandEncoder(), pass = enc.beginRenderPass({ colorAttachments: [{ view: this.gpu.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.043, g: 0.055, b: 0.075, a: 1 } }], depthStencilAttachment: { view: this.depth.createView(), depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' } });
    pass.setPipeline(this.pDraw); pass.setBindGroup(0, this.gDraw); pass.draw(6, engine.n); pass.end(); device.queue.submit([enc.finish()]);
    this.drawGizmo(B);
  }
  // ---- orbit widget: the three axes as the camera sees them; Top view returns to the starting orientation
  gizmo() {
    const el = this.ctx.gizmoEl; el.innerHTML = '<svg viewBox="-40 -40 80 80" width="80" height="80"><circle r="36" fill="rgba(11,14,19,.7)" stroke="#243040"/><g id="g3axes"></g></svg><button id="g3top">Top view</button>';
    el.querySelector('#g3top').addEventListener('click', () => this.topView());
  }
  drawGizmo(B) {
    const g = this.ctx.gizmoEl.querySelector('#g3axes'); if (!g) return; const ax = [['x', [1, 0, 0], '#e15759'], ['y', [0, 1, 0], '#59a14f'], ['z', [0, 0, 1], '#4e79a7']];
    g.innerHTML = ax.map(([nm, v, col]) => { const x = 28 * (v[0] * B.right[0] + v[1] * B.right[1] + v[2] * B.right[2]), y = -28 * (v[0] * B.up[0] + v[1] * B.up[1] + v[2] * B.up[2]); return `<line x1="0" y1="0" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${col}" stroke-width="2"/><text x="${(x * 1.15).toFixed(1)}" y="${(y * 1.15 + 3).toFixed(1)}" fill="${col}" font-size="9" text-anchor="middle">${nm}</text>`; }).join('');
  }
  wire() {
    const cv = this.ctx.canvas; let drag = null;
    cv.addEventListener('pointerdown', (e) => { this.dragged = true; drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2, cam: { ...this.cam, target: [...this.cam.target] } }; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', (e) => {
      if (!drag || !this.on) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y, c = this.cam;
      if (!drag.pan) { c.yaw = drag.cam.yaw - dx * 0.006; c.pitch = Math.max(-TOP, Math.min(TOP, drag.cam.pitch + dy * 0.006)); }
      else { const B = this.basis(), s = (drag.cam.dist * Math.tan(c.fov / 2) * 2) / cv.clientHeight; c.target = [0, 1, 2].map((k) => drag.cam.target[k] - dx * s * B.right[k] + dy * s * B.up[k]); }
    });
    cv.addEventListener('pointerup', () => { drag = null; }); cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('wheel', (e) => { e.preventDefault(); if (this.on) this.cam.dist *= Math.exp(e.deltaY * 0.0015); }, { passive: false });
  }
  async positions() { return this.fit.read(this.posA, this.ctx.engine.n * 12); }
}
