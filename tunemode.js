// Tune mode of the live map: the base map's UMAP settings change while the live map stays on screen, with the stream.
// The base rows are laid out by livefit.js on the same GPU device, in the background; the screen follows each change as it
// is computed (WGSL_FOLLOW, drawn from its own buffer) and the live map's positions change only at the hand-over at its end;
// each arrival is carried along by its nearest base points on the opening map (a GPU kernel), so the whole map moves as
// one sheet. A placement flush waits only for the hand-over (milliseconds); regions and chips are off while tuning; Done
// tuning runs one NodeGardener quiet rebuild (base rows leashed at their tuned places, arrivals free), none runs during tuning.
//
// What changes and what does not. Only positions change here: Batched RevAdopt placement and the quiet rebuilds keep the
// live map's 15-neighbour lists and its a, b (spread 1.2, min distance 0.03) and repulsion, so tuning shapes the map but
// not the rules arrivals are placed by. With no tuning nothing in this module runs.
import { LiveFit, epochParamsFor, abFor, startFor, tunedFromParam } from './livefit.js';

const NN = 6;   // base points that carry an arrival
// a slider change waits until the slider has paused this long, then runs once to the value it landed on: restarting the
// computation every 120 ms during a drag moved dots toward the values passed on the way and back again (on a min distance
// drag 0.03 -> 0.4 on visualization publications, 5.4 % of dots overshot their end point by more than a quarter of their
// move; waiting 150 ms: 1.4 %, 250 ms: 0.3 %)
const QUIET_MS = 200;
const WARM_SHARE = 0.1;   // ASRS, min distance drag with the 200 ms wait: dots overshooting by over a quarter of their move 2.3 % -> 1.5 %, the map still makes 93 % of its move
// a move job is computed with the dots held still, then the screen glides straight to its result (finish the work, then
// animate). Following the job as it ran retraced its path from the opening map on every change: on a slow drag (a step
// every 0.4 s, visualization publications) 11-23 % of dots moved backwards by over a quarter of their move on spread and
// 20-25 % on min distance; held: 0.5 % and 4 %, median 0, and what is left is the maps themselves (the same setting gives
// the same map to the last bit; neighbouring settings really do place some dots back and forth). A job takes about 0.25 s here
const HOLD_MOVE = true;
const HOLD_BUDGET_MS = 36;   // GPU time a frame for a held job (dots still or gliding): at 12 ms a change took 1.2 s on ASRS, at 36 ms 0.65 s with the longest frame 19 ms (48 ms: 0.5 s, but 34 ms frames on visualization publications)
const SETTLE_LR = 0.03;   // keep settling: the constant learning rate, as a share of the fit's
const SETTLE_TAU = 20;    // keep settling: seconds over which a row's wander is pulled back toward the fitted map
// keep settling's pull: every row moves a share k of the way back toward its place on the fitted map, once a frame, so
// wander levels off instead of growing; how far a row strays then reflects how loosely its neighbours hold it
const WGSL_PULL = /* wgsl */`
struct U { n: u32, k: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> home: array<vec2<f32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= u.n) { return; }
  pos[i] = pos[i] + u.k * (home[i] - pos[i]);
}`;
const PRESETS = {
  umap: (n) => ({ optimizer: 'sgd', init: 'spectral', k: 15, lr: 1, anneal: true, epochs: n > 10000 ? 200 : 500, gamma: 1, negRate: 5, minDist: 0.1, spread: 1 }),
  page: () => ({ optimizer: 'adam', init: 'pca', k: 15, lr: 0.1, anneal: true, epochs: 1500, gamma: 6, negRate: 5, minDist: 0.03, spread: 1.2 }),
};

// For each arrival: its NN nearest base rows on the map now, inverse-square-distance weights, and its position now.
const WGSL_CARRY_NN = /* wgsl */`
const NN: u32 = ${NN}u;
struct P { nArr: u32, nBase: u32, eps2: f32, pad: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> nIdx: array<u32>;
@group(0) @binding(4) var<storage, read_write> nW: array<f32>;
@group(0) @binding(5) var<storage, read_write> aStart: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let a = gid.x; if (a >= U.nArr) { return; }
  let r = rows[a]; let p = vec2<f32>(pos[2u * r], pos[2u * r + 1u]);
  var bd: array<f32, NN>; var bi: array<u32, NN>;
  for (var k: u32 = 0u; k < NN; k = k + 1u) { bd[k] = 3.0e38; bi[k] = 0u; }
  for (var j: u32 = 0u; j < U.nBase; j = j + 1u) {
    let d = p - vec2<f32>(pos[2u * j], pos[2u * j + 1u]); let d2 = dot(d, d);
    if (d2 < bd[NN - 1u]) { var k = NN - 1u; loop { if (k == 0u) { break; } if (bd[k - 1u] <= d2) { break; } bd[k] = bd[k - 1u]; bi[k] = bi[k - 1u]; k = k - 1u; } bd[k] = d2; bi[k] = j; }
  }
  var s: f32 = 0.0; for (var k: u32 = 0u; k < NN; k = k + 1u) { s = s + 1.0 / (bd[k] + U.eps2); }
  for (var k: u32 = 0u; k < NN; k = k + 1u) { nIdx[a * NN + k] = bi[k]; nW[a * NN + k] = (1.0 / (bd[k] + U.eps2)) / s; }
  aStart[2u * a] = p.x; aStart[2u * a + 1u] = p.y;
}`;
// arrival = its position at the job's start + the weighted displacement of its base points since then
const WGSL_CARRY = /* wgsl */`
const NN: u32 = ${NN}u;
struct P { nArr: u32, nBase: u32, eps2: f32, pad: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read_write> pos: array<f32>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
@group(0) @binding(3) var<storage, read> nIdx: array<u32>;
@group(0) @binding(4) var<storage, read> nW: array<f32>;
@group(0) @binding(5) var<storage, read> aStart: array<f32>;
@group(0) @binding(6) var<storage, read> bStart: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let a = gid.x; if (a >= U.nArr) { return; }
  var d = vec2<f32>(0.0, 0.0);
  for (var k: u32 = 0u; k < NN; k = k + 1u) { let j = nIdx[a * NN + k]; d = d + nW[a * NN + k] * (vec2<f32>(pos[2u * j], pos[2u * j + 1u]) - vec2<f32>(bStart[2u * j], bStart[2u * j + 1u])); }
  let r = rows[a]; pos[2u * r] = aStart[2u * a] + d.x; pos[2u * r + 1u] = aStart[2u * a + 1u] + d.y;
}`;

// What is on screen while a change is computed: each base row follows the computation as it runs. Its target is the job's
// positions now, turned and shifted onto the opening map (the same pinning finish() applies, from alignSums' totals), plus
// what is left of the offset the screen had when the job began (from - ref), fading out as the job's epochs run; so a
// job starts exactly where the screen is, ends exactly on its result, and a new change mid-way never pulls the map back.
// A critically damped spring follows the target (no jump in position or speed). hold = 1 (a refit, whose early epochs
// are its start positions): stay put until the result.
const WGSL_FOLLOW = /* wgsl */`
struct P { n: u32, hold: u32, w: f32, om: f32, dt: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> s: array<f32>;   // alignSums: mean opening map, mean job, Sxx, Sxy, Syx, Syy
@group(0) @binding(2) var<storage, read> job: array<f32>;
@group(0) @binding(3) var<storage, read> opening: array<f32>;
@group(0) @binding(4) var<storage, read_write> startAt: array<f32>;
@group(0) @binding(5) var<storage, read_write> ease: array<f32>;
@group(0) @binding(6) var<storage, read_write> vel: array<f32>;
@group(0) @binding(7) var<storage, read_write> show: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  var x = vec2<f32>(ease[2u * i], ease[2u * i + 1u]); var v = vec2<f32>(vel[2u * i], vel[2u * i + 1u]);
  // hold = 2 (the first frame of a hold): stop where the row coasts to from its speed (x + v / om), so a row gliding when
  // a new change begins slows to a stop without passing the stop point and springing back
  if (U.hold == 2u) { let c = x + v / U.om; startAt[2u * i] = c.x; startAt[2u * i + 1u] = c.y; }
  let f = vec2<f32>(startAt[2u * i], startAt[2u * i + 1u]);
  var t = f;
  if (U.hold == 0u) {
    let th = atan2(s[5] - s[6], s[4] + s[7]); let c = cos(th); let sn = sin(th);
    let b = vec2<f32>(job[2u * i] - s[2], job[2u * i + 1u] - s[3]);
    let al = vec2<f32>(s[0] + c * b.x - sn * b.y, s[1] + sn * b.x + c * b.y); let o = vec2<f32>(opening[2u * i], opening[2u * i + 1u]);
    // the screen's offset at the job's start fades as this row's job actually carries it the same way (by the time the job
    // has moved the row as far along the offset as the offset reached, none of it is left), and in any case by the job's
    // end: the job's movement and the old offset never add up, so a slider drag (a new job every 120 ms, each from the
    // opening map) cannot overshoot; sideways jitter of the job's first epochs does not count, so nothing slides back either
    let off = f - o; let dl2 = dot(off, off); var g: f32 = 1.0 - U.w;
    if (dl2 > 0.0) { g = min(g, clamp(1.0 - dot(al - o, off) / dl2, 0.0, 1.0)); }
    t = al + g * off;
  }
  v = v + (U.om * U.om * (t - x) - 2.0 * U.om * v) * U.dt; x = x + v * U.dt;
  ease[2u * i] = x.x; ease[2u * i + 1u] = x.y; vel[2u * i] = v.x; vel[2u * i + 1u] = v.y; show[2u * i] = x.x; show[2u * i + 1u] = x.y;
}`;
const FOLLOW_OMEGA = 10;   // spring rate (1/s): lags a moving target by about 0.2 s, settles onto a still one in about 0.6 s

export class TuneMode {
  // ctx: { device, engine, data, renderer, state, drawFrame, status, rebuildRegions(), rebuild(reason) -> Promise, panel helpers via $ }
  constructor(ctx) {
    this.ctx = ctx; this.active = false; this.fit = null; this.P = { restart: 0.2 }; this.mode = 'move'; this.S = { job: null, pending: null, inflight: false, busy: false, chunk: 20, e0: 0, startedXY: null };
    const d = ctx.device, mk = (code) => d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: 'main' } });
    this.pNN = mk(WGSL_CARRY_NN); this.pCarry = mk(WGSL_CARRY); this.pFollow = mk(WGSL_FOLLOW); this.pPull = mk(WGSL_PULL); this.uniP = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.uni = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniF = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  get nBase() { return this.ctx.data.nBase; }
  // ---- entering: the base rows exactly where they are on screen, nothing moves until a setting changes
  async enter() {
    if (this.ctx.cast?.on) return this.enter3();
    const { engine, status } = this.ctx, n = this.nBase;
    await this.settleOff(); await this.ensureFit();
    if (this.ctx.staticMap && !this.prefetched) { this.prefetched = true; setTimeout(() => this.fit?.prefetchLists(), 1000); }   // the full lists (any k up to 64) in the background
    this.syncFromEngine(); this.S.startedXY = engine.pos.slice(0, n * 2); this.fit.markReference(); this.S.startKey = this.key(); this.S.startP = { ...this.P };   // every change is computed from this map
    this.setupCarry();   // arrivals are carried by their nearest base points on the opening map
    this.active = true; this.ctx.state.tuneMode = true; this.ctx.shrinkRegions(); this.show(); if (!this.looping) this.loop();   // the regions and chips shrink away
    status('Tuning the base map: each change moves the map; arrivals move with it');
  }
  // the base map's own LiveFit (neighbour lists, graph and settings of the live map's fit), made once
  async ensureFit(quiet = false) {
    const { device, data, engine } = this.ctx, n = this.nBase, status = quiet ? () => {} : this.ctx.status;
    if (!this.fit) {
      status('Tune: neighbour lists of the base map…');
      this.fit = new LiveFit(device, data.featS.subarray(0, n * data.dim), n, data.dim);
      // a static map lends its exact 15-neighbour lists, so tuning starts at once; the full lists (any k up to 64) follow in
      // the background, about 5 s of GPU work on 182,000 rows in short pieces, so k is usually ready by the time it is moved
      const shipped = this.ctx.state.shippedLists;   // lists that came with the dataset (arXiv): k up to theirs, never searched
      if (shipped && shipped.n === n && this.fit.useLists(shipped.idx, shipped.dist, shipped.k)) { this.fit.noSearch = true; this.ctx.$('tu-k').max = shipped.k; }
      else if (!(this.ctx.staticMap && this.fit.useLists(engine.knnIdx, engine.knnDist, 15))) await this.fit.neighbours();
      Object.assign(this.P, PRESETS.page(n));
      const s = data.baseMeta?.handoff?.settings; if (s) { try { Object.assign(this.P, tunedFromParam(s)); } catch { /* keep the live map's fit settings */ } }
      this.AB = abFor(this.P); await this.fit.ensureK(this.P.k, () => status('Tune: neighbour lists for a larger k…')); await this.fit.setK(this.P.k, this.P.epochs);
    }
  }
  // ---- keep settling: once the fit is done the optimizer keeps running on the GPU at a small constant learning rate
  // (SETTLE_LR of the fit's), so the map never freezes: rows held firmly by their neighbours barely move, loosely held
  // rows (bridges between groups) keep wandering. Every half second the map is pinned back onto the map it started from
  // (turn and shift only), so it cannot drift or spin as a whole. Only on a map with no arrivals (a static map, or a live
  // map before Play); it stops for anything else that moves rows (Play, tuning, 3D, a re-layout, a glide) and the CPU
  // copy of the positions then catches up once.
  canSettle() { const { state, engine, cast } = this.ctx; return !this.active && !this.d3 && !cast?.on && !state.playing && !state.rebuilding && !state.morphing && !state.catchingUp && engine.n === this.nBase && !!this.ctx.data?.baseXY; }
  async settleOn() {
    if (this.idle || this.idleStarting || !this.canSettle()) return; this.idleStarting = true;
    try {
      await this.ensureFit(true); if (!this.canSettle()) return;
      this.syncFromEngine(); this.fit.markReference(); this.fit.rewind(this.P.optimizer === 'sgd');
      this.idle = { e0: this.fit.epoch, chunk: 1, aligning: false, lastAlign: performance.now(), lastPull: performance.now(), inflight: false,
        pull: this.ctx.device.createBindGroup({ layout: this.pPull.getBindGroupLayout(0), entries: [this.uniP, this.fit.posBuf, this.fit.refBuf].map((buffer, binding) => ({ binding, resource: { buffer } })) }) };
      this.ctx.renderer.setPosOverride(this.fit.posBuf); if (!this.looping) this.loop();
    } finally { this.idleStarting = false; }
  }
  async settleOff() {
    const I = this.idle; if (!I) return; this.idle = null;
    while (I.inflight || I.aligning) await new Promise((r) => setTimeout(r, 4));
    await this.fit.alignTo(this.fit.refBuf, { scale: false, reflect: false });
    this.pushToEngine(); this.ctx.renderer.setPosOverride(null); await this.pullFromEngine();   // the settled map is the map from now on
  }
  idleTick() {
    const I = this.idle; if (!this.canSettle()) { this.settleOff(); return; }
    if (I.inflight || I.aligning) return;
    if (performance.now() - I.lastAlign > 500) { I.aligning = true; this.fit.alignTo(this.fit.refBuf, { scale: false, reflect: false }).finally(() => { I.aligning = false; I.lastAlign = performance.now(); }); return; }
    const P = this.P, f = this.fit, t0 = performance.now(), m = I.chunk;
    for (let s = 0; s < m; s++) { const p = epochParamsFor({ ...P, lr: P.lr * (this.ctx.state.settleLr || SETTLE_LR), anneal: false }, this.AB, f.epoch - I.e0); if (P.optimizer === 'sgd') f.sgdEpoch(p); else f.adamStep(p); }
    { const now = performance.now(), k = 1 - Math.exp(-Math.min(0.1, (now - I.lastPull) / 1000) / SETTLE_TAU); I.lastPull = now; const d = this.ctx.device;
      d.queue.writeBuffer(this.uniP, 0, new Float32Array([0, k, 0, 0])); d.queue.writeBuffer(this.uniP, 0, new Uint32Array([this.nBase]));
      const enc = d.createCommandEncoder(), pass = enc.beginComputePass(); pass.setPipeline(this.pPull); pass.setBindGroup(0, I.pull); pass.dispatchWorkgroups(Math.ceil(this.nBase / 64)); pass.end(); d.queue.submit([enc.finish()]); }
    I.inflight = true;   // a few milliseconds of GPU work a frame: the chunk adapts to about 4 ms
    this.ctx.device.queue.onSubmittedWorkDone().then(() => { const ms = performance.now() - t0; I.chunk = Math.max(1, Math.min(20, Math.round(m * 4 / Math.max(0.5, ms)))); }).finally(() => { I.inflight = false; });
  }
  async exit() {
    if (this.d3) return this.exit3();
    this.active = false;
    while (this.S.job || this.S.pending || this.S.inflight || this.S.busy || this.S.finishing) { this.S.pending = null; if (this.S.job) this.S.job.stop = true; await new Promise((r) => setTimeout(r, 10)); }
    if (this.F?.on) {   // a change stopped part-way: the screen settles back onto the map last handed over, then the engine draws again
      const enc = this.ctx.device.createCommandEncoder(); enc.copyBufferToBuffer(this.ctx.engine.posBuf, 0, this.fit.posBuf, 0, this.nBase * 8); this.ctx.device.queue.submit([enc.finish()]);
      this.F.hold = false; this.F.until = Math.max(this.F.until, performance.now() + 900); if (!this.looping) this.loop();
      while (this.F.on) await new Promise((r) => setTimeout(r, 16));
    }
    this.freeCarry(); this.ctx.state.tuneMode = false; await this.settleUp();
  }
  // ---- tuning in 3D (cast3d.js does the computing): its own settings, starting from the lift's; each change computed from
  // the 3D map tuning opened on, glided in; leaving 3D afterwards flattens the result into the 2D map
  async enter3() {
    const cast = this.ctx.cast; this.d3 = true; this.P2 = this.P;
    if (!this.P3) this.P3 = { ...cast.liftSettings, optimizer: 'sgd', init: 'spectral', restart: 0.2 };
    this.P = this.P3; this.AB = abFor(this.P);
    if (cast.fit.graph.k !== this.P.k || cast.fit.graph.epochs !== this.P.epochs) await cast.setK(this.P.k, this.P.epochs);
    cast.markRef(); this.S.startKey = this.key(); this.S.startP = { ...this.P };
    this.active = true; this.ctx.state.tuneMode = true; this.show(); if (!this.looping) this.loop();
    this.ctx.status('Tuning the 3D map: each change is computed from this 3D map; 2D afterwards keeps the result');
  }
  async exit3() {
    this.active = false; while (this.S.pending || this.S.busy3 || this.S.kBusy) { this.S.pending = null; await new Promise((r) => setTimeout(r, 10)); }
    this.ctx.state.tuneMode = false; this.P3 = this.P; this.P = this.P2; this.AB = abFor(this.P); this.d3 = false;
  }
  tick3() {
    const S = this.S, cast = this.ctx.cast;
    if (!S.busy3 && !S.kBusy && S.pending) {
      const kind = S.pending === 'refit' ? 'refit' : 'move', exact = this.key() === S.startKey; S.pending = null; S.moved = true; this.everMoved3 = true; S.busy3 = true; const t0 = performance.now();
      cast.retune(this.P, kind, exact).then(() => { S.lastMs = performance.now() - t0; }).catch((e) => this.ctx.status('3D tuning failed: ' + (e.message || e), true)).finally(() => { S.busy3 = false; });
    }
    this.showState();
  }
  // fit positions = the live map's base rows now (after any flush or rebuild in between)
  syncFromEngine() { const n = this.nBase, enc = this.ctx.device.createCommandEncoder(); enc.copyBufferToBuffer(this.ctx.engine.posBuf, 0, this.fit.posBuf, 0, n * 8); enc.copyBufferToBuffer(this.ctx.engine.posBuf, 0, this.fit.nextPosBuf, 0, n * 8); this.ctx.device.queue.submit([enc.finish()]); }

  changed(restart = false) { if (!this.active) return; this.S.pending = restart || this.mode === 'refit' ? 'refit' : 'move'; this.S.lastInput = performance.now(); }
  async setK(k) { if (this.S.kBusy) { this.S.kPending = k; return; } this.S.kBusy = true; const G = () => (this.d3 ? this.ctx.cast.fit.graph : this.fit.graph); try { while (this.S.inflight || this.S.busy || this.S.busy3) await new Promise((r) => setTimeout(r, 2)); if (this.d3) await this.ctx.cast.setK(k, this.P.epochs); else { await this.fit.ensureK(k, () => this.ctx.status('Tune: neighbour lists for a larger k…')); await this.fit.setK(k, this.P.epochs); } this.changed(); } finally { this.S.kBusy = false; } if (this.S.kPending != null && (this.S.kPending !== G().k || this.P.epochs !== G().epochs)) { const k2 = this.S.kPending; this.S.kPending = null; await this.setK(k2); } else this.S.kPending = null; }

  // ---- carrying arrivals: rows alive after the base, their base points now
  setupCarry() {
    const { engine, device } = this.ctx, n = this.nBase, rows = []; for (let r = n; r < engine.n; r++) if (engine.alive[r]) rows.push(r);
    this.freeCarry(); this.carryN = engine.n; if (this.F) this.F.carryGroup = null; if (!rows.length) return;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, b = (bytes) => device.createBuffer({ size: Math.max(16, bytes), usage: S });
    const c = { n: rows.length, rows: b(rows.length * 4), idx: b(rows.length * NN * 4), w: b(rows.length * NN * 4), start: b(rows.length * 8), bStart: b(n * 8) };
    device.queue.writeBuffer(c.rows, 0, Uint32Array.from(rows));
    let w2 = 0; { const p = engine.pos; let x0 = Infinity, x1 = -Infinity; for (let i = 0; i < n; i++) { x0 = Math.min(x0, p[2 * i]); x1 = Math.max(x1, p[2 * i]); } w2 = ((x1 - x0) * 1e-3) ** 2 || 1e-8; }
    device.queue.writeBuffer(this.uni, 0, new Uint32Array([rows.length, n, 0, 0])); device.queue.writeBuffer(this.uni, 8, new Float32Array([w2]));
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(engine.posBuf, 0, c.bStart, 0, n * 8);
    const p = enc.beginComputePass(); p.setPipeline(this.pNN); p.setBindGroup(0, device.createBindGroup({ layout: this.pNN.getBindGroupLayout(0), entries: [this.uni, engine.posBuf, c.rows, c.idx, c.w, c.start].map((x, i) => ({ binding: i, resource: { buffer: x } })) })); p.dispatchWorkgroups(Math.ceil(rows.length / 64)); p.end();
    device.queue.submit([enc.finish()]);
    c.group = device.createBindGroup({ layout: this.pCarry.getBindGroupLayout(0), entries: [this.uni, engine.posBuf, c.rows, c.idx, c.w, c.start, c.bStart].map((x, i) => ({ binding: i, resource: { buffer: x } })) });
    this.carry = c;
  }
  freeCarry() { const c = this.carry; if (!c) return; for (const k of ['rows', 'idx', 'w', 'start', 'bStart']) c[k].destroy(); this.carry = null; }
  // the tuned base into the live map's buffer, arrivals carried along
  pushToEngine() {
    const { engine, device } = this.ctx, n = this.nBase, enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.fit.posBuf, 0, engine.posBuf, 0, n * 8);
    if (this.carry) { const p = enc.beginComputePass(); p.setPipeline(this.pCarry); p.setBindGroup(0, this.carry.group); p.dispatchWorkgroups(Math.ceil(this.carry.n / 64)); p.end(); }
    device.queue.submit([enc.finish()]);
  }
  // the CPU copy of positions the live map's placement, regions and rebuilds read, and the leash anchors, follow the GPU
  async pullFromEngine() {
    const { engine, data } = this.ctx, m = engine.n, xy = await this.fit.read(engine.posBuf, m * 8);
    engine.pos.set(xy.subarray(0, m * 2)); for (let i = 0; i < m; i++) if (engine.anchored[i]) { engine.anchor[2 * i] = xy[2 * i]; engine.anchor[2 * i + 1] = xy[2 * i + 1]; }
    data.baseXY.set(xy.subarray(0, this.nBase * 2));   // the tuned base is the base map from now on (Reset to base returns here)
  }

  // ---- jobs: move (from the map on screen, pinned against drift) or refit (from the start, aligned, then a glide)
  // the base is moving only while a chunk is applied; a placement flush (main.js) waits on tuneWait, then goes before the next chunk
  setBusy(b) { const st = this.ctx.state; if (b === !!st.tuneBusy) return; st.tuneBusy = b; if (b) st.tuneWait = new Promise((r) => { this.release = r; }); else { this.release?.(); this.release = null; } }
  key() { const P = this.P; return [P.k, P.minDist, P.spread, P.optimizer, P.init, P.lr, P.anneal, P.epochs, P.gamma, P.negRate, P.restart].join('|'); }
  // A change is computed in the background from the map tune mode opened on (move: warm start from it with the learning rate
  // restarted at a fraction; refit: from the start positions), never from the previous result, so each setting gives one
  // map: sweeping a slider back and forth cannot ratchet the map outward or inward, and the opening settings give back the
  // opening map exactly. The result is shown by a glide from what is on screen.
  async begin(kind) {
    const S = this.S, st = this.ctx.state; S.busy = true;
    try {
      while (S.inflight || st.rebuilding) await new Promise((r) => setTimeout(r, 2));
      S.job = null;
      this.followOn(); { const enc = this.ctx.device.createCommandEncoder(); enc.copyBufferToBuffer(S.glideOn ? this.ctx.engine.posBuf : this.F.ease, 0, this.F.from, 0, this.nBase * 8); this.ctx.device.queue.submit([enc.finish()]); }   // the job starts where the screen is
      // held: the screen keeps gliding to the map finished last (the live map's positions), or else coasts to a stop
      this.F.hold = kind === 'refit' || (this.holdMove ?? HOLD_MOVE); this.F.coast = this.F.hold && !S.glideOn; this.F.until = Infinity; S.glideOn = false;
      this.fit.setPositions(kind === 'refit' ? await startFor(this.fit, this.P.init) : S.startedXY);
      this.fit.rewind(this.P.optimizer === 'sgd'); S.e0 = 0; S.job = { kind, key: this.key(), ab: { ...this.AB }, lrScale: kind === 'move' ? this.P.restart : 1, t0: performance.now() };
      // the opening settings give the opening map exactly (finish() puts it back): no epochs to run, whose drift away from
      // it and back would show as a bounce; the screen settles straight onto it
      if (S.job.key === S.startKey) S.e0 = this.fit.epoch - this.P.epochs;
    } finally { S.busy = false; }
  }
  // warm: the learning rate eases in over the job's first share of epochs (Adam's first steps are full-size whatever the
  // gradient, and their momentum carried rows past their end point)
  step(m, lrScale) { const f = this.fit, P = this.P, warm = (this.warmShare ?? WARM_SHARE) * P.epochs; for (let s = 0; s < m; s++) { const done = f.epoch - this.S.e0, p = epochParamsFor({ ...P, lr: P.lr * lrScale }, this.AB, done); if (warm > 0 && done < warm) p.alpha *= (done + 1) / warm; if (P.optimizer === 'sgd') f.sgdEpoch(p); else f.adamStep(p); } }
  // the running job's map handed to the live map without ending the job (anneal off): pinned to the opening map, then pushed
  async handOver(job) {
    this.setBusy(true);
    try {
      await this.fit.alignTo(this.fit.refBuf, job.kind === 'refit' ? {} : { scale: false, reflect: false });
      if (!this.carry || this.carryN !== this.ctx.engine.n) this.setupCarry();
      this.pushToEngine(); await this.pullFromEngine(); this.S.handedAB = job.ab;
      if (this.F) this.F.hold = false;
    } finally { this.setBusy(false); }
  }
  async finish() {
    const S = this.S, job = S.job; S.job = null;
    if (!job.stop) {
      const { engine } = this.ctx;
      this.setBusy(true);   // a placement flush waits only for this hand-over (milliseconds)
      try {
        if (job.key === S.startKey) this.fit.setPositions(S.startedXY);   // the opening settings: the opening map, exactly
        else await this.fit.alignTo(this.fit.refBuf, job.kind === 'refit' ? {} : { scale: false, reflect: false });   // pinned to the opening map (a refit may also turn, mirror and scale)
        if (this.F) { this.F.hold = false; this.F.until = performance.now() + 1500; }   // the screen settles onto the result (a refit glides to it from where it held)
        if (!this.carry || this.carryN !== engine.n) this.setupCarry();   // rows placed since: carried from the map handed over last
        this.pushToEngine(); await this.pullFromEngine(); S.handedAB = job.ab; S.glideOn = true;   // the curve the map on screen was fitted with
        if (!this.demoOn) this.ctx.status(job.key === S.startKey ? 'Clusters moved back: the opening settings give the opening map' : `Clusters moved: UMAP re-run with your new settings (${job.kind === 'move' ? 'from the opening map' : 'from scratch'})`);
      } finally { this.setBusy(false); }
    }
    else if (this.F) this.F.until = performance.now();   // stopped (Done tuning): exit() settles the screen
    S.lastMs = performance.now() - job.t0; this.showState();
  }
  // ---- the screen while tuning (WGSL_FOLLOW): base rows follow the job, arrivals are carried by them, other rows as placed
  followOn() {
    const { device, engine, renderer, data } = this.ctx, n = this.nBase;
    if (!this.F) {
      const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, b = (bytes) => device.createBuffer({ size: Math.max(16, bytes), usage: S });
      const F = this.F = { on: false, hold: false, until: 0, last: 0, sums: b(64), from: b(n * 8), ease: b(n * 8), vel: b(n * 8), show: b(data.N * 8) };
      F.sumGroup = this.fit.bind('alignSums', [this.fit.uni.align, this.fit.refBuf, this.fit.posBuf, F.sums]);
      F.group = device.createBindGroup({ layout: this.pFollow.getBindGroupLayout(0), entries: [this.uniF, F.sums, this.fit.posBuf, this.fit.refBuf, F.from, F.ease, F.vel, F.show].map((x, i) => ({ binding: i, resource: { buffer: x } })) });
    }
    const F = this.F; if (F.on) return;
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(engine.posBuf, 0, F.ease, 0, n * 8); enc.clearBuffer(F.vel); enc.copyBufferToBuffer(engine.posBuf, 0, F.show, 0, engine.n * 8); device.queue.submit([enc.finish()]);
    F.on = true; F.last = performance.now(); renderer.setPosOverride(F.show);
  }
  followOff() { if (!this.F?.on) return; this.F.on = false; this.ctx.renderer.setPosOverride(null); }
  followStep() {
    const F = this.F, { device, engine } = this.ctx, n = this.nBase, now = performance.now(), dt = Math.min(1 / 30, Math.max(0, (now - F.last) / 1000)); F.last = now;
    if (!this.S.job && !this.S.finishing && !this.S.busy && now > F.until) { this.followOff(); return; }
    // how far the job has come, for the backstop that clears the screen's old offset by the job's end whatever the job did:
    // a warm-started move covers most of its way early (on ASRS, after a quarter of the epochs 38 % of the way is left,
    // after half 15 %), so 1 - (1 - share of epochs)^3; the offset itself fades by the job's own movement (WGSL_FOLLOW).
    // A later backstop (x^4) let leftover offsets add up on a slider drag (8 % overshoot); this one leaves only a dip of
    // about 1.5 % of the map's size at the start of a drag
    const j = this.S.job, x = j ? Math.min(1, Math.max(0, (this.fit.epoch - this.S.e0) / this.P.epochs)) : 1, w = 1 - (1 - x) ** 3;
    const buf = new ArrayBuffer(32), u = new Uint32Array(buf), f = new Float32Array(buf); u[0] = n; u[1] = F.hold ? (F.coast ? 2 : 1) : 0; F.coast = false; f[2] = w; f[3] = FOLLOW_OMEGA; f[4] = dt;
    device.queue.writeBuffer(this.uniF, 0, buf); device.queue.writeBuffer(this.fit.uni.align, 0, new Uint32Array([n, 0, 0, 0]));
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(engine.posBuf, 0, F.show, 0, engine.n * 8);
    const p = enc.beginComputePass();
    if (!F.hold) { p.setPipeline(this.fit.pipes.alignSums); p.setBindGroup(0, F.sumGroup); p.dispatchWorkgroups(1); }
    p.setPipeline(this.pFollow); p.setBindGroup(0, F.group); p.dispatchWorkgroups(Math.ceil(n / 64));
    if (this.carry) {
      const c = this.carry; if (!F.carryGroup) F.carryGroup = device.createBindGroup({ layout: this.pCarry.getBindGroupLayout(0), entries: [this.uni, F.show, c.rows, c.idx, c.w, c.start, c.bStart].map((x, i) => ({ binding: i, resource: { buffer: x } })) });
      p.setPipeline(this.pCarry); p.setBindGroup(0, F.carryGroup); p.dispatchWorkgroups(Math.ceil(c.n / 64));
    }
    p.end(); device.queue.submit([enc.finish()]);
  }
  // Done tuning: one quiet rebuild (if anything has arrived) settles arrivals around the tuned base and brings regions,
  // map lists and readouts up to date; none runs while tuning, where arrivals are carried with the base instead
  async settleUp() {
    if (!this.S.moved) { this.ctx.rebuildRegions(); return; }
    this.S.moved = false;
    // the live map keeps the tuned curve (min distance, spread): re-layouts and placements from now on use the settings the
    // map was fitted with; with the old curve the first re-layout pulled the clusters back toward their old tightness
    if (this.S.handedAB) { this.ctx.data.a = this.S.handedAB.a; this.ctx.data.b = this.S.handedAB.b; }
    if (this.ctx.engine.n > this.nBase) await this.ctx.rebuild('tuned').catch((e) => this.ctx.status('Rebuild failed: ' + (e.message || e), true));
    else this.ctx.rebuildRegions();
  }
  loop() { if (!this.active && !this.S.job && !this.F?.on && !this.idle) { this.looping = false; return; } this.looping = true; this.tick(); requestAnimationFrame(() => this.loop()); }
  tick() {
    if (this.ctx.renderer.gpuBusy) return;   // the GPU is two frames behind: queue nothing more this frame (job step, follow spring, draw)
    if (this.idle) { this.idleTick(); this.ctx.drawFrame(); return; }
    if (this.d3) return this.tick3();
    const S = this.S, st = this.ctx.state;
    if (this.fit?.graph && !S.busy && !S.inflight && !S.finishing && !S.kBusy && !st.rebuilding) {
      // while a slider is dragged, each job runs at least 120 ms before the next replaces it, so the screen keeps moving toward the new setting
      // held (HOLD_MOVE): no pause wait and no restarts; a running job finishes and is shown, then the latest setting runs,
      // so a drag shows a stream of finished maps (anneal off: the job never finishes, so a change replaces it)
      const hold = this.holdMove ?? HOLD_MOVE, now = performance.now(), quiet = this.quietMs ?? (hold ? 0 : QUIET_MS);
      const free = !S.job || (hold ? !this.P.anneal : now - S.job.t0 >= (this.minJobMs ?? 120));
      if (S.pending && now - (S.lastInput || 0) >= quiet && free) { const k = S.pending; S.pending = null; S.moved = true; this.everMoved = true; this.begin(k); }
      else if (S.job) {
        // Anneal off: the learning rate never falls, so the job never ends; past its epochs it keeps going (the map stays
        // unsettled) and hands the map over every 2 s, so Done tuning keeps what is on screen
        const forever = !this.P.anneal && !S.job.stop, left = forever ? Infinity : this.P.epochs - (this.fit.epoch - S.e0);
        if (forever && this.fit.epoch - S.e0 >= this.P.epochs && performance.now() - (S.job.handed || S.job.t0) > 2000 && !S.handing) {
          S.handing = true; S.job.handed = performance.now(); this.handOver(S.job).finally(() => { S.handing = false; });
        }
        if (left <= 0 || S.job.stop) { S.finishing = true; this.finish().finally(() => { S.finishing = false; }); }
        else {   // in the background: the live map's positions do not change until the hand-over, so placement runs freely
          const m = Math.min(left, S.chunk), t0 = performance.now(); this.step(m, S.job.lrScale); S.inflight = true;
          const budget = this.F?.hold ? (this.holdBudget ?? HOLD_BUDGET_MS) : 12;
          this.ctx.device.queue.onSubmittedWorkDone().then(() => { const ms = performance.now() - t0; S.chunk = Math.max(2, Math.min(4000, Math.round(m * budget / Math.max(0.5, ms)))); }).finally(() => { S.inflight = false; });
        }
      }
    }
    if (this.F?.on) this.followStep();
    this.ctx.drawFrame(); this.showState();
  }

  // ---- panel (index.html #tune)
  show() { const $ = this.ctx.$; $('tune').hidden = false; if (!this.wired) this.wire(); this.showValues(); }
  hide() { this.ctx.$('tune').hidden = true; }
  showValues() {
    const $ = this.ctx.$, P = this.P, f = { k: String, epochs: String, gamma: (v) => (+v).toFixed(1), negRate: String, minDist: (v) => (+v).toFixed(2), spread: (v) => (+v).toFixed(2), restart: (v) => `×${(+v).toFixed(2)}` };
    for (const k of Object.keys(f)) { $('tu-' + k).value = P[k]; $('tu-' + k + 'v').textContent = f[k](P[k]); }
    $('tu-lr').value = Math.log10(P.lr); $('tu-lrv').textContent = P.lr >= 0.1 ? P.lr.toFixed(2) : P.lr.toPrecision(2); $('tu-anneal').checked = P.anneal;
    $('tu-optimizer').value = P.optimizer; $('tu-init').value = P.init; $('tu-optimizer').disabled = $('tu-init').disabled = !!this.d3;   // 3D: SGD, refits start from the flat 2D map
    $('tu-mode').value = this.mode; $('tu-restartRow').hidden = this.mode !== 'move';
    $('tu-ab').textContent = `a ${this.AB.a.toFixed(3)} · b ${this.AB.b.toFixed(3)}`;
  }
  // "Show me": moves the sliders through a short tour, each step waiting until its map has settled on screen
  settled() { const S = this.S; return !S.job && !S.finishing && !S.busy && !S.busy3 && !this.F?.on; }
  async demo() {
    const $ = this.ctx.$, P = this.P, k0 = P.k, md0 = P.minDist, sp0 = P.spread, steps = [
      ['k', 5, 'Neighbours 5: each item listens to only its 5 nearest, so the map breaks into many small, tight groups.'],
      ['k', 50, 'Neighbours 50: each item listens to 50, so groups merge into broader regions.'],
      ['k', k0, `Neighbours back to ${k0}.`],
      ['minDist', 0.6, 'Min distance 0.6: similar items may no longer sit close, so clumps loosen into an even cloud.'],
      ['minDist', md0, `Min distance back to ${md0}: tight clumps again.`],
      ['spread', 2.5, 'Spread 2.5: the groups scale out.'],
      ['spread', sp0, `Spread back to ${sp0}. Now try the sliders yourself.`]];
    this.demoOn = true; const btn = $('tu-demo'), cap = $('tu-demoCap'); btn.textContent = 'Stop'; cap.hidden = false;
    const wait = async (ms) => { const t = performance.now() + ms; while (this.demoOn && performance.now() < t) await new Promise((r) => setTimeout(r, 50)); };
    let done = false;
    try {
      for (const [key, v, text] of steps) {
        if (!this.demoOn || !this.active) break;
        // the control lights up first and the caption says what will happen; then it moves, and the result is held
        const el = $('tu-' + key), row = el.closest('.ctl'), help = row?.nextElementSibling;
        for (const x of document.querySelectorAll('#tune .demo-hl')) x.classList.remove('demo-hl');
        row?.classList.add('demo-hl'); if (help?.classList.contains('tu-help')) help.classList.add('demo-hl'); row?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        cap.textContent = text; await wait(1600); if (!this.demoOn) break;
        el.value = v; el.dispatchEvent(new Event('input'));
        await wait(300); while (this.demoOn && this.active && !this.settled()) await new Promise((r) => setTimeout(r, 100));
        await wait(3000);
      }
      done = this.demoOn;
    } finally { this.demoStop(done); }
  }
  demoStop(keepCaption = false) { const $ = this.ctx.$; this.demoOn = false; for (const x of document.querySelectorAll('#tune .demo-hl')) x.classList.remove('demo-hl'); $('tu-demo').textContent = 'Show me'; $('tu-demoCap').hidden = !keepCaption; }
  showState() {
    const $ = this.ctx.$, S = this.S, j = S.job; if (!$('tune') || $('tune').hidden) return;
    if (this.d3) { $('tu-state').textContent = S.busy3 ? '3D · moving…' : S.lastMs ? `3D · done in ${S.lastMs.toFixed(0)} ms` : '3D · ready'; return; }
    $('tu-state').textContent = j ? (!this.P.anneal && this.fit.epoch - S.e0 >= this.P.epochs ? `no anneal: still moving · step ${this.fit.epoch - S.e0}` : `${j.kind === 'move' ? 'moving' : 'refitting'} · ${Math.max(0, this.fit.epoch - S.e0)} / ${this.P.epochs}`) : S.lastMs ? `done in ${S.lastMs.toFixed(0)} ms` : 'ready';
  }
  wire() {
    this.wired = true; const $ = this.ctx.$, P = this.P, custom = () => { $('tu-preset').value = 'custom'; };
    const on = (key, parse, after) => $('tu-' + key).addEventListener('input', (e) => { P[key] = parse(e.target.value); custom(); after(); this.showValues(); });
    on('k', (v) => +v, () => this.setK(P.k)); on('epochs', (v) => +v, () => this.setK(P.k));
    on('lr', (v) => Math.pow(10, +v), () => this.changed()); on('gamma', (v) => +v, () => this.changed()); on('negRate', (v) => +v, () => this.changed());
    on('minDist', (v) => +v, () => { this.AB = abFor(P); this.changed(); }); on('spread', (v) => +v, () => { this.AB = abFor(P); this.changed(); });
    $('tu-restart').addEventListener('input', (e) => { P.restart = +e.target.value; this.showValues(); });
    $('tu-anneal').addEventListener('change', (e) => { P.anneal = e.target.checked; custom(); this.changed(); });
    $('tu-optimizer').addEventListener('change', (e) => { P.optimizer = e.target.value; custom(); this.changed(); });
    $('tu-init').addEventListener('change', (e) => { P.init = e.target.value; custom(); this.changed(true); });
    $('tu-mode').addEventListener('change', (e) => { this.mode = e.target.value; this.showValues(); });
    $('tu-preset').addEventListener('change', async (e) => { if (e.target.value === 'custom') return; const was = P.k, r = P.restart; Object.assign(P, PRESETS[e.target.value](this.nBase)); P.restart = r; this.AB = abFor(P); this.showValues(); if (P.k !== was) await this.setK(P.k); else this.changed(); });
    $('tu-demo').addEventListener('click', () => (this.demoOn ? this.demoStop() : this.demo()));
    // any change by hand stops the demo (the demo's own slider moves are untrusted events)
    for (const ev of ['input', 'change']) $('tune').addEventListener(ev, (e) => { if (e.isTrusted && this.demoOn) this.demoStop(); }, true);
    // Undo: the opening settings again, which give back the opening map exactly (finish() recognises them)
    $('tu-undo').addEventListener('click', async () => {
      const k0 = P.k, e0 = P.epochs; Object.assign(P, this.S.startP); this.AB = abFor(P); $('tu-preset').value = 'custom'; this.showValues();
      if (P.k !== k0 || P.epochs !== e0) await this.setK(P.k); else this.changed();
    });
  }
}
