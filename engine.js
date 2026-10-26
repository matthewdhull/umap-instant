// Streaming engine for the Jetstream demo: batched reverse-adopter placement onto a frozen map, one
// flush per tick of stream time (or per websocket batch when live). GPU (WebGPU) does the exact search,
// the adopter radius test, the fuzzy weights and the Jacobi placement; the CPU does neighbour-list
// bookkeeping (merges, evictions, incident rows) and label inheritance.
import { K, TOPK_THREADS, WGSL_DISTANCE, WGSL_TOPK_PARTIAL, WGSL_TOPK_MERGE, WGSL_WEIGHTS, WGSL_JACOBI, WGSL_MEAN, WGSL_RANKS, WGSL_RANKS_GRID, RANK_CAP, WGSL_REBUILD } from './kernels.js';
import { exactLists } from './livefit.js';

const QTILE = 64;                 // queries per distance tile (d2 scratch = QTILE x N floats)
const ADOPTER_CAP = 1 << 20;      // adopter candidate entries per flush (16 MB)
export const JACOBI_ITERS = 20, JACOBI_TOL = 1e-6;
export const FLUSH_CAP = 2048;    // arrivals per flush; a larger backlog is placed in consecutive flushes
const EMPTY = 0xffffffff, UNLABELLED = 0xffff;
const RANKS_GRID_MIN = 262144;   // maps this big score through the grid (K7g); smaller ones keep the pass over every row (K7)

// Load a capture (relay output + derived base). extraCapacity reserves rows for live arrivals.
export async function loadData(captureUrl, baseName, onProgress, extraCapacity = 0) {
  const dd = captureUrl + 'derived/', bd = dd + baseName + '/';
  const j = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error('fetch failed: ' + u); return r.json(); };
  const b = async (u, T) => { const r = await fetch(u); if (!r.ok) throw new Error('fetch failed: ' + u); return new T(await r.arrayBuffer()); };
  onProgress?.('loading the per-post table');
  const [rows, base] = await Promise.all([j(dd + 'rows_meta.json'), j(bd + 'base_meta.json')]);
  const [t, reply, posts, baseXY, bIdx, bDist, bRho, bSigma, bLabel] = await Promise.all([
    b(dd + 't_f32.bin', Float32Array), b(dd + 'reply_u8.bin', Uint8Array), j(dd + 'posts.json'),
    b(bd + 'base_xy_f32.bin', Float32Array), b(bd + 'base_knn_idx_i32.bin', Int32Array), b(bd + 'base_knn_dist_f32.bin', Float32Array),
    b(bd + 'base_rho_f32.bin', Float32Array), b(bd + 'base_sigma_f32.bin', Float32Array), b(bd + 'base_label_u16.bin', Uint16Array)]);
  const nCap = rows.n, dim = base.dim || rows.dim, vecFile = base.vectors_file || 'vectors.f32.bin';
  onProgress?.(`loading ${dim}-dimension post vectors (${(nCap * dim * 4 / 1048576).toFixed(0)} MB, ${vecFile})`);
  // A base may list the capture rows in its own page order (base rows first, then the stream by time) and may leave
  // rows out; without an order file, page row = capture row.
  const [feat, order] = await Promise.all([b(captureUrl + vecFile, Float32Array), base.order_file ? b(bd + base.order_file, Uint32Array) : null]);
  if (feat.length < nCap * dim) throw new Error('vector file is shorter than the per-post table');
  const n = order ? order.length : nCap, N = n + extraCapacity;
  const featS = new Float32Array(N * dim), tS = new Float32Array(N).fill(Infinity), wallS = new Float32Array(N), replyS = new Uint8Array(N);
  let postsP;
  if (order) {
    postsP = new Array(n);
    for (let i = 0; i < n; i++) { const o = order[i]; featS.set(feat.subarray(o * dim, o * dim + dim), i * dim); tS[i] = t[o]; wallS[i] = t[o]; replyS[i] = reply[o]; postsP[i] = posts[o]; }
  } else { featS.set(feat.subarray(0, n * dim)); tS.set(t.subarray(0, n)); wallS.set(t.subarray(0, n)); replyS.set(reply.subarray(0, n)); postsP = posts.slice(0, n); }
  // wallS = true seconds since the capture start (stream time collapses the gaps between live sessions)
  const labelS = new Uint16Array(N).fill(UNLABELLED); labelS.set(bLabel.subarray(0, base.n_base));
  return { N, n, nCapture: nCap, dim, nBase: base.n_base, encoder: base.encoder || rows.encoder || 'minilm', model: base.model || rows.model, k: base.k, a: base.a, b: base.b, baseName, baseMeta: base, rowsMeta: rows, t0us: rows.t0_us,
           labels: base.labels, labelColors: base.label_colors, labelTerms: base.label_terms, baseSeconds: base.base_seconds,
           tS, wallS, labelS, replyS, posts: postsP, featS, baseXY, bIdx, bDist, bRho, bSigma };
}


// ---- small CPU helpers for the in-browser base fit
export function mulberry32(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
// First two principal directions of the centred vectors by power iteration (25 rounds, deflated), with the standard
// deviation of the projections, so other rows can be projected onto the same axes later (the loading preview).
export function pcaAxes(feat, n, dim) {
  const mean = new Float64Array(dim); for (let i = 0; i < n; i++) for (let d = 0; d < dim; d++) mean[d] += feat[i * dim + d]; for (let d = 0; d < dim; d++) mean[d] /= n;
  const rng = mulberry32(7), comps = [], sds = [];
  for (let c = 0; c < 2; c++) {
    let v = new Float64Array(dim); for (let d = 0; d < dim; d++) v[d] = rng() - 0.5;
    const u = new Float64Array(n);
    for (let it = 0; it < 25; it++) {
      for (let i = 0; i < n; i++) { let acc = 0; const b = i * dim; for (let d = 0; d < dim; d++) acc += (feat[b + d] - mean[d]) * v[d]; u[i] = acc; }
      const w = new Float64Array(dim); for (let i = 0; i < n; i++) { const b = i * dim, ui = u[i]; for (let d = 0; d < dim; d++) w[d] += (feat[b + d] - mean[d]) * ui; }
      if (c === 1) { const p = comps[0]; let dot = 0; for (let d = 0; d < dim; d++) dot += w[d] * p[d]; for (let d = 0; d < dim; d++) w[d] -= dot * p[d]; }
      let nrm = 0; for (let d = 0; d < dim; d++) nrm += w[d] * w[d]; nrm = Math.sqrt(nrm) || 1; for (let d = 0; d < dim; d++) v[d] = w[d] / nrm;
    }
    comps.push(v); let m = 0, m2 = 0; for (let i = 0; i < n; i++) { m += u[i]; m2 += u[i] * u[i]; } m /= n; sds.push(Math.sqrt(Math.max(1e-12, m2 / n - m * m)));
  }
  return { mean, comps, sds };
}
// Project rows onto the axes, scaled so the axes' own sample has standard deviation sd per axis.
export function pcaProject(feat, n, dim, axes, sd = 3.0, out = null) {
  const y = out || new Float32Array(n * 2), { mean, comps, sds } = axes;
  for (let i = 0; i < n; i++) { const b = i * dim; for (let c = 0; c < 2; c++) { const v = comps[c]; let acc = 0; for (let d = 0; d < dim; d++) acc += (feat[b + d] - mean[d]) * v[d]; y[2 * i + c] = acc / sds[c] * sd; } }
  return y;
}
export function pcaInit(feat, n, dim, sd = 3.0) { return pcaProject(feat, n, dim, pcaAxes(feat, n, dim), sd); }
// k-means (k-means++ start, seeded) on L2-normalised vectors; labels renumbered by cluster size, largest first.
// Async: yields to the page between iterations (each is ~0.2 s on 15,000 x 384) and reports progress.
export async function kmeans(feat, n, dim, k, { iters = 15, seed = 1, onProgress = null } = {}) {
  const rng = mulberry32(seed), cent = new Float64Array(k * dim), lab = new Int32Array(n), dmin = new Float64Array(n).fill(Infinity);
  const dist2 = (i, c) => { let acc = 0; const b = i * dim, cb = c * dim; for (let d = 0; d < dim; d++) { const t = feat[b + d] - cent[cb + d]; acc += t * t; } return acc; };
  let first = Math.floor(rng() * n); for (let d = 0; d < dim; d++) cent[d] = feat[first * dim + d];
  for (let c = 1; c < k; c++) {
    let total = 0; for (let i = 0; i < n; i++) { const dd = dist2(i, c - 1); if (dd < dmin[i]) dmin[i] = dd; total += dmin[i]; }
    let r = rng() * total, pick = n - 1; for (let i = 0; i < n; i++) { r -= dmin[i]; if (r <= 0) { pick = i; break; } }
    for (let d = 0; d < dim; d++) cent[c * dim + d] = feat[pick * dim + d];
  }
  const sum = new Float64Array(k * dim), cnt = new Int32Array(k);
  for (let it = 0; it < iters; it++) {
    let changed = 0;
    for (let i = 0; i < n; i++) { let best = 0, bd = Infinity; for (let c = 0; c < k; c++) { const dd = dist2(i, c); if (dd < bd) { bd = dd; best = c; } } if (lab[i] !== best) { lab[i] = best; changed++; } }
    sum.fill(0); cnt.fill(0); for (let i = 0; i < n; i++) { const c = lab[i], b = i * dim, cb = c * dim; cnt[c]++; for (let d = 0; d < dim; d++) sum[cb + d] += feat[b + d]; }
    for (let c = 0; c < k; c++) if (cnt[c]) for (let d = 0; d < dim; d++) cent[c * dim + d] = sum[c * dim + d] / cnt[c];
    onProgress?.(it + 1, iters, changed); await new Promise(r => setTimeout(r, 0));
    if (!changed && it > 0) break;
  }
  const order = [...Array(k).keys()].sort((a, b) => cnt[b] - cnt[a]); const remap = new Int32Array(k); order.forEach((c, r) => { remap[c] = r; });
  const out = new Uint16Array(n); for (let i = 0; i < n; i++) out[i] = remap[lab[i]];
  return out;
}

function makePipeline(device, code, label) {
  const module = device.createShaderModule({ code, label });
  return { module, pipeline: device.createComputePipeline({ label, layout: 'auto', compute: { module, entryPoint: 'main' } }) };
}

export class StreamingEngine {
  constructor(device, data, { noLedger = false, lazyLedger = false } = {}) {
    this.noLedger = noLedger;   // throwaway engines (base fits) need no 2-D lists
    // lazyLedger (a static map): the 2-D lists and the impostor ledger (about 10 s of GPU work at 182,000 rows) start when
    // something first needs them (placement, a rebuild), not behind the map's first glide; the result is the same
    this.lazyLedger = lazyLedger;
    this.device = device; this.data = data; const { N, dim, k } = data; this.N = N; this.dim = dim;
    if (k !== K) throw new Error('kernels are compiled for k=15');
    const d = device, S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC, U = GPUBufferUsage.UNIFORM;
    this.wMax = FLUSH_CAP;
    // features: N rows plus wMax scratch rows at the end for neighbour-list refresh queries (rows that lost a retired neighbour)
    this.featBuf = d.createBuffer({ label: 'features', size: (N + this.wMax) * dim * 4, usage: S | CD });
    d.queue.writeBuffer(this.featBuf, 0, data.featS.subarray(0, data.n * dim));
    this.alive = new Uint32Array(N).fill(1); this.aliveBuf = d.createBuffer({ label: 'alive', size: N * 4, usage: S | CD }); d.queue.writeBuffer(this.aliveBuf, 0, this.alive);
    this.qsrc = new Uint32Array(N + this.wMax); for (let i = 0; i < N + this.wMax; i++) this.qsrc[i] = i;
    this.qsrcBuf = d.createBuffer({ label: 'query source row', size: (N + this.wMax) * 4, usage: S | CD }); d.queue.writeBuffer(this.qsrcBuf, 0, this.qsrc);
    this.deadMark = new Uint8Array(N); this.deadPlaced = 0; this.retiredTotal = 0; this.compactions = 0;
    // Impostor ledger (NodeGardener v3's trigger instrument): exact 15-lists in the 2-D map, maintained like the feature lists
    // (same kernels on the position buffer with dim 2); an entry i->j is an impostor when j is a map neighbour of i but not a
    // meaning neighbour. net = entries on the ledger now; gross = entries ever created (never decremented); since = gross since the last reset.
    this.radius2DBuf = d.createBuffer({ label: 'radius2 (map)', size: N * 4, usage: S | CD });
    this.knn2Idx = new Int32Array(N * K); this.knn2Dist = new Float32Array(N * K); this.radius2D = new Float32Array(N); this.imp = new Uint8Array(N);
    this.ledger = { net: 0, gross: 0, since: 0, created: 0, removed: 0 };
    this.radius2Buf = d.createBuffer({ label: 'radius2', size: N * 4, usage: S | CD });
    this.posBuf = d.createBuffer({ label: 'positions', size: (N + this.wMax) * 8, usage: S | CD | CS });   // + scratch rows for 2-D list refreshes
    this.d2Buf = d.createBuffer({ label: 'd2 scratch', size: QTILE * N * 4, usage: S });
    this.nnBuf = d.createBuffer({ label: 'new-vs-new d2', size: this.wMax * this.wMax * 4, usage: S | CS });
    this.adopterCountBuf = d.createBuffer({ label: 'adopter count', size: 16, usage: S | CD | CS });
    this.adopterBuf = d.createBuffer({ label: 'adopters', size: ADOPTER_CAP * 16, usage: S | CS });
    this.pDistBuf = d.createBuffer({ label: 'partial dist', size: QTILE * TOPK_THREADS * K * 4, usage: S });
    this.pIdBuf = d.createBuffer({ label: 'partial id', size: QTILE * TOPK_THREADS * K * 4, usage: S });
    this.topDistBuf = d.createBuffer({ label: 'top dist', size: this.wMax * K * 4, usage: S | CS });
    this.topIdBuf = d.createBuffer({ label: 'top id', size: this.wMax * K * 4, usage: S | CS });
    const nTiles = Math.ceil(this.wMax / QTILE) * 2 + 2;
    this.paramBufs = Array.from({ length: nTiles }, (_, i) => d.createBuffer({ label: 'params ' + i, size: 32, usage: U | CD }));
    this.histBaseZero = d.createBuffer({ label: 'histBase 0', size: 16, usage: U | CD }); d.queue.writeBuffer(this.histBaseZero, 0, new Uint32Array([0, 0, 0, 0]));
    this.histBaseNew = d.createBuffer({ label: 'histBase new', size: 16, usage: U | CD });
    this.rowsBuf = d.createBuffer({ label: 'weight rows', size: 16, usage: U | CD });
    this.jpBufs = Array.from({ length: JACOBI_ITERS }, (_, i) => d.createBuffer({ label: 'jacobi params ' + i, size: 32, usage: U | CD }));
    this.traceBuf = d.createBuffer({ label: 'trace', size: 68 * 4, usage: S | CD | CS });
    this.movementBuf = d.createBuffer({ label: 'movement', size: this.wMax * 4, usage: S });
    this.curBuf = d.createBuffer({ label: 'jacobi A', size: this.wMax * 8, usage: S | CD | CS });
    this.updBuf = d.createBuffer({ label: 'jacobi B', size: this.wMax * 8, usage: S | CD | CS });
    this.grow = {}; this.staging = [];
    this.pipes = {
      dist: makePipeline(d, WGSL_DISTANCE, 'K1 distance+adopters'), partial: makePipeline(d, WGSL_TOPK_PARTIAL, 'K2 partial top15'),
      merge: makePipeline(d, WGSL_TOPK_MERGE, 'K3 merge top15'), weights: makePipeline(d, WGSL_WEIGHTS, 'K4 weights'),
      jacobi: makePipeline(d, WGSL_JACOBI, 'K5 jacobi'), mean: makePipeline(d, WGSL_MEAN, 'K6 mean movement'), ranks: makePipeline(d, WGSL_RANKS, 'K7 2-D ranks'), ranksGrid: makePipeline(d, WGSL_RANKS_GRID, 'K7g 2-D ranks on a grid'), rebuild: makePipeline(d, WGSL_REBUILD, 'K8 quiet rebuild step') };
    // quiet rebuild state: every row's anchor (its position at the last rebuild, or the base fit) and whether it is leashed
    this.anchor = new Float32Array(N * 2); this.anchored = new Uint8Array(N); this.rebuilds = 0; this.lastRebuild = null; this.placedAtRebuild = 0;
    this.rpBufs = Array.from({ length: 80 }, (_, i) => d.createBuffer({ label: 'rebuild params ' + i, size: 48, usage: U | CD }));
    this.startBuf = d.createBuffer({ label: 'start positions', size: N * 8, usage: S | CD });
    this.knnIdx = new Int32Array(N * K); this.knnDist = new Float32Array(N * K); this.radius2 = new Float32Array(N);
    this.rho = new Float32Array(N); this.sigma = new Float32Array(N); this.pos = new Float32Array(N * 2);
    this.ranks = new Uint32Array(N * K); this.start = new Float32Array(N * 2); this.settleDist = new Float32Array(N);
    this.onLabels = null;   // callback (start, count) after labels are assigned
    this.reset();
  }

  async checkShaders() {
    for (const [name, p] of Object.entries(this.pipes)) {
      const info = await p.module.getCompilationInfo();
      const errs = info.messages.filter(m => m.type === 'error');
      if (errs.length) throw new Error(`shader ${name}: ` + errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }
  }

  reset() {
    const { nBase, baseXY, bIdx, bDist, bRho, bSigma } = this.data;
    this.knnIdx.fill(-1); this.knnDist.fill(Infinity); this.radius2.fill(0); this.rho.fill(0); this.sigma.fill(0); this.pos.fill(0);
    this.knnIdx.set(bIdx); this.knnDist.set(bDist); this.rho.set(bRho); this.sigma.set(bSigma); this.pos.set(baseXY);
    for (let i = 0; i < nBase; i++) { const r = bDist[i * K + K - 1]; this.radius2[i] = r * r; }
    this.n = nBase; this.hash = 0x811c9dc5; this.flushes = 0; this.placedTotal = 0;
    this.alive.fill(1); this.device.queue.writeBuffer(this.aliveBuf, 0, this.alive); this.deadMark.fill(0); this.deadPlaced = 0; this.retiredTotal = 0;
    this.data.labelS.fill(UNLABELLED, nBase);
    this.ranks.fill(0); this.start.set(baseXY); this.settleDist.fill(0);
    this.device.queue.writeBuffer(this.posBuf, 0, this.pos); this.device.queue.writeBuffer(this.radius2Buf, 0, this.radius2); this.device.queue.writeBuffer(this.startBuf, 0, this.start);
    this.knn2Idx.fill(-1); this.knn2Dist.fill(Infinity); this.radius2D.fill(0); this.imp.fill(0); this.ledger = { net: 0, gross: 0, since: 0, created: 0, removed: 0 };
    this.anchor.fill(0); this.anchor.set(baseXY); this.anchored.fill(0); this.anchored.fill(1, 0, nBase); this.rebuilds = 0; this.lastRebuild = null; this.placedAtRebuild = 0;
    this.baseRanksReady = this.noLedger ? Promise.resolve() : this.computeRanks(0, nBase, nBase, this.data.bIdx);   // a throwaway engine's map is never scored
    this._ledgerP = this.noLedger ? Promise.resolve() : this.lazyLedger ? null : this._initLedger();
  }
  get ledgerReady() { if (!this._ledgerP) this._ledgerP = this._initLedger(); return this._ledgerP; }
  // Exact 2-D 15-lists of every base row, then the base map's own impostor count (the ledger's starting level).
  async _initLedger() {
    await this.baseRanksReady; const rows = []; for (let i = 0; i < this.n; i++) rows.push(i);
    await this._refresh(rows, { feat: this.posBuf, dim: 2, kI: this.knn2Idx, kD: this.knn2Dist, radius: this.radius2D, radiusBuf: this.radius2DBuf, weights: false });
    let net = 0; for (const i of rows) { const c = this._impostors(i).size; this.imp[i] = c; net += c; }
    this.ledger = { net, gross: 0, since: 0, created: 0, removed: 0, base: net }; return net;
  }
  breathe() { return new Promise(r => setTimeout(r, 0)); }
  // A pause for long CPU loops: call it every few hundred rows; it gives the page its turn once 8 ms have passed since the
  // last one (so the work never blocks a frame for long), and does nothing otherwise. The loops compute the same either way.
  _turner() { let t = performance.now(); return async () => { if (performance.now() - t > 8) { await this.breathe(); t = performance.now(); } }; }
  _impostors(i) { const out = new Set(); const b = i * K; for (let k = 0; k < K; k++) { const j = this.knn2Idx[b + k]; if (j < 0) continue; let meaning = false; for (let m = 0; m < K; m++) if (this.knnIdx[b + m] === j) { meaning = true; break; } if (!meaning) out.add(j); } return out; }
  // Re-count impostors for touched rows against their previous sets: created entries raise gross, the balance moves net.
  _ledgerUpdate(before) { let created = 0, removed = 0; for (const [i, old] of before) { const now = this._impostors(i); for (const j of now) if (!old.has(j)) created++; for (const j of old) if (!now.has(j)) removed++; this.imp[i] = now.size; } const L = this.ledger; L.net += created - removed; L.gross += created; L.since += created; L.created = created; L.removed = removed; }
  async _ledgerUpdateAsync(before) { let created = 0, removed = 0, c = 0; const turn = this._turner(); for (const [i, old] of before) { const now = this._impostors(i); for (const j of now) if (!old.has(j)) created++; for (const j of old) if (!now.has(j)) removed++; this.imp[i] = now.size; if ((++c & 511) === 0) await turn(); } const L = this.ledger; L.net += created - removed; L.gross += created; L.since += created; L.created = created; L.removed = removed; }
  resetLedgerSince() { this.ledger.since = 0; }

  // Rows not yet placed whose stream time is at or before `clock` (rows are in time order).
  pendingUntil(clock) { let p = this.n; const t = this.data.tS, L = this.data.n; while (p < L && t[p] <= clock) p++; return p - this.n; }

  // Live mode: append embedded posts behind the loaded rows (features to the GPU immediately).
  appendRows(vecs, times, posts, replies, walls = times) {
    const c = times.length, at = this.data.n; if (at + c > this.N) throw new Error('capacity exhausted: reload with a larger ?cap=');
    this.data.featS.set(vecs, at * this.dim); this.device.queue.writeBuffer(this.featBuf, at * this.dim * 4, vecs);
    for (let i = 0; i < c; i++) { this.data.tS[at + i] = times[i]; this.data.wallS[at + i] = walls[i]; this.data.replyS[at + i] = replies[i]; this.data.posts.push(posts[i]); }
    this.data.n += c; return at;
  }

  // ---- Capped history: retire placed arrivals (never base rows), refresh the neighbour lists that referenced them
  // exactly, and compact the row arrays once enough rows are dead so GPU memory stays bounded.
  retire(rows) {
    const q = this.device.queue, dead = [];
    for (const r of rows) if (r >= this.data.nBase && r < this.n && this.alive[r]) { this.alive[r] = 0; this.deadMark[r] = 1; this.radius2[r] = 0; dead.push(r); }
    if (!dead.length) return { dead: 0, affected: [] };
    for (const r of dead) { this.radius2D[r] = 0; this.ledger.net -= this.imp[r]; this.imp[r] = 0; q.writeBuffer(this.aliveBuf, r * 4, new Uint32Array([0])); q.writeBuffer(this.radius2Buf, r * 4, new Float32Array([0])); q.writeBuffer(this.radius2DBuf, r * 4, new Float32Array([0])); }
    this.deadPlaced += dead.length; this.retiredTotal += dead.length;
    const affected = [], kI = this.knnIdx, kI2 = this.knn2Idx, dm = this.deadMark;   // living rows whose 15-lists (either space) name a retired row
    for (let i = 0; i < this.n; i++) { if (!this.alive[i]) continue; const b = i * K; let hit = false; for (let k = 0; k < K && !hit; k++) { const j = kI[b + k], j2 = kI2[b + k]; hit = (j >= 0 && dm[j]) || (j2 >= 0 && dm[j2]); } if (hit) affected.push(i); }
    return { dead: dead.length, affected };
  }
  // Exact 15-nearest lists among living placed rows for the given rows (each row excludes itself), plus their
  // radius, rho and sigma. Uses the search kernels in refresh mode through the scratch query slots.
  async refreshLists(rows) {
    if (!rows.length) return 0;
    await this.ledgerReady;
    const before = new Map(); for (const i of rows) before.set(i, this._impostors(i));
    await this._refresh(rows, { feat: this.featBuf, dim: this.dim, kI: this.knnIdx, kD: this.knnDist, radius: this.radius2, radiusBuf: this.radius2Buf, weights: true });
    await this._refresh(rows, { feat: this.posBuf, dim: 2, kI: this.knn2Idx, kD: this.knn2Dist, radius: this.radius2D, radiusBuf: this.radius2DBuf, weights: false });
    this._ledgerUpdate(before);
    return rows.length;
  }
  // Exact 15-nearest lists among living placed rows for the given rows (each row excludes itself) in one space, plus their
  // radius and, for the feature space, rho and sigma. Uses the search kernels in refresh mode through the scratch query slots.
  async _refresh(rows, { feat, dim, kI, kD, radius, radiusBuf, weights, chunkRows = this.wMax }) {
    const d = this.device, q = d.queue, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const src = feat === this.posBuf ? this.pos : this.data.featS;
    for (let lo = 0; lo < rows.length; lo += chunkRows) {
      const chunk = rows.slice(lo, lo + chunkRows), w = chunk.length, fq = new Float32Array(w * dim), ids = new Uint32Array(w);
      chunk.forEach((r, s) => { fq.set(src.subarray(r * dim, (r + 1) * dim), s * dim); ids[s] = r; });
      q.writeBuffer(feat, this.N * dim * 4, fq); q.writeBuffer(this.qsrcBuf, this.N * 4, ids); q.writeBuffer(this.adopterCountBuf, 0, new Uint32Array(4));
      const enc = d.createCommandEncoder(); let tile = 0;
      for (let qb = 0; qb < w; qb += QTILE, tile++) {
        const qc = Math.min(QTILE, w - qb), pb = this.paramBufs[tile];
        q.writeBuffer(pb, 0, new Uint32Array([this.n, this.N + qb, qc, dim, this.N, ADOPTER_CAP, this.N, 2]));
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipes.dist.pipeline); pass.setBindGroup(0, this.bind(this.pipes.dist, [pb, feat, radiusBuf, this.d2Buf, this.adopterCountBuf, this.adopterBuf, this.histBaseZero, this.aliveBuf, this.qsrcBuf]));
        pass.dispatchWorkgroups(Math.ceil(this.n / 256), qc);
        pass.setPipeline(this.pipes.partial.pipeline); pass.setBindGroup(0, this.bind(this.pipes.partial, [pb, this.d2Buf, this.pDistBuf, this.pIdBuf])); pass.dispatchWorkgroups(qc);
        pass.setPipeline(this.pipes.merge.pipeline); pass.setBindGroup(0, this.bind(this.pipes.merge, [pb, this.pDistBuf, this.pIdBuf, this.topDistBuf, this.topIdBuf])); pass.dispatchWorkgroups(Math.ceil(qc / 64));
        pass.end();
      }
      q.submit([enc.finish()]);
      const [topDRaw, topIRaw] = await this.readMany([{ buffer: this.topDistBuf, bytes: w * K * 4 }, { buffer: this.topIdBuf, bytes: w * K * 4 }]);
      const topD = new Float32Array(topDRaw), topI = new Uint32Array(topIRaw);
      chunk.forEach((r, s) => { for (let k = 0; k < K; k++) { kD[r * K + k] = topD[s * K + k]; kI[r * K + k] = topI[s * K + k]; } const rad = topD[s * K + K - 1]; radius[r] = rad * rad; });
      if (weights) {   // rho and sigma for the refreshed rows (feature space only)
        const distBuf = this.growable('weight dists', w * K * 4, S), wBuf = this.growable('weights', w * K * 4, S), rhoBuf = this.growable('rho', w * 4, S), sigBuf = this.growable('sigma', w * 4, S);
        q.writeBuffer(distBuf, 0, topD); q.writeBuffer(this.rowsBuf, 0, new Uint32Array([w, 0, 0, 0]));
        { const e = d.createCommandEncoder(), pass = e.beginComputePass(); pass.setPipeline(this.pipes.weights.pipeline); pass.setBindGroup(0, this.bind(this.pipes.weights, [this.rowsBuf, distBuf, wBuf, rhoBuf, sigBuf])); pass.dispatchWorkgroups(Math.ceil(w / 64)); pass.end(); q.submit([e.finish()]); }
        const [rhoRaw, sigRaw] = await this.readMany([{ buffer: rhoBuf, bytes: w * 4 }, { buffer: sigBuf, bytes: w * 4 }]);
        const RHO = new Float32Array(rhoRaw), SIG = new Float32Array(sigRaw); chunk.forEach((r, s) => { this.rho[r] = RHO[s]; this.sigma[r] = SIG[s]; });
      }
    }
    q.writeBuffer(radiusBuf, 0, radius);
  }
  // Drop retired rows from every per-row array (order kept, so rows stay in time order), renumber neighbour lists and
  // re-upload the GPU state. Returns the old-to-new row map (-1 = removed), or null if nothing was dead.
  compact() {
    if (!this.deadPlaced) return null;
    const L = this.data.n, n = this.n, dim = this.dim, map = new Int32Array(this.N).fill(-1); let m = 0, nPlaced = 0;
    for (let i = 0; i < L; i++) if (i >= n || this.alive[i]) { map[i] = m++; if (i < n) nPlaced++; }
    const D = this.data, mv = (arr, i, j, s) => { if (i !== j) arr.copyWithin(j * s, i * s, (i + 1) * s); };
    const kept = [];
    for (let i = 0; i < L; i++) { const j = map[i]; if (j < 0) continue;
      mv(D.featS, i, j, dim); mv(this.knnIdx, i, j, K); mv(this.knnDist, i, j, K); mv(this.knn2Idx, i, j, K); mv(this.knn2Dist, i, j, K); mv(this.ranks, i, j, K); mv(this.pos, i, j, 2); mv(this.start, i, j, 2); mv(this.radius2D, i, j, 1); mv(this.imp, i, j, 1); mv(this.anchor, i, j, 2); mv(this.anchored, i, j, 1);
      mv(this.radius2, i, j, 1); mv(this.rho, i, j, 1); mv(this.sigma, i, j, 1); mv(this.settleDist, i, j, 1); mv(D.tS, i, j, 1); mv(D.wallS, i, j, 1); mv(D.labelS, i, j, 1); mv(D.replyS, i, j, 1);
      kept.push(D.posts[i]); }
    D.posts = kept; for (let i = m; i < L; i++) { D.tS[i] = Infinity; }
    for (let i = 0; i < m * K; i++) { const j = this.knnIdx[i]; this.knnIdx[i] = j >= 0 ? map[j] : -1; const j2 = this.knn2Idx[i]; this.knn2Idx[i] = j2 >= 0 ? map[j2] : -1; }
    this.alive.fill(1); this.deadMark.fill(0); this.deadPlaced = 0; this.n = nPlaced; D.n = m; this.compactions++;
    const q = this.device.queue;
    q.writeBuffer(this.featBuf, 0, D.featS.subarray(0, m * dim)); q.writeBuffer(this.posBuf, 0, this.pos.subarray(0, m * 2)); q.writeBuffer(this.startBuf, 0, this.start.subarray(0, m * 2));
    q.writeBuffer(this.radius2Buf, 0, this.radius2); q.writeBuffer(this.radius2DBuf, 0, this.radius2D); q.writeBuffer(this.aliveBuf, 0, this.alive);
    return map;
  }

  // ---- Quiet rebuild (NodeGardener v3): every living row re-optimised for `steps` Adam steps on the UMAP objective over the
  // symmetrised fuzzy graph of the meaning lists, with five fixed negatives per row; rows present at the previous rebuild are
  // leashed to their anchor with weight lambda, arrivals since then re-seat freely. Features, meaning lists, rho and sigma are
  // untouched; positions, the map lists, the ledger and the local-fit ranks are refreshed afterwards. Old positions stay in
  // `start` so the page can morph. Returns movement statistics in neighbourhood widths (the base map's median 15th-neighbour
  // distance on the map).
  // Fuzzy weights of the given rows' meaning lists (the weights kernel, in chunks of 65,536 rows).
  async _rowWeights(rows) {
    const d = this.device, q = d.queue, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, nr = rows.length;
    const W = new Float32Array(nr * K); const CH = 65536;
    for (let lo = 0; lo < nr; lo += CH) {
      const c = Math.min(CH, nr - lo), dists = new Float32Array(c * K); for (let r = 0; r < c; r++) dists.set(this.knnDist.subarray(rows[lo + r] * K, rows[lo + r] * K + K), r * K);
      const distBuf = this.growable('weight dists', c * K * 4, S), wBuf = this.growable('weights', c * K * 4, S), rhoBuf = this.growable('rho', c * 4, S), sigBuf = this.growable('sigma', c * 4, S);
      q.writeBuffer(distBuf, 0, dists); q.writeBuffer(this.rowsBuf, 0, new Uint32Array([c, 0, 0, 0]));
      { const e = d.createCommandEncoder(), pass = e.beginComputePass(); pass.setPipeline(this.pipes.weights.pipeline); pass.setBindGroup(0, this.bind(this.pipes.weights, [this.rowsBuf, distBuf, wBuf, rhoBuf, sigBuf])); pass.dispatchWorkgroups(Math.ceil(c / 64)); pass.end(); q.submit([e.finish()]); }
      const [wRaw] = await this.readMany([{ buffer: wBuf, bytes: c * K * 4 }]); W.set(new Float32Array(wRaw), lo * K);
    }
    return W;
  }
  // Symmetrised graph in CSR form over the given rows: w_ij + w_ji - w_ij w_ji, each row's edges sorted by neighbour so
  // the edge sum has a fixed order. cidx maps a row to its compact index (-1 = not in the set). Gives the page its turn
  // every 8 ms (_turner).
  async _csrGraph(rows, cidx, W) {
    const nr = rows.length, deg = new Uint32Array(nr + 1), turn = this._turner();
    for (let c = 0; c < nr; c++) { if ((c & 1023) === 1023) await turn(); const i = rows[c]; for (let k = 0; k < K; k++) { const j = this.knnIdx[i * K + k]; if (j < 0 || cidx[j] < 0) continue; deg[c]++; deg[cidx[j]]++; } }
    const off = new Uint32Array(nr + 1); for (let c = 0; c < nr; c++) off[c + 1] = off[c] + deg[c];
    const E0 = off[nr], tIdx = new Uint32Array(E0), tW = new Float32Array(E0), fill = new Uint32Array(nr);
    for (let c = 0; c < nr; c++) { if ((c & 1023) === 1023) await turn(); const i = rows[c]; for (let k = 0; k < K; k++) { const j = this.knnIdx[i * K + k]; if (j < 0 || cidx[j] < 0) continue; const cj = cidx[j], w = W[c * K + k]; let p = off[c] + fill[c]++; tIdx[p] = cj; tW[p] = w; p = off[cj] + fill[cj]++; tIdx[p] = c; tW[p] = w; } }
    const idx = new Uint32Array(E0), ew = new Float32Array(E0), off2 = new Uint32Array(nr + 1); let E = 0, ord = new Uint32Array(256);
    for (let c = 0; c < nr; c++) {
      if ((c & 4095) === 4095) await turn();
      const a = off[c], m = off[c + 1] - a; if (m > ord.length) ord = new Uint32Array(2 * m);
      // stable insertion sort of the row's entries by neighbour (the same order as Array.sort, without an array per row:
      // at 3 million rows those arrays cost more than the sorting)
      if (m <= 256) for (let t = 0; t < m; t++) { const p = a + t, key = tIdx[p]; let z = t; while (z > 0 && tIdx[ord[z - 1]] > key) { ord[z] = ord[z - 1]; z--; } ord[z] = p; }
      else { const o = []; for (let t = 0; t < m; t++) o.push(a + t); o.sort((x, y) => tIdx[x] - tIdx[y]); ord.set(o); }   // a hub row: n log n
      off2[c] = E; let last = -1;
      for (let t = 0; t < m; t++) { const p = ord[t], j = tIdx[p], w = tW[p]; if (j === last) { const u = ew[E - 1]; ew[E - 1] = u + w - u * w; } else { idx[E] = j; ew[E] = w; E++; last = j; } }
    }
    off2[nr] = E;
    return { off2, idx, ew, E };
  }
  // Adam steps on the GPU from y0 (nr x 2) with per-row anchors (x, y, leash), in groups of 8 submissions so concurrent
  // flush work can slot in between. lrOf(step) gives the learning rate; negatives(step) may return a fresh negative
  // table (nr x 5) before a group starts (null keeps the current one).
  async _runSteps(nr, y0, anchorL, graph, neg, { steps, lrOf, clip, negScale, beta1, beta2, eps, negatives = null }) {
    const d = this.device, q = d.queue, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, E = graph.E;
    const offBuf = this.growable('rb off', (nr + 1) * 4, S), idxBuf = this.growable('rb idx', Math.max(4, E * 4), S), ewBuf = this.growable('rb ew', Math.max(4, E * 4), S), negBuf = this.growable('rb neg', nr * 20, S), ancBuf = this.growable('rb anchor', nr * 12, S);
    const yA = this.growable('rb yA', nr * 8, S), yB = this.growable('rb yB', nr * 8, S), adamBuf = this.growable('rb adam', nr * 16, S);
    q.writeBuffer(offBuf, 0, graph.off2); q.writeBuffer(idxBuf, 0, graph.idx.subarray(0, E)); q.writeBuffer(ewBuf, 0, graph.ew.subarray(0, E)); q.writeBuffer(negBuf, 0, neg); q.writeBuffer(ancBuf, 0, anchorL); q.writeBuffer(yA, 0, y0); q.writeBuffer(adamBuf, 0, new Float32Array(nr * 4));
    for (let g0 = 0; g0 < steps; g0 += 8) { const e = d.createCommandEncoder();
      if (negatives) { const fresh = negatives(g0); if (fresh) q.writeBuffer(negBuf, 0, fresh); }
      for (let st = g0; st < Math.min(steps, g0 + 8); st++) {
        const rp = this.rpBufs[st % this.rpBufs.length];
        const u = new DataView(new ArrayBuffer(48)); u.setUint32(0, nr, true); u.setUint32(4, st, true); u.setFloat32(8, this.data.a, true); u.setFloat32(12, this.data.b, true); u.setFloat32(16, lrOf(st), true); u.setFloat32(20, clip, true); u.setFloat32(24, beta1, true); u.setFloat32(28, beta2, true); u.setFloat32(32, eps, true); u.setFloat32(36, negScale, true);
        q.writeBuffer(rp, 0, u.buffer);
        const cur = st % 2 === 0 ? yA : yB, nxt = st % 2 === 0 ? yB : yA;
        const pass = e.beginComputePass(); pass.setPipeline(this.pipes.rebuild.pipeline);
        pass.setBindGroup(0, this.bind(this.pipes.rebuild, [rp, offBuf, idxBuf, ewBuf, negBuf, ancBuf, cur, nxt, adamBuf])); pass.dispatchWorkgroups(Math.ceil(nr / 64)); pass.end();
      }
      q.submit([e.finish()]); await q.onSubmittedWorkDone(); }
    const [yRaw] = await this.readMany([{ buffer: steps % 2 === 0 ? yA : yB, bytes: nr * 8 }]);
    return new Float32Array(yRaw);
  }
  // The base fit's all-pairs lists: every row's exact 15 nearest other rows with the tune mode's search (livefit.js
  // exactLists: per-query workgroup selection, 4 x 4 blocked distances), then rho and sigma from the weights kernel.
  // Entry for entry the same idx, dist, rho and sigma as _refresh over every row (whose dist/partial/merge pipeline took
  // 96 s on 163,749 x 384 rows); _refresh itself still serves placement and quiet rebuilds.
  // given: neighbour lists shipped with the dataset ({ idx, dist, k } with k >= K, nearest first, computed ahead of time on
  // the full vectors, as for arXiv): their first K columns are used and nothing is searched here
  async _baseLists(n, given = null) {
    const d = this.device, q = d.queue, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    let I, D;
    if (given) { I = new Int32Array(n * K); D = new Float32Array(n * K); for (let r = 0; r < n; r++) for (let c = 0; c < K; c++) { I[r * K + c] = given.idx[r * given.k + c]; D[r * K + c] = given.dist[r * given.k + c]; } }
    else ({ I, D } = await exactLists(d, this.featBuf, n, this.dim, K));
    for (let i = 0; i < n * K; i++) { this.knnIdx[i] = I[i]; this.knnDist[i] = D[i]; }
    for (let r = 0; r < n; r++) { const rad = D[r * K + K - 1]; this.radius2[r] = rad * rad; }
    q.writeBuffer(this.radius2Buf, 0, this.radius2);
    const CH = 65536;
    for (let lo = 0; lo < n; lo += CH) {
      const c = Math.min(CH, n - lo);
      const distBuf = this.growable('weight dists', c * K * 4, S), wBuf = this.growable('weights', c * K * 4, S), rhoBuf = this.growable('rho', c * 4, S), sigBuf = this.growable('sigma', c * 4, S);
      q.writeBuffer(distBuf, 0, D.subarray(lo * K, (lo + c) * K)); q.writeBuffer(this.rowsBuf, 0, new Uint32Array([c, 0, 0, 0]));
      { const e = d.createCommandEncoder(), pass = e.beginComputePass(); pass.setPipeline(this.pipes.weights.pipeline); pass.setBindGroup(0, this.bind(this.pipes.weights, [this.rowsBuf, distBuf, wBuf, rhoBuf, sigBuf])); pass.dispatchWorkgroups(Math.ceil(c / 64)); pass.end(); q.submit([e.finish()]); }
      const [rhoRaw, sigRaw] = await this.readMany([{ buffer: rhoBuf, bytes: c * 4 }, { buffer: sigBuf, bytes: c * 4 }]);
      this.rho.set(new Float32Array(rhoRaw), lo); this.sigma.set(new Float32Array(sigRaw), lo);
    }
  }
  // Fresh base map in the browser: exact 15-lists, rho and sigma for the first nBase rows of `feat` (self excluded), the
  // symmetrised fuzzy graph, and a UMAP layout from a PCA start with Adam (negatives redrawn every 8 steps, learning rate
  // decaying to zero). Returns the arrays loadData would have read from a base folder. Float32 throughout.
  static async fitBase(device, feat, nBase, dim, { a, b, steps = 800, lr = 0.08, clip = 4, negScale = 1, seed = 1, initSd = 3.0, negEvery = 8, init = null, onProgress = null, search = 'fast', lists = null } = {}) {
    const t0 = performance.now();
    const data = { N: nBase, n: nBase, dim, k: K, nBase, a, b, featS: feat, baseXY: new Float32Array(nBase * 2), bIdx: new Int32Array(nBase * K), bDist: new Float32Array(nBase * K), bRho: new Float32Array(nBase), bSigma: new Float32Array(nBase), tS: new Float32Array(nBase), wallS: new Float32Array(nBase), labelS: new Uint16Array(nBase), replyS: new Uint8Array(nBase), posts: [] };
    const tmp = new StreamingEngine(device, data, { noLedger: true }); await tmp.ledgerReady;
    const rows = new Array(nBase); for (let i = 0; i < nBase; i++) rows[i] = i;
    onProgress?.('base fit: exact neighbour lists');
    if (search === 'refresh' || nBase <= K) await tmp._refresh(rows, { feat: tmp.featBuf, dim, kI: tmp.knnIdx, kD: tmp.knnDist, radius: tmp.radius2, radiusBuf: tmp.radius2Buf, weights: true });
    else await tmp._baseLists(nBase, lists && lists.n === nBase ? lists : null);
    const t1 = performance.now(); onProgress?.('base fit: fuzzy graph');
    const cidx = new Int32Array(nBase); for (let i = 0; i < nBase; i++) cidx[i] = i;
    const W = await tmp._rowWeights(rows); const graph = await tmp._csrGraph(rows, cidx, W);
    const t2 = performance.now(); onProgress?.('base fit: layout');
    // init may be a function of the exact lists just computed (a static map seeds every row from its preview through them)
    const y0 = (typeof init === 'function' ? await init(tmp.knnIdx, nBase) : init) || pcaInit(feat, nBase, dim, initSd);
    const anchorL = new Float32Array(nBase * 3);   // no leashes: every row is free
    let rng = mulberry32(seed); const drawNeg = () => { const nq = new Uint32Array(nBase * 5); for (let c = 0; c < nBase; c++) for (let s = 0; s < 5; s++) { let k = Math.floor(rng() * nBase); if (k === c) k = (k + 1) % nBase; nq[c * 5 + s] = k; } return nq; };
    const y1 = await tmp._runSteps(nBase, y0, anchorL, graph, drawNeg(), { steps, lrOf: (st) => lr * (1 - st / steps), clip, negScale, beta1: 0.9, beta2: 0.999, eps: 1e-8, negatives: (g0) => (g0 % negEvery === 0 ? drawNeg() : null) });
    for (let c = 0; c < nBase * 2; c++) if (!Number.isFinite(y1[c])) throw new Error('non-finite position in the base fit');
    const out = { xy: y1, y0, idx: tmp.knnIdx.slice(0, nBase * K), dist: tmp.knnDist.slice(0, nBase * K), rho: tmp.rho.slice(0, nBase), sigma: tmp.sigma.slice(0, nBase), edges: graph.E, msLists: t1 - t0, msGraph: t2 - t1, msSteps: performance.now() - t2, ms: performance.now() - t0, steps };
    tmp.destroy(); return out;
  }
  // Release every GPU buffer this engine owns (used by the throwaway engine of fitBase).
  destroy() { for (const k of Object.keys(this)) { const v = this[k]; if (v instanceof GPUBuffer) { try { v.destroy(); } catch {} } else if (Array.isArray(v)) for (const b of v) if (b instanceof GPUBuffer) { try { b.destroy(); } catch {} } } if (this.grow) for (const b of Object.values(this.grow)) { try { b.destroy(); } catch {} } }

  async quietRebuild({ steps = 80, lr = 0.02, lambda = 10, clip = 4, negScale = 1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
    await this.ledgerReady; const d = this.device, q = d.queue, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, t0 = performance.now();
    const rows = []; for (let i = 0; i < this.n; i++) if (this.alive[i]) rows.push(i);
    const nr = rows.length, cidx = new Int32Array(this.N).fill(-1); rows.forEach((r, c) => { cidx[r] = c; });
    const W = await this._rowWeights(rows);
    const graph = await this._csrGraph(rows, cidx, W), E = graph.E;
    // 3. five fixed negative rows per row (hash of row, slot and rebuild count), anchors and leashes, Adam state
    const neg = new Uint32Array(nr * 5), turn = this._turner(); let h = (this.rebuilds + 1) * 0x9e3779b9;
    for (let c = 0; c < nr; c++) { if ((c & 4095) === 4095) await turn(); for (let sidx = 0; sidx < 5; sidx++) { h = Math.imul(h ^ (c * 5 + sidx + 1), 0x85ebca6b) >>> 0; h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35) >>> 0; h ^= h >>> 16; let k = h % nr; if (k === c) k = (k + 1) % nr; neg[c * 5 + sidx] = k; } }
    const anchorL = new Float32Array(nr * 3), y0 = new Float32Array(nr * 2);
    for (let c = 0; c < nr; c++) { const i = rows[c]; y0[2 * c] = this.pos[2 * i]; y0[2 * c + 1] = this.pos[2 * i + 1]; anchorL[3 * c] = this.anchored[i] ? this.anchor[2 * i] : this.pos[2 * i]; anchorL[3 * c + 1] = this.anchored[i] ? this.anchor[2 * i + 1] : this.pos[2 * i + 1]; anchorL[3 * c + 2] = this.anchored[i] ? lambda : 0; }
    const t1 = performance.now();
    // 4. the steps
    const y1 = await this._runSteps(nr, y0, anchorL, graph, neg, { steps, lrOf: () => lr, clip, negScale, beta1, beta2, eps });
    for (let c = 0; c < nr * 2; c++) if (!Number.isFinite(y1[c])) throw new Error('non-finite position after the rebuild');
    const t2 = performance.now();
    // 5. commit: old positions into `start` (for the morph), new positions into pos, anchors reset, movement statistics
    this.start.set(this.pos.subarray(0, this.n * 2)); q.writeBuffer(this.startBuf, 0, this.start.subarray(0, this.n * 2));
    const width = this.widthUnit(), mvHist = [], mvArr = [];
    for (let c = 0; c < nr; c++) { const i = rows[c]; const dx = y1[2 * c] - this.pos[2 * i], dy = y1[2 * c + 1] - this.pos[2 * i + 1], m = Math.hypot(dx, dy) / width; (this.anchored[i] ? mvHist : mvArr).push(m); this.pos[2 * i] = y1[2 * c]; this.pos[2 * i + 1] = y1[2 * c + 1]; this.anchor[2 * i] = y1[2 * c]; this.anchor[2 * i + 1] = y1[2 * c + 1]; this.anchored[i] = 1; }
    q.writeBuffer(this.posBuf, 0, this.pos.subarray(0, this.n * 2));
    // quantiles from one sort per group (the same values as sorting for each quantile), with a turn between the sorts
    await turn(); const sHist = Float64Array.from(mvHist).sort(); await turn(); const sArr = Float64Array.from(mvArr).sort(); await turn();
    const qtl = (arr, p) => { const a = arr === mvHist ? sHist : sArr; if (!a.length) return 0; return a[Math.min(a.length - 1, Math.floor(p * a.length))]; };
    // 6. map lists, ledger and ranks follow the new layout
    const before = new Map(); for (let c = 0; c < nr; c++) { before.set(rows[c], this._impostors(rows[c])); if ((c & 511) === 511) await turn(); }
    await this._refresh(rows, { feat: this.posBuf, dim: 2, kI: this.knn2Idx, kD: this.knn2Dist, radius: this.radius2D, radiusBuf: this.radius2DBuf, weights: false, chunkRows: 256 });   // small pieces: a concurrent flush waits behind one piece at a time
    const netBefore = this.ledger.net, grossKeep = this.ledger.gross; await this._ledgerUpdateAsync(before); this.ledger.gross = grossKeep; this.ledger.since = 0;
    await this.computeRanks(0, this.n, this.n, this.knnIdx.subarray(0, this.n * K));
    const t3 = performance.now(); this.rebuilds++; this.placedAtRebuild = this.placedTotal;
    this.lastRebuild = { rows: nr, edges: E, steps, lr, lambda, histMedian: qtl(mvHist, 0.5), histP95: qtl(mvHist, 0.95), histMax: qtl(mvHist, 1), arrMedian: qtl(mvArr, 0.5), arrP95: qtl(mvArr, 0.95), nHist: mvHist.length, nArr: mvArr.length, ledgerBefore: netBefore, ledgerAfter: this.ledger.net, msPrep: t1 - t0, msSteps: t2 - t1, msAfter: t3 - t2, msTotal: t3 - t0 };
    return this.lastRebuild;
  }
  // One neighbourhood width: the base map's median distance to its 15th map neighbour (from the base's own 2-D lists).
  widthUnit() { if (this._width) return this._width; const a = []; for (let i = 0; i < this.data.nBase; i++) if (this.radius2D[i] > 0) a.push(Math.sqrt(this.radius2D[i])); a.sort((x, y) => x - y); this._width = a.length ? a[a.length >> 1] : 1; return this._width; }

  async computeRanks(qBase, count, n, nbrIds, posBuf = this.posBuf, target = this.ranks, targetOffset = qBase * K) {
    if (n >= RANKS_GRID_MIN) return this.computeRanksGrid(qBase, count, n, nbrIds, posBuf, target, targetOffset);   // a pass over every row costs n x n
    const d = this.device, q = d.queue, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const nbrBuf = this.growable('rank nbr', count * K * 4, S), outBuf = this.growable('rank out', count * K * 4, S);
    q.writeBuffer(nbrBuf, 0, nbrIds instanceof Int32Array ? new Uint32Array(nbrIds.buffer, nbrIds.byteOffset, count * K) : nbrIds);
    const CH = count > 8192 ? 1024 : 4096;   // whole-map recomputes in small pieces, for the same reason
    for (let lo = 0; lo < count; lo += CH) {
      const c = Math.min(CH, count - lo); const rp = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      q.writeBuffer(rp, 0, new Uint32Array([n, qBase + lo, c, 0]));
      const e = d.createCommandEncoder(), pass = e.beginComputePass(); pass.setPipeline(this.pipes.ranks.pipeline);
      pass.setBindGroup(0, d.createBindGroup({ layout: this.pipes.ranks.pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: rp } }, { binding: 1, resource: { buffer: posBuf } }, { binding: 2, resource: { buffer: nbrBuf, offset: lo * K * 4, size: c * K * 4 } }, { binding: 3, resource: { buffer: outBuf, offset: lo * K * 4, size: c * K * 4 } }] }));
      pass.dispatchWorkgroups(c); pass.end(); q.submit([e.finish()]);
      if (count > 8192) await q.onSubmittedWorkDone();   // whole-map recomputes (rebuilds) let flush work slot in between chunks
    }
    const [raw] = await this.readMany([{ buffer: outBuf, bytes: count * K * 4 }]);
    target.set(new Uint32Array(raw), targetOffset);
  }

  // computeRanks through a uniform grid (K7g): the same counts below RANK_CAP, RANK_CAP above. The grid holds about four rows
  // a cell over the map's bounding box, built here by a counting sort of the GPU's own positions (so every distance is
  // computed from the same float32 values as K7's).
  async computeRanksGrid(qBase, count, n, nbrIds, posBuf = this.posBuf, target = this.ranks, targetOffset = qBase * K) {
    const d = this.device, q = d.queue, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, turn = this._turner();
    const [praw] = await this.readMany([{ buffer: posBuf, bytes: n * 8 }]); const P = new Float32Array(praw);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { const x = P[2 * i], y = P[2 * i + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const cs = Math.max(Math.sqrt(Math.max((x1 - x0) * (y1 - y0), 1e-12) * 4 / n), 1e-9);
    const gx = Math.min(Math.floor((x1 - x0) / cs) + 1, 8192), gy = Math.min(Math.floor((y1 - y0) / cs) + 1, 8192), csx = Math.max(cs, (x1 - x0) / (gx - 0.5)), csy = Math.max(cs, (y1 - y0) / (gy - 0.5)), c2 = Math.max(csx, csy);
    const cellOf = new Uint32Array(n), start = new Uint32Array(gx * gy + 1);
    for (let i = 0; i < n; i++) { const cx = Math.min(gx - 1, Math.max(0, Math.floor((P[2 * i] - x0) / c2))), cy = Math.min(gy - 1, Math.max(0, Math.floor((P[2 * i + 1] - y0) / c2))); const c = cy * gx + cx; cellOf[i] = c; start[c + 1]++; }
    for (let c = 0; c < gx * gy; c++) start[c + 1] += start[c];
    await turn(); const fill = start.slice(0, gx * gy), pts = new Float32Array(n * 2), ids = new Uint32Array(n);
    for (let i = 0; i < n; i++) { const t = fill[cellOf[i]]++; pts[2 * t] = P[2 * i]; pts[2 * t + 1] = P[2 * i + 1]; ids[t] = i; }
    const startBuf = this.growable('rank grid start', start.byteLength, S), ptsBuf = this.growable('rank grid points', pts.byteLength, S), idsBuf = this.growable('rank grid ids', ids.byteLength, S);
    q.writeBuffer(startBuf, 0, start); q.writeBuffer(ptsBuf, 0, pts); q.writeBuffer(idsBuf, 0, ids);
    const nbrBuf = this.growable('rank nbr', count * K * 4, S), outBuf = this.growable('rank out', count * K * 4, S);
    q.writeBuffer(nbrBuf, 0, nbrIds instanceof Int32Array ? new Uint32Array(nbrIds.buffer, nbrIds.byteOffset, count * K) : nbrIds);
    const CH = 32768, pipe = this.pipes.ranksGrid.pipeline;
    for (let lo = 0; lo < count; lo += CH) {   // pieces, each waited for, so no one submission runs long and flush work can slot in
      const c = Math.min(CH, count - lo), rp = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), pb = new ArrayBuffer(32);
      new Uint32Array(pb, 0, 4).set([qBase + lo, c, gx, gy]); new Float32Array(pb, 16, 3).set([x0, y0, c2]); q.writeBuffer(rp, 0, pb);
      const e = d.createCommandEncoder(), pass = e.beginComputePass(); pass.setPipeline(pipe);
      pass.setBindGroup(0, d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: rp } }, { binding: 1, resource: { buffer: posBuf } }, { binding: 2, resource: { buffer: nbrBuf, offset: lo * K * 4, size: c * K * 4 } }, { binding: 3, resource: { buffer: outBuf, offset: lo * K * 4, size: c * K * 4 } }, { binding: 4, resource: { buffer: startBuf, size: start.byteLength } }, { binding: 5, resource: { buffer: ptsBuf, size: pts.byteLength } }, { binding: 6, resource: { buffer: idsBuf, size: ids.byteLength } }] }));
      pass.dispatchWorkgroups(Math.ceil(c / 64)); pass.end(); q.submit([e.finish()]); await q.onSubmittedWorkDone(); rp.destroy();
    }
    const [raw] = await this.readMany([{ buffer: outBuf, bytes: count * K * 4 }]);
    target.set(new Uint32Array(raw), targetOffset);
    this.lastRanks = { grid: [gx, gy], cell: c2, cap: RANK_CAP };
  }

  adoptersOf(p) { const out = []; const n = this.n, idx = this.knnIdx; for (let i = 0; i < n; i++) { const b = i * K; for (let k = 0; k < K; k++) if (idx[b + k] === p) { out.push(i); break; } } return out; }

  growable(role, bytes, usage) {
    const cur = this.grow[role];
    if (cur && cur.size >= bytes) return cur;
    if (cur) cur.destroy();
    const size = Math.max(bytes, 4096, cur ? cur.size * 2 : 0);
    return this.grow[role] = this.device.createBuffer({ label: role, size, usage });
  }
  stagingFor(bytes) {
    let s = this.staging.find(x => x.size >= bytes && !x.busy);
    if (!s) { s = this.device.createBuffer({ label: 'staging', size: Math.max(bytes, 1 << 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); this.staging.push(s); }
    return s;
  }
  async readMany(regions) {
    let total = 0; const offs = regions.map(r => { const o = total; total += Math.ceil(r.bytes / 4) * 4; return o; });
    const st = this.stagingFor(total); st.busy = true; const enc = this.device.createCommandEncoder();
    regions.forEach((r, i) => { if (r.bytes) enc.copyBufferToBuffer(r.buffer, r.offset || 0, st, offs[i], Math.ceil(r.bytes / 4) * 4); });
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ, 0, total);
    const all = st.getMappedRange(0, total).slice(0); st.unmap(); st.busy = false;
    return regions.map((r, i) => all.slice(offs[i], offs[i] + Math.ceil(r.bytes / 4) * 4));
  }
  bind(pipe, buffers) {
    return this.device.createBindGroup({ layout: pipe.pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }

  // Place the next w arrivals (w <= FLUSH_CAP). Returns a stats record.
  async flush(w) {
    this.device.pushErrorScope('validation'); this.device.pushErrorScope('out-of-memory');
    try { return await this._flush(w); }
    finally { const oom = await this.device.popErrorScope(), val = await this.device.popErrorScope(); if (oom || val) throw new Error('GPU error during flush: ' + (oom || val).message); }
  }
  // Exact top-15 search of the w rows [start, start+w) against the placed rows [0, start) plus the new-vs-new block, in the
  // given space (features, or the 2-D map through the position buffer with dim 2), with the adopter radius test.
  async _search({ feat, dim, radiusBuf, start, w }) {
    const d = this.device, q = d.queue;
    q.writeBuffer(this.adopterCountBuf, 0, new Uint32Array([0, 0, 0, 0]));
    q.writeBuffer(this.histBaseNew, 0, new Uint32Array([start, 0, 0, 0]));
    const enc = d.createCommandEncoder(); let tile = 0;
    for (let qb = start; qb < start + w; qb += QTILE, tile++) {
      const qc = Math.min(QTILE, start + w - qb), pb = this.paramBufs[tile];
      q.writeBuffer(pb, 0, new Uint32Array([start, qb, qc, dim, this.N, ADOPTER_CAP, start, 0]));
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipes.dist.pipeline); pass.setBindGroup(0, this.bind(this.pipes.dist, [pb, feat, radiusBuf, this.d2Buf, this.adopterCountBuf, this.adopterBuf, this.histBaseZero, this.aliveBuf, this.qsrcBuf]));
      pass.dispatchWorkgroups(Math.ceil(start / 256), qc);
      pass.setPipeline(this.pipes.partial.pipeline); pass.setBindGroup(0, this.bind(this.pipes.partial, [pb, this.d2Buf, this.pDistBuf, this.pIdBuf])); pass.dispatchWorkgroups(qc);
      pass.setPipeline(this.pipes.merge.pipeline); pass.setBindGroup(0, this.bind(this.pipes.merge, [pb, this.pDistBuf, this.pIdBuf, this.topDistBuf, this.topIdBuf])); pass.dispatchWorkgroups(Math.ceil(qc / 64));
      pass.end();
    }
    for (let qb = start; qb < start + w; qb += QTILE, tile++) {
      const qc = Math.min(QTILE, start + w - qb), pb = this.paramBufs[tile];
      q.writeBuffer(pb, 0, new Uint32Array([w, qb, qc, dim, w, ADOPTER_CAP, start, 1]));
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipes.dist.pipeline); pass.setBindGroup(0, this.bind(this.pipes.dist, [pb, feat, radiusBuf, this.nnBuf, this.adopterCountBuf, this.adopterBuf, this.histBaseNew, this.aliveBuf, this.qsrcBuf]));
      pass.dispatchWorkgroups(Math.ceil(w / 256), qc); pass.end();
    }
    q.submit([enc.finish()]);
    const [cntRaw, topDRaw, topIRaw, nnRaw] = await this.readMany([{ buffer: this.adopterCountBuf, bytes: 16 }, { buffer: this.topDistBuf, bytes: w * K * 4 }, { buffer: this.topIdBuf, bytes: w * K * 4 }, { buffer: this.nnBuf, bytes: w * w * 4 }]);
    const nCand = new Uint32Array(cntRaw)[0]; if (nCand > ADOPTER_CAP) throw new Error('adopter candidate list overflow: ' + nCand);
    const [adRaw] = nCand ? await this.readMany([{ buffer: this.adopterBuf, bytes: nCand * 16 }]) : [new ArrayBuffer(0)];
    return { nCand, topD: new Float32Array(topDRaw), topI: new Uint32Array(topIRaw), nn: new Float32Array(nnRaw), adRaw };
  }
  // Order-independent merge of a search result into the 15-lists of one space: adopter candidates into history rows
  // (sorted, so the result is the same whatever order the GPU appended them), the new rows' own lists, radii. Returns the
  // history rows whose lists changed.
  _mergeLists({ kI, kD, radius, start, w, nCand, topD, topI, nn, adRaw }) {
    const adU = new Uint32Array(adRaw), adF = new Float32Array(adRaw);
    const candOrder = new Uint32Array(nCand); for (let c = 0; c < nCand; c++) candOrder[c] = c;
    candOrder.sort((x, y) => (adU[4 * x] - adU[4 * y]) || (adU[4 * x + 1] - adU[4 * y + 1]));
    const insert = (row, dd, id) => {
      const base = row * K; let pos = K - 1;
      if (dd > kD[base + pos] || (dd === kD[base + pos] && id > kI[base + pos])) return false;
      while (pos > 0 && (dd < kD[base + pos - 1] || (dd === kD[base + pos - 1] && id < kI[base + pos - 1]))) { kD[base + pos] = kD[base + pos - 1]; kI[base + pos] = kI[base + pos - 1]; pos--; }
      kD[base + pos] = dd; kI[base + pos] = id; return true;
    };
    const changed = []; let last = -1;
    for (let c = 0; c < nCand; c++) { const e = candOrder[c], i = adU[4 * e]; insert(i, Math.sqrt(adF[4 * e + 2]), adU[4 * e + 1]); if (i !== last) { changed.push(i); last = i; } }
    for (const i of changed) { const r = kD[i * K + K - 1]; radius[i] = r * r; }
    for (let r = 0; r < w; r++) {
      const g = start + r;
      for (let k = 0; k < K; k++) { kD[g * K + k] = topD[r * K + k]; kI[g * K + k] = topI[r * K + k]; }
      for (let s = 0; s < w; s++) if (s !== r) insert(g, Math.sqrt(nn[r * w + s]), start + s);
      const rad = kD[g * K + K - 1]; radius[g] = rad * rad;
    }
    return changed;
  }
  async _flush(w) {
    const start = this.n; if (w <= 0) return null; if (w > this.wMax) throw new Error('flush larger than FLUSH_CAP'); if (start + w > this.data.n) throw new Error('flush beyond loaded rows');
    const d = this.device, q = d.queue, t0 = performance.now();

    // ---- 1. GPU: exact search of every arrival against the frozen history, adopter radius test, new-vs-new distances.
    const { nCand, topD, topI, nn, adRaw } = await this._search({ feat: this.featBuf, dim: this.dim, radiusBuf: this.radius2Buf, start, w });
    const t1 = performance.now();

    // ---- 2. CPU: graph bookkeeping (order-independent merge of adopter candidates) and label inheritance.
    const kI = this.knnIdx, kD = this.knnDist;
    const histTop = new Float32Array(w * K); histTop.set(topD);   // the history-only lists, for the barycentre start weights
    const before = new Map(); { const adU = new Uint32Array(adRaw); for (let c = 0; c < nCand; c++) { const i = adU[4 * c]; if (!before.has(i)) before.set(i, this._impostors(i)); } }   // ledger: impostor sets before any list changes
    const changed = this._mergeLists({ kI, kD, radius: this.radius2, start, w, nCand, topD, topI, nn, adRaw });
    const labelS = this.data.labelS, votes = new Map();
    for (let r = 0; r < w; r++) {
      const g = start + r;
      // label = majority among the 15 nearest HISTORY neighbours; ties go to the label seen first (nearest)
      if (this.data.catIndex) { const c = this.data.posts[g]?.cat; labelS[g] = this.data.catIndex.get(c) ?? this.data.catOther; continue; }   // a static dataset's own label
      votes.clear(); let bestL = labelS[topI[r * K]], bestC = 0;
      for (let k = 0; k < K; k++) { const l = labelS[topI[r * K + k]]; const c = (votes.get(l) || 0) + 1; votes.set(l, c); if (c > bestC) { bestC = c; bestL = l; } }
      labelS[g] = bestL;
    }
    this.onLabels?.(start, w);
    const adopters = Array.from({ length: w }, () => []);
    for (const i of changed) for (let k = 0; k < K; k++) { const j = kI[i * K + k]; if (j >= start) adopters[j - start].push([i, k]); }
    for (let r = 0; r < w; r++) for (let k = 0; k < K; k++) { const j = kI[(start + r) * K + k]; if (j >= start) adopters[j - start].push([start + r, k]); }
    const t2 = performance.now();

    // ---- 3. GPU: fuzzy weights for changed history rows, the new rows, and the history-only start rows.
    const rows = changed.length + 2 * w, dists = new Float32Array(rows * K);
    changed.forEach((i, r) => dists.set(this.knnDist.subarray(i * K, i * K + K), r * K));
    for (let r = 0; r < w; r++) dists.set(this.knnDist.subarray((start + r) * K, (start + r) * K + K), (changed.length + r) * K);
    dists.set(histTop, (changed.length + w) * K);
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const distBuf = this.growable('weight dists', rows * K * 4, S), wBuf = this.growable('weights', rows * K * 4, S), rhoBuf = this.growable('rho', rows * 4, S), sigBuf = this.growable('sigma', rows * 4, S);
    q.writeBuffer(distBuf, 0, dists); q.writeBuffer(this.rowsBuf, 0, new Uint32Array([rows, 0, 0, 0]));
    { const e = d.createCommandEncoder(), pass = e.beginComputePass(); pass.setPipeline(this.pipes.weights.pipeline);
      pass.setBindGroup(0, this.bind(this.pipes.weights, [this.rowsBuf, distBuf, wBuf, rhoBuf, sigBuf])); pass.dispatchWorkgroups(Math.ceil(rows / 64)); pass.end(); q.submit([e.finish()]); }
    const [wRaw, rhoRaw, sigRaw] = await this.readMany([{ buffer: wBuf, bytes: rows * K * 4 }, { buffer: rhoBuf, bytes: rows * 4 }, { buffer: sigBuf, bytes: rows * 4 }]);
    const W = new Float32Array(wRaw), RHO = new Float32Array(rhoRaw), SIG = new Float32Array(sigRaw);
    changed.forEach((i, r) => { this.rho[i] = RHO[r]; this.sigma[i] = SIG[r]; });
    for (let r = 0; r < w; r++) { this.rho[start + r] = RHO[changed.length + r]; this.sigma[start + r] = SIG[changed.length + r]; }
    const wRow = new Map(); changed.forEach((i, r) => wRow.set(i, r)); for (let r = 0; r < w; r++) wRow.set(start + r, changed.length + r);

    // ---- 4. CPU: incident rows (forward neighbours + adopters, fuzzy union) and barycentre start.
    const rowsEdges = []; let width = 0;
    for (let r = 0; r < w; r++) {
      const g = start + r, acc = new Map(), wr = wRow.get(g);
      for (let k = 0; k < K; k++) acc.set(this.knnIdx[g * K + k], W[wr * K + k]);
      for (const [i, k] of adopters[r]) { const wv = W[wRow.get(i) * K + k], old = acc.get(i); acc.set(i, old === undefined ? wv : old + wv - old * wv); }
      const e = [...acc.entries()].sort((x, y) => x[0] - y[0]); rowsEdges.push(e); if (e.length > width) width = e.length;
    }
    const nbr = new Uint32Array(w * width).fill(EMPTY), ew = new Float32Array(w * width), init = new Float32Array(w * 2);
    rowsEdges.forEach((e, r) => e.forEach(([j, wv], c) => { nbr[r * width + c] = j; ew[r * width + c] = wv; }));
    for (let r = 0; r < w; r++) {
      const wr = changed.length + w + r; let sx = 0, sy = 0, sw = 0;
      for (let k = 0; k < K; k++) { const j = topI[r * K + k], wv = W[wr * K + k]; sx += wv * this.pos[2 * j]; sy += wv * this.pos[2 * j + 1]; sw += wv; }
      init[2 * r] = sx / sw; init[2 * r + 1] = sy / sw;
    }
    const t3 = performance.now();

    // ---- 5. GPU: Jacobi fixed point, 20 iterations with GPU-side tolerance stop, one readback.
    const nbrBuf = this.growable('nbr', w * width * 4, S), ewBuf = this.growable('ew', w * width * 4, S);
    q.writeBuffer(nbrBuf, 0, nbr); q.writeBuffer(ewBuf, 0, ew); q.writeBuffer(this.curBuf, 0, init); q.writeBuffer(this.traceBuf, 0, new Float32Array(68));
    { const e = d.createCommandEncoder();
      for (let it = 0; it < JACOBI_ITERS; it++) {
        const jp = this.jpBufs[it]; const u = new DataView(new ArrayBuffer(32));
        u.setUint32(0, w, true); u.setUint32(4, width, true); u.setUint32(8, start, true); u.setUint32(12, it, true); u.setFloat32(16, this.data.a, true); u.setFloat32(20, this.data.b, true); u.setFloat32(24, JACOBI_TOL, true);
        q.writeBuffer(jp, 0, u.buffer);
        const cur = it % 2 === 0 ? this.curBuf : this.updBuf, upd = it % 2 === 0 ? this.updBuf : this.curBuf;
        const pass = e.beginComputePass();
        pass.setPipeline(this.pipes.jacobi.pipeline); pass.setBindGroup(0, this.bind(this.pipes.jacobi, [jp, this.posBuf, cur, upd, nbrBuf, ewBuf, this.movementBuf, this.traceBuf])); pass.dispatchWorkgroups(Math.ceil(w / 64));
        pass.setPipeline(this.pipes.mean.pipeline); pass.setBindGroup(0, this.bind(this.pipes.mean, [jp, this.movementBuf, this.traceBuf])); pass.dispatchWorkgroups(1);
        pass.end();
      }
      e.copyBufferToBuffer(this.curBuf, 0, this.posBuf, start * 8, w * 8);
      q.submit([e.finish()]); }
    const [posRaw, trRaw] = await this.readMany([{ buffer: this.curBuf, bytes: w * 8 }, { buffer: this.traceBuf, bytes: 68 * 4 }]);
    const newPos = new Float32Array(posRaw), trace = new Float32Array(trRaw);
    if (trace[64] !== 0) throw new Error('Jacobi row with zero weight');
    let iters = JACOBI_ITERS, stop = 'max_iterations';
    for (let it = 0; it < JACOBI_ITERS; it++) if (trace[it] < JACOBI_TOL) { iters = it + 1; stop = 'movement_tolerance'; break; }
    for (let r = 0; r < w; r++) { if (!Number.isFinite(newPos[2 * r]) || !Number.isFinite(newPos[2 * r + 1])) throw new Error('non-finite placement'); }
    this.pos.set(newPos, start * 2); this.start.set(init, start * 2); q.writeBuffer(this.startBuf, start * 8, init);
    for (let r = 0; r < w; r++) this.settleDist[start + r] = Math.hypot(newPos[2 * r] - init[2 * r], newPos[2 * r + 1] - init[2 * r + 1]);
    const bytes = new Uint8Array(newPos.buffer); let h = this.hash; for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193) >>> 0; } this.hash = h;
    // ---- 6. the map's own 15-lists for the new rows and the history rows they enter (exact, dim 2 on the positions), then the ledger
    await this.ledgerReady;
    const r2 = await this._search({ feat: this.posBuf, dim: 2, radiusBuf: this.radius2DBuf, start, w });
    const adU2 = new Uint32Array(r2.adRaw); for (let c = 0; c < r2.nCand; c++) { const i = adU2[4 * c]; if (!before.has(i)) before.set(i, this._impostors(i)); }
    for (let r = 0; r < w; r++) before.set(start + r, new Set());
    const changed2 = this._mergeLists({ kI: this.knn2Idx, kD: this.knn2Dist, radius: this.radius2D, start, w, nCand: r2.nCand, topD: r2.topD, topI: r2.topI, nn: r2.nn, adRaw: r2.adRaw });
    this._ledgerUpdate(before);
    this.n += w; this.flushes++; this.placedTotal += w;
    q.writeBuffer(this.radius2Buf, 0, this.radius2); q.writeBuffer(this.radius2DBuf, 0, this.radius2D);
    await this.computeRanks(start, w, this.n, this.knnIdx.subarray(start * K, (start + w) * K));
    const t4 = performance.now();
    return { start, w, n: this.n, tFirst: this.data.tS[start], tLast: this.data.tS[start + w - 1], adoptersChanged: changed.length, candidateEdges: nCand, width, iters, stop, movement: trace[Math.max(0, iters - 1)], mapAdopters: changed2.length, ledger: { ...this.ledger },
             msSearch: t1 - t0, msGraph: t2 - t1, msWeights: t3 - t2, msSolve: t4 - t3, msTotal: t4 - t0, hash: this.hash };
  }
}
