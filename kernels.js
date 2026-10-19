// WGSL compute kernels for the Jetstream live streaming-UMAP demo (copied from the ASRS month-by-month
// replay; K1 additionally reads an alive mask and a per-query source row so retired rows drop out of the
// search and a neighbour-list refresh can exclude the row it stands for; with every row alive the
// results, and the replay hash, are unchanged. Every kernel is dimension-general and float32). Numerical reductions use a
// fixed order; the only atomic is the adopter-list append counter, and that list is sorted on the CPU
// before use, so a replay is reproducible for a given device and flush schedule.
export const K = 15;
export const TOPK_THREADS = 256;   // threads per query in the partial top-15 pass

// Shared parameter block (8 x u32 = 32 bytes).
//   nHist      rows in the searched block            qBase     first query (stream position)
//   qCount     queries in this tile                  dim       feature dimension (50)
//   stride     row stride of the d2 scratch buffer   adopterCap  capacity of the adopter list
//   flushBase  first stream position of the flush    mode      0 = history search, 1 = new-vs-new block, 2 = list refresh (no adoption)
export const WGSL_COMMON = /* wgsl */`
const K: u32 = ${K}u;
const BIG: f32 = 3.0e38;   // empty-slot sentinel (finite, above any real squared distance; WGSL rejects constant inf)
struct Params { nHist: u32, qBase: u32, qCount: u32, dim: u32, stride: u32, adopterCap: u32, flushBase: u32, mode: u32 };
`;

// K1: squared Euclidean distance (direct sum of squared differences, no GEMM rearrangement) from
// every searched row i to every query q in the tile. History rows whose current 15th-neighbour
// radius covers the query are appended to the adopter candidate list (new rows carry radius 0,
// so the new-vs-new pass appends nothing).
export const WGSL_DISTANCE = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> feats: array<f32>;
@group(0) @binding(2) var<storage, read> radius2: array<f32>;
@group(0) @binding(3) var<storage, read_write> d2: array<f32>;
@group(0) @binding(4) var<storage, read_write> adopterCount: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> adopters: array<vec3<u32>>;   // (historyRow, queryGlobal, bits(d2))
@group(0) @binding(6) var<uniform> histBase: u32;   // first stream position of the searched block
@group(0) @binding(7) var<storage, read> alive: array<u32>;   // 0 = retired row: never a neighbour, never an adopter
@group(0) @binding(8) var<storage, read> qsrc: array<u32>;    // per query slot, the row it stands for (itself; a list-refresh query copies another row)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let q = gid.y;
  if (i >= P.nHist || q >= P.qCount) { return; }
  let h = histBase + i; let qg = P.qBase + q;
  let rowI = h * P.dim; let rowQ = qg * P.dim;
  var acc: f32 = 0.0;
  for (var j: u32 = 0u; j < P.dim; j = j + 1u) { let t = feats[rowI + j] - feats[rowQ + j]; acc = acc + t * t; }
  let isSelf = (h == qsrc[qg]);
  if (alive[h] == 0u || isSelf) { acc = BIG; }
  let row = select(q, P.qBase - P.flushBase + q, P.mode == 1u);
  d2[row * P.stride + i] = acc;
  if (P.mode != 2u && !isSelf && acc < radius2[h]) {   // mode 2 = neighbour-list refresh: no adoption
    let slot = atomicAdd(&adopterCount, 1u);
    if (slot < P.adopterCap) { adopters[slot] = vec3<u32>(h, qg, bitcast<u32>(acc)); }
  }
}`;

// K2: per query, each of 256 threads scans a strided slice of its d2 row and keeps a private
// top-15 ordered by (distance, id). Empty slots hold BIG / 0xffffffff.
export const WGSL_TOPK_PARTIAL = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> d2: array<f32>;
@group(0) @binding(2) var<storage, read_write> pDist: array<f32>;
@group(0) @binding(3) var<storage, read_write> pId: array<u32>;
fn before(av: f32, ai: u32, bv: f32, bi: u32) -> bool { return av < bv || (av == bv && ai < bi); }
@compute @workgroup_size(${TOPK_THREADS})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let q = wg.x; let t = lid.x;
  var vals: array<f32, K>; var ids: array<u32, K>;
  for (var k: u32 = 0u; k < K; k = k + 1u) { vals[k] = BIG; ids[k] = 0xffffffffu; }
  for (var i: u32 = t; i < P.nHist; i = i + ${TOPK_THREADS}u) {
    let v = d2[q * P.stride + i];
    if (!before(v, i, vals[K - 1u], ids[K - 1u])) { continue; }
    var pos: u32 = K - 1u;
    while (pos > 0u && before(v, i, vals[pos - 1u], ids[pos - 1u])) { vals[pos] = vals[pos - 1u]; ids[pos] = ids[pos - 1u]; pos = pos - 1u; }
    vals[pos] = v; ids[pos] = i;
  }
  let base = (q * ${TOPK_THREADS}u + t) * K;
  for (var k: u32 = 0u; k < K; k = k + 1u) { pDist[base + k] = vals[k]; pId[base + k] = ids[k]; }
}`;

// K3: one thread per query merges its 256 sorted partial lists into the final top-15 (same tie
// rule) and writes the flush-relative row (qBase - flushBase + q). Distances come out as sqrt.
export const WGSL_TOPK_MERGE = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pDist: array<f32>;
@group(0) @binding(2) var<storage, read> pId: array<u32>;
@group(0) @binding(3) var<storage, read_write> outDist: array<f32>;
@group(0) @binding(4) var<storage, read_write> outId: array<u32>;
fn before(av: f32, ai: u32, bv: f32, bi: u32) -> bool { return av < bv || (av == bv && ai < bi); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x; if (q >= P.qCount) { return; }
  var vals: array<f32, K>; var ids: array<u32, K>;
  for (var k: u32 = 0u; k < K; k = k + 1u) { vals[k] = BIG; ids[k] = 0xffffffffu; }
  for (var t: u32 = 0u; t < ${TOPK_THREADS}u; t = t + 1u) {
    let base = (q * ${TOPK_THREADS}u + t) * K;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
      let v = pDist[base + k]; let id = pId[base + k];
      if (id == 0xffffffffu) { break; }
      if (!before(v, id, vals[K - 1u], ids[K - 1u])) { break; }   // partial lists are sorted: later entries cannot enter
      var pos: u32 = K - 1u;
      while (pos > 0u && before(v, id, vals[pos - 1u], ids[pos - 1u])) { vals[pos] = vals[pos - 1u]; ids[pos] = ids[pos - 1u]; pos = pos - 1u; }
      vals[pos] = v; ids[pos] = id;
    }
  }
  let outRow = P.qBase - P.flushBase + q;
  for (var k: u32 = 0u; k < K; k = k + 1u) { outDist[outRow * K + k] = sqrt(vals[k]); outId[outRow * K + k] = ids[k]; }
}`;

// K4: fuzzy weights per row from 15 sorted distances. rho = smallest positive distance; sigma
// solves sum exp(-max(d-rho,0)/sigma) = log2(15) with the project's CUDA bracket-and-bisect
// (low 1e-12, high max(mean,1e-3) doubled until it covers, then 64 bisections).
export const WGSL_WEIGHTS = /* wgsl */`
const K: u32 = ${K}u;
@group(0) @binding(0) var<uniform> rows: u32;
@group(0) @binding(1) var<storage, read> dists: array<f32>;
@group(0) @binding(2) var<storage, read_write> outW: array<f32>;
@group(0) @binding(3) var<storage, read_write> outRho: array<f32>;
@group(0) @binding(4) var<storage, read_write> outSigma: array<f32>;
fn sumExp(row: u32, rho: f32, scale: f32) -> f32 {
  var s: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) { let sh = max(dists[row * K + k] - rho, 0.0); s = s + exp(-sh / scale); }
  return s;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x; if (r >= rows) { return; }
  var rho: f32 = 0.0; var mean: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) { let d = dists[r * K + k]; mean = mean + d; if (d > 0.0 && (rho == 0.0 || d < rho)) { rho = d; } }
  mean = mean / f32(K);
  let tgt = log2(f32(K));
  var low: f32 = 1e-12; var high: f32 = max(mean, 1e-3);
  for (var it: u32 = 0u; it < 64u; it = it + 1u) { if (sumExp(r, rho, high) >= tgt) { break; } high = high * 2.0; }
  for (var it: u32 = 0u; it < 64u; it = it + 1u) { let mid = 0.5 * (low + high); if (sumExp(r, rho, mid) > tgt) { high = mid; } else { low = mid; } }
  let sigma = 0.5 * (low + high);
  for (var k: u32 = 0u; k < K; k = k + 1u) { let sh = max(dists[r * K + k] - rho, 0.0); outW[r * K + k] = exp(-sh / sigma); }
  outRho[r] = rho; outSigma[r] = sigma;
}`;

// K5: one Jacobi fixed-point step for every new row: y <- sum c_j t_j / sum c_j with
// c_j = w_j * 2ab d^(2b-2) / (1 + a d^(2b)), d clamped at 1e-6. History targets are frozen; new
// targets read the current iterate. Once the recorded mean movement of the previous iteration
// is below tol the step becomes a copy (GPU-side stop, no per-iteration readback).
export const WGSL_JACOBI = /* wgsl */`
struct JP { rows: u32, width: u32, nHist: u32, iter: u32, a: f32, b: f32, tol: f32, pad: f32 };
@group(0) @binding(0) var<uniform> P: JP;
@group(0) @binding(1) var<storage, read> hist: array<f32>;
@group(0) @binding(2) var<storage, read> cur: array<f32>;
@group(0) @binding(3) var<storage, read_write> upd: array<f32>;
@group(0) @binding(4) var<storage, read> nbr: array<u32>;
@group(0) @binding(5) var<storage, read> ew: array<f32>;
@group(0) @binding(6) var<storage, read_write> movement: array<f32>;
@group(0) @binding(7) var<storage, read_write> trace: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x; if (r >= P.rows) { return; }
  let x = cur[2u * r]; let y = cur[2u * r + 1u];
  if (P.iter > 0u && trace[P.iter - 1u] < P.tol) { upd[2u * r] = x; upd[2u * r + 1u] = y; movement[r] = 0.0; return; }
  var sx: f32 = 0.0; var sy: f32 = 0.0; var den: f32 = 0.0;
  for (var c: u32 = 0u; c < P.width; c = c + 1u) {
    let p = r * P.width + c; let j = nbr[p];
    if (j == 0xffffffffu) { continue; }
    var tx: f32; var ty: f32;
    if (j < P.nHist) { tx = hist[2u * j]; ty = hist[2u * j + 1u]; } else { let s = j - P.nHist; tx = cur[2u * s]; ty = cur[2u * s + 1u]; }
    let dx = x - tx; let dy = y - ty; var d = sqrt(dx * dx + dy * dy); if (d < 1e-6) { d = 1e-6; }
    let coef = ew[p] * (2.0 * P.a * P.b) * pow(d, 2.0 * P.b - 2.0) / (1.0 + P.a * pow(d, 2.0 * P.b));
    den = den + coef; sx = sx + coef * tx; sy = sy + coef * ty;
  }
  if (den > 0.0) { let nx = sx / den; let ny = sy / den; upd[2u * r] = nx; upd[2u * r + 1u] = ny; movement[r] = sqrt((nx - x) * (nx - x) + (ny - y) * (ny - y)); }
  else { upd[2u * r] = x; upd[2u * r + 1u] = y; movement[r] = 0.0; trace[64u] = 1.0; }
}`;

// K6: fixed-order mean of the per-row movement into trace[iter] (one thread, sequential sum).
export const WGSL_MEAN = /* wgsl */`
struct JP { rows: u32, width: u32, nHist: u32, iter: u32, a: f32, b: f32, tol: f32, pad: f32 };
@group(0) @binding(0) var<uniform> P: JP;
@group(0) @binding(1) var<storage, read> movement: array<f32>;
@group(0) @binding(2) var<storage, read_write> trace: array<f32>;
@compute @workgroup_size(1)
fn main() { var s: f32 = 0.0; for (var r: u32 = 0u; r < P.rows; r = r + 1u) { s = s + movement[r]; } trace[P.iter] = s / f32(P.rows); }`;

// K7: 2-D neighbourhood ranks. For each query report q and each of its 15 feature-space
// neighbours j, count how many placed reports lie strictly closer to q in the 2-D map than j does.
// A neighbour is "kept within window W" when that count is below W. One workgroup per query;
// per-thread partial counts are merged with workgroup atomics (integer, order-independent).
export const WGSL_RANKS = /* wgsl */`
const K: u32 = ${K}u;
struct RP { n: u32, qBase: u32, qCount: u32, pad: u32 };
@group(0) @binding(0) var<uniform> P: RP;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> nbr: array<u32>;
@group(0) @binding(3) var<storage, read_write> ranks: array<u32>;
var<workgroup> counts: array<atomic<u32>, K>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let q = wg.x; let g = P.qBase + q; let qx = pos[2u * g]; let qy = pos[2u * g + 1u];
  var r: array<f32, K>;
  for (var k: u32 = 0u; k < K; k = k + 1u) { let j = nbr[q * K + k]; let dx = pos[2u * j] - qx; let dy = pos[2u * j + 1u] - qy; r[k] = dx * dx + dy * dy; }
  if (lid.x < K) { atomicStore(&counts[lid.x], 0u); }
  workgroupBarrier();
  var local: array<u32, K>;
  for (var k: u32 = 0u; k < K; k = k + 1u) { local[k] = 0u; }
  for (var i: u32 = lid.x; i < P.n; i = i + 256u) {
    if (i == g) { continue; }
    let dx = pos[2u * i] - qx; let dy = pos[2u * i + 1u] - qy; let d = dx * dx + dy * dy;
    for (var k: u32 = 0u; k < K; k = k + 1u) { if (d < r[k]) { local[k] = local[k] + 1u; } }
  }
  for (var k: u32 = 0u; k < K; k = k + 1u) { atomicAdd(&counts[k], local[k]); }
  workgroupBarrier();
  if (lid.x < K) { ranks[q * K + lid.x] = atomicLoad(&counts[lid.x]); }
}`;

// K8: one step of the quiet rebuild (NodeGardener v3's maintenance move, WP-H4's penalised optimiser in float32):
// every living row takes one synchronous Adam step on the UMAP cross-entropy over its symmetrised fuzzy edges (CSR),
// five fixed negative rows, and a leash to its anchor (weight lambda for rows present at the previous rebuild, 0 for
// arrivals since). Reads Y(t), writes Y(t+1): no row sees another row's new position inside a step, and the edge sum
// runs in CSR order, so a rebuild is reproducible for a given device.
export const WGSL_REBUILD = /* wgsl */`
struct RP { n: u32, step: u32, a: f32, b: f32, lr: f32, clip: f32, beta1: f32, beta2: f32, eps: f32, negScale: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> P: RP;
@group(0) @binding(1) var<storage, read> off: array<u32>;       // CSR row offsets, n+1
@group(0) @binding(2) var<storage, read> idx: array<u32>;       // CSR column (neighbour row)
@group(0) @binding(3) var<storage, read> ew: array<f32>;        // CSR edge weight (symmetrised fuzzy union)
@group(0) @binding(4) var<storage, read> neg: array<u32>;       // 5 fixed negative rows per row
@group(0) @binding(5) var<storage, read> anchorL: array<f32>;   // per row: anchor x, anchor y, leash weight
@group(0) @binding(6) var<storage, read> ycur: array<f32>;
@group(0) @binding(7) var<storage, read_write> ynext: array<f32>;
@group(0) @binding(8) var<storage, read_write> adam: array<f32>; // per row: m.x, m.y, v.x, v.y
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= P.n) { return; }
  let yi = vec2<f32>(ycur[2u * i], ycur[2u * i + 1u]);
  var g = vec2<f32>(0.0, 0.0);
  for (var e: u32 = off[i]; e < off[i + 1u]; e = e + 1u) {   // attraction along fuzzy edges: d/dy of w * log(1 + a d^2b)
    let j = idx[e]; let yj = vec2<f32>(ycur[2u * j], ycur[2u * j + 1u]); let d = yi - yj; let d2 = max(dot(d, d), 1e-12);
    let p = pow(d2, P.b); let c = 2.0 * P.a * P.b * p / (d2 * (1.0 + P.a * p));
    g = g + ew[e] * c * d;
  }
  for (var s: u32 = 0u; s < 5u; s = s + 1u) {   // repulsion from fixed negatives: d/dy of -log(1 - q)
    let k = neg[5u * i + s]; if (k == i) { continue; }
    let yk = vec2<f32>(ycur[2u * k], ycur[2u * k + 1u]); let d = yi - yk; let d2 = max(dot(d, d), 1e-12);
    let p = pow(d2, P.b); let c = 2.0 * P.b / ((0.001 + d2) * (1.0 + P.a * p));
    g = g - P.negScale * c * d;
  }
  let lam = anchorL[3u * i + 2u];
  if (lam > 0.0) { g = g + 2.0 * lam * (yi - vec2<f32>(anchorL[3u * i], anchorL[3u * i + 1u])); }
  let gn = length(g); if (gn > P.clip) { g = g * (P.clip / gn); }
  var m = vec2<f32>(adam[4u * i], adam[4u * i + 1u]); var v = vec2<f32>(adam[4u * i + 2u], adam[4u * i + 3u]);
  m = P.beta1 * m + (1.0 - P.beta1) * g; v = P.beta2 * v + (1.0 - P.beta2) * g * g;
  let t = f32(P.step + 1u); let mh = m / (1.0 - pow(P.beta1, t)); let vh = v / (1.0 - pow(P.beta2, t));
  let y = yi - P.lr * mh / (sqrt(vh) + P.eps);
  ynext[2u * i] = y.x; ynext[2u * i + 1u] = y.y; adam[4u * i] = m.x; adam[4u * i + 1u] = m.y; adam[4u * i + 2u] = v.x; adam[4u * i + 3u] = v.y;
}`;
