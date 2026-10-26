// Live-parameter UMAP of a base map on WebGPU (the tuning page, tune.html). Nothing here is used by the streaming page:
// its base fit, Batched RevAdopt placement and NodeGardener quiet rebuilds are unchanged.
//
// What each parameter costs when it changes:
//   neighbour lists   exact, computed once per dataset for up to K_MAX others per row (GPU distance tiles + GPU top-K)
//   k                 fuzzy weights for the first k of those lists (GPU) and their symmetric union (CPU bookkeeping), ~10-100 ms
//   everything else   a uniform: the next epoch uses it. With annealing on, any change restarts the learning-rate ramp.
//   initialisation    new starting positions (spectral: GPU power iteration on the fuzzy graph; PCA; random)
// Two optimisers, both reading and writing the same position buffer the renderer draws from (no readback per frame):
//   SGD    umap-learn's optimize_layout_euclidean: each edge is sampled every wmax/w epochs, pulls both ends together and
//          pushes its head away from `negative sample rate` random rows with strength gamma; per-coordinate clip at 4;
//          learning rate annealed linearly over the epoch budget. One GPU thread per row applies that row's due edges in
//          turn (each pull twice: once as head, once as the tail of the reverse edge, which umap-learn's move_other gives)
//          and reads every other row from the previous epoch, so no update is lost and a run is reproducible on one device.
//          (A first version ran one thread per edge with unsynchronised writes, umap-learn's parallel style; with thousands
//          of threads most writes to a row were overwritten and it needed 1,000 epochs to reach what umap-learn reaches in 200.)
//   Adam   the gradient of this page's base fit (StreamingEngine.fitBase / WGSL_REBUILD): attraction along every fuzzy
//          edge, repulsion from `negatives` random rows scaled by the repulsion strength, gradient-norm clip at 4, Adam
//          moments; one thread per row, synchronous, so a run is reproducible on one device. Negatives are drawn on the
//          GPU each epoch, where fitBase draws them on the CPU every eight steps.
export const K_MAX = 64;
const BIG = 3.0e38;

// a and b of the output kernel 1 / (1 + a d^2b) for a min_dist and spread, by least squares on 300 points in [0, 3 spread]
// (umap-learn's find_ab_params, which uses scipy's curve_fit; here a damped Gauss-Newton fit).
export function findAB(minDist, spread) {
  const xs = [], ys = []; for (let i = 0; i < 300; i++) { const x = (3 * spread * i) / 299; xs.push(x); ys.push(x < minDist ? 1 : Math.exp(-(x - minDist) / spread)); }
  let a = 1.5, b = 0.9, lam = 1e-3;
  const sse = (a, b) => { let s = 0; for (let i = 0; i < 300; i++) { const f = 1 / (1 + a * Math.pow(xs[i], 2 * b)); s += (f - ys[i]) ** 2; } return s; };
  let cur = sse(a, b);
  for (let it = 0; it < 200; it++) {
    let haa = 0, hab = 0, hbb = 0, ga = 0, gb = 0;
    for (let i = 0; i < 300; i++) {
      const x = xs[i]; if (x <= 0) continue; const u = Math.pow(x, 2 * b), den = 1 + a * u, f = 1 / den, r = f - ys[i];
      const da = -u / (den * den), db = -a * u * 2 * Math.log(x) / (den * den);
      haa += da * da; hab += da * db; hbb += db * db; ga += da * r; gb += db * r;
    }
    const A = haa * (1 + lam), B = hbb * (1 + lam), det = A * B - hab * hab; if (!det) break;
    const sa = -(B * ga - hab * gb) / det, sb = -(A * gb - hab * ga) / det;
    const na = Math.max(1e-4, a + sa), nb = Math.max(1e-3, b + sb), nxt = sse(na, nb);
    if (nxt < cur) { a = na; b = nb; lam *= 0.3; if (cur - nxt < 1e-14) { cur = nxt; break; } cur = nxt; } else lam *= 10;
  }
  return { a, b };
}

const WGSL_D2 = /* wgsl */`
struct P { n: u32, dim: u32, q0: u32, qn: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read_write> d2: array<f32>;   // [row][query], so the selection pass reads coalesced
var<workgroup> tq: array<array<f32, 16>, 16>;
var<workgroup> tr: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let row = wg.x * 16u + l.x; let q = wg.y * 16u + l.y;
  let loadRow = wg.x * 16u + l.y; let loadQ = wg.y * 16u + l.y;
  var acc: f32 = 0.0;
  for (var d0: u32 = 0u; d0 < U.dim; d0 = d0 + 16u) {
    let d = d0 + l.x;
    tr[l.y][l.x] = select(0.0, feat[loadRow * U.dim + d], loadRow < U.n && d < U.dim);
    tq[l.y][l.x] = select(0.0, feat[(U.q0 + loadQ) * U.dim + d], loadQ < U.qn && d < U.dim);
    workgroupBarrier();
    for (var t: u32 = 0u; t < 16u; t = t + 1u) { let df = tq[l.y][t] - tr[l.x][t]; acc = acc + df * df; }
    workgroupBarrier();
  }
  if (row < U.n && q < U.qn) { d2[row * U.qn + q] = acc; }
}`;

const WGSL_SELECT = /* wgsl */`
const KM: u32 = ${K_MAX}u;
struct P { n: u32, qn: u32, q0: u32, km: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> d2: array<f32>;
@group(0) @binding(2) var<storage, read_write> outI: array<u32>;
@group(0) @binding(3) var<storage, read_write> outD: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x; if (q >= U.qn) { return; }
  let me = U.q0 + q;
  var bd: array<f32, KM>; var bi: array<u32, KM>;
  for (var j: u32 = 0u; j < KM; j = j + 1u) { bd[j] = ${BIG}; bi[j] = 0xffffffffu; }
  let last = U.km - 1u;
  for (var i: u32 = 0u; i < U.n; i = i + 1u) {
    if (i == me) { continue; }
    let v = d2[i * U.qn + q];
    if (v < bd[last]) {
      var j = last;
      loop { if (j == 0u) { break; } if (bd[j - 1u] <= v) { break; } bd[j] = bd[j - 1u]; bi[j] = bi[j - 1u]; j = j - 1u; }
      bd[j] = v; bi[j] = i;
    }
  }
  for (var j: u32 = 0u; j < U.km; j = j + 1u) { outI[me * U.km + j] = bi[j]; outD[me * U.km + j] = sqrt(max(bd[j], 0.0)); }
}`;

// The selection with a workgroup per query: its 64 threads each keep the nearest LIST of every 64th row (the distance rows
// of this query are contiguous: d2q), then km rounds of a tree minimum over the 64 list heads merge them. Ordered by
// (distance, row), which is the one-thread version's order (rows taken in ascending order, a tie never displacing an
// earlier row), so the lists are the same; but a chunk of a few hundred queries keeps the whole GPU busy, where one thread
// per query took a minute at 182,000 rows. A thread's list is short (16: a 64-long list per thread does not stay in
// registers and costs four times as much); a thread whose list runs out while rounds remain may hold more of the nearest
// km, so its query is flagged and the second pass (lists of 64, only flagged queries) redoes it: exact either way.
const selectPar = (LIST, recheck) => /* wgsl */`
const KM: u32 = ${K_MAX}u;
const LIST: u32 = ${LIST}u;
struct P { n: u32, qn: u32, q0: u32, km: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> d2: array<f32>;
@group(0) @binding(2) var<storage, read_write> outI: array<u32>;
@group(0) @binding(3) var<storage, read_write> outD: array<f32>;
@group(0) @binding(4) var<storage, read_write> flag: array<u32>;
var<workgroup> sd: array<f32, 64>;
var<workgroup> si: array<u32, 64>;
var<workgroup> sw: array<u32, 64>;
var<workgroup> wf: u32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let q = wg.x; let t = l.x; let me = U.q0 + q; let base = q * U.n;
  ${recheck ? 'if (t == 0u) { wf = flag[me]; } if (workgroupUniformLoad(&wf) == 0u) { return; }   // the whole workgroup returns together' : ''}
  var bd: array<f32, LIST>; var bi: array<u32, LIST>;
  for (var j: u32 = 0u; j < LIST; j = j + 1u) { bd[j] = ${BIG}; bi[j] = 0xffffffffu; }
  let last = LIST - 1u; var cut: f32 = ${BIG};   // bd[last] kept in a register: the list itself is touched only on an insert
  for (var i: u32 = t; i < U.n; i = i + 64u) {
    if (i == me) { continue; }
    let v = d2[base + i];
    if (v < cut) {
      var j = last;
      loop { if (j == 0u) { break; } if (bd[j - 1u] <= v) { break; } bd[j] = bd[j - 1u]; bi[j] = bi[j - 1u]; j = j - 1u; }
      bd[j] = v; bi[j] = i; cut = bd[last];
    }
  }
  var h: u32 = 0u; var over: u32 = 0u;
  for (var k: u32 = 0u; k < U.km; k = k + 1u) {
    var hd: f32 = ${BIG}; var hi: u32 = 0xffffffffu; if (h < LIST) { hd = bd[h]; hi = bi[h]; }
    sd[t] = hd; si[t] = hi; sw[t] = t;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (t < s) { let od = sd[t + s]; let oi = si[t + s]; if (od < sd[t] || (od == sd[t] && oi < si[t])) { sd[t] = od; si[t] = oi; sw[t] = sw[t + s]; } }
      workgroupBarrier();
    }
    if (t == 0u) { outI[me * U.km + k] = si[0]; outD[me * U.km + k] = sqrt(max(sd[0], 0.0)); }
    let win = sw[0];
    workgroupBarrier();
    if (t == win) { h = h + 1u; if (h == LIST && cut < ${BIG} && k + 1u < U.km) { over = 1u; } }
  }
  ${recheck ? '' : 'sd[t] = f32(over); workgroupBarrier(); if (t == 0u) { var any: f32 = 0.0; for (var j: u32 = 0u; j < 64u; j = j + 1u) { any = max(any, sd[j]); } flag[me] = u32(any); }'}
}`;
const WGSL_SELECT_PAR = selectPar(16, false), WGSL_SELECT_RECHECK = selectPar(K_MAX, true);

// Distances for the per-query selection, laid out [query][row], each thread computing a 4 x 4 block of them from one
// staged tile (64 rows x 64 queries per workgroup, 16 dimensions at a time): four times the arithmetic per value read.
// Every distance is still the sum of its squared differences in dimension order, padding adding exact zeros, so the
// values are the same as WGSL_D2's.
const WGSL_D2Q = /* wgsl */`
struct P { n: u32, dim: u32, q0: u32, qn: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read_write> d2: array<f32>;   // [query][row]: each query's distances contiguous
var<workgroup> tq: array<array<f32, 16>, 64>;
var<workgroup> tr: array<array<f32, 16>, 64>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let r0 = wg.x * 64u; let q0 = wg.y * 64u; let tid = l.y * 16u + l.x;
  var acc: array<array<f32, 4>, 4>;
  for (var a: u32 = 0u; a < 4u; a = a + 1u) { for (var b: u32 = 0u; b < 4u; b = b + 1u) { acc[a][b] = 0.0; } }
  for (var d0: u32 = 0u; d0 < U.dim; d0 = d0 + 16u) {
    for (var e: u32 = 0u; e < 4u; e = e + 1u) {
      let f = tid * 4u + e; let rr = f / 16u; let dd = f % 16u; let d = d0 + dd;
      let row = r0 + rr; let qq = q0 + rr;
      tr[rr][dd] = select(0.0, feat[row * U.dim + d], row < U.n && d < U.dim);
      tq[rr][dd] = select(0.0, feat[(U.q0 + qq) * U.dim + d], qq < U.qn && d < U.dim);
    }
    workgroupBarrier();
    for (var t: u32 = 0u; t < 16u; t = t + 1u) {
      var qa: array<f32, 4>; var ra: array<f32, 4>;
      for (var a: u32 = 0u; a < 4u; a = a + 1u) { qa[a] = tq[l.y * 4u + a][t]; ra[a] = tr[l.x * 4u + a][t]; }
      for (var a: u32 = 0u; a < 4u; a = a + 1u) { for (var b: u32 = 0u; b < 4u; b = b + 1u) { let df = qa[a] - ra[b]; acc[a][b] = acc[a][b] + df * df; } }
    }
    workgroupBarrier();
  }
  for (var a: u32 = 0u; a < 4u; a = a + 1u) {
    let q = q0 + l.y * 4u + a; if (q >= U.qn) { continue; }
    for (var b: u32 = 0u; b < 4u; b = b + 1u) { let row = r0 + l.x * 4u + b; if (row < U.n) { d2[q * U.n + row] = acc[a][b]; } }
  }
}`;
// Fuzzy weights for the first k of each row's lists: the streaming page's rule (WGSL_WEIGHTS) with k as a parameter.
const WGSL_SIGMA = /* wgsl */`
struct P { n: u32, km: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> dists: array<f32>;
@group(0) @binding(2) var<storage, read_write> outW: array<f32>;
fn sumExp(row: u32, rho: f32, scale: f32) -> f32 {
  var s: f32 = 0.0;
  for (var k: u32 = 0u; k < U.k; k = k + 1u) { let sh = max(dists[row * U.km + k] - rho, 0.0); s = s + exp(-sh / scale); }
  return s;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x; if (r >= U.n) { return; }
  var rho: f32 = 0.0; var mean: f32 = 0.0;
  for (var k: u32 = 0u; k < U.k; k = k + 1u) { let d = dists[r * U.km + k]; mean = mean + d; if (d > 0.0 && (rho == 0.0 || d < rho)) { rho = d; } }
  mean = mean / f32(U.k);
  let tgt = log2(f32(U.k));
  var low: f32 = 1e-12; var high: f32 = max(mean, 1e-3);
  for (var it: u32 = 0u; it < 64u; it = it + 1u) { if (sumExp(r, rho, high) >= tgt) { break; } high = high * 2.0; }
  for (var it: u32 = 0u; it < 64u; it = it + 1u) { let mid = 0.5 * (low + high); if (sumExp(r, rho, mid) > tgt) { high = mid; } else { low = mid; } }
  let sigma = 0.5 * (low + high);
  for (var k: u32 = 0u; k < U.k; k = k + 1u) { let sh = max(dists[r * U.km + k] - rho, 0.0); outW[r * U.k + k] = exp(-sh / sigma); }
}`;

const WGSL_HASH = /* wgsl */`
fn pcg(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
fn rand01(h: u32) -> f32 { return f32(pcg(h) >> 8u) / 16777216.0; }`;

const WGSL_SGD = WGSL_HASH + /* wgsl */`
struct P { n: u32, seed: u32, groups: u32, group: u32, epoch: f32, alpha: f32, a: f32, b: f32, gamma: f32, negRate: f32, wmax: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> nextS: array<f32>;
@group(0) @binding(5) var<storage, read> ycur: array<f32>;
@group(0) @binding(6) var<storage, read_write> ynext: array<f32>;
fn clip4(v: vec2<f32>) -> vec2<f32> { return clamp(v, vec2<f32>(-4.0), vec2<f32>(4.0)); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  if (U.groups > 1u && pcg(i ^ 0x5bd1e995u) % U.groups != U.group) { return; }
  var yi = vec2<f32>(ycur[2u * i], ycur[2u * i + 1u]);
  for (var e: u32 = off[i]; e < off[i + 1u]; e = e + 1u) {
    if (nextS[e] > U.epoch) { continue; }
    nextS[e] = max(nextS[e], U.epoch) + U.wmax / w[e];
    let j = idx[e]; let yj = vec2<f32>(ycur[2u * j], ycur[2u * j + 1u]);
    for (var rep: u32 = 0u; rep < 2u; rep = rep + 1u) {   // the pull this edge gives as head, then the one its reverse edge gives as tail
      let d = yi - yj; let d2 = dot(d, d);
      if (d2 > 0.0) { let c = -2.0 * U.a * U.b * pow(d2, U.b - 1.0) / (U.a * pow(d2, U.b) + 1.0); yi = yi + clip4(c * d) * U.alpha; }
    }
    let h0 = pcg(e ^ pcg(U.seed + bitcast<u32>(U.epoch)));
    var nNeg = u32(floor(U.negRate)); if (rand01(h0) < fract(U.negRate)) { nNeg = nNeg + 1u; }
    for (var p: u32 = 0u; p < nNeg; p = p + 1u) {
      let k = pcg(h0 + 0x9e3779b9u * (p + 1u)) % U.n; if (k == i) { continue; }
      let dk = yi - vec2<f32>(ycur[2u * k], ycur[2u * k + 1u]); let dk2 = dot(dk, dk);
      if (dk2 > 0.0) { let c = 2.0 * U.gamma * U.b / ((0.001 + dk2) * (U.a * pow(dk2, U.b) + 1.0)); yi = yi + clip4(c * dk) * U.alpha; }
    }
  }
  ynext[2u * i] = yi.x; ynext[2u * i + 1u] = yi.y;
}`;

const WGSL_ADAM = WGSL_HASH + /* wgsl */`
struct P { n: u32, step: u32, seed: u32, nNeg: u32, a: f32, b: f32, lr: f32, gamma: f32, beta1: f32, beta2: f32, eps: f32, clip: f32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> ew: array<f32>;
@group(0) @binding(4) var<storage, read> ycur: array<f32>;
@group(0) @binding(5) var<storage, read_write> ynext: array<f32>;
@group(0) @binding(6) var<storage, read_write> adam: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  let yi = vec2<f32>(ycur[2u * i], ycur[2u * i + 1u]);
  var g = vec2<f32>(0.0, 0.0);
  for (var e: u32 = off[i]; e < off[i + 1u]; e = e + 1u) {
    let j = idx[e]; let d = yi - vec2<f32>(ycur[2u * j], ycur[2u * j + 1u]); let d2 = max(dot(d, d), 1e-12);
    let p = pow(d2, U.b); g = g + ew[e] * (2.0 * U.a * U.b * p / (d2 * (1.0 + U.a * p))) * d;
  }
  let h0 = pcg(i ^ pcg(U.seed + U.step * 2654435761u));
  for (var s: u32 = 0u; s < U.nNeg; s = s + 1u) {
    let k = pcg(h0 + 0x9e3779b9u * (s + 1u)) % U.n; if (k == i) { continue; }
    let d = yi - vec2<f32>(ycur[2u * k], ycur[2u * k + 1u]); let d2 = max(dot(d, d), 1e-12);
    let p = pow(d2, U.b); g = g - U.gamma * (2.0 * U.b / ((0.001 + d2) * (1.0 + U.a * p))) * d;
  }
  let gn = length(g); if (gn > U.clip) { g = g * (U.clip / gn); }
  var m = vec2<f32>(adam[4u * i], adam[4u * i + 1u]); var v = vec2<f32>(adam[4u * i + 2u], adam[4u * i + 3u]);
  m = U.beta1 * m + (1.0 - U.beta1) * g; v = U.beta2 * v + (1.0 - U.beta2) * g * g;
  let t = f32(U.step + 1u); let mh = m / (1.0 - pow(U.beta1, t)); let vh = v / (1.0 - pow(U.beta2, t));
  let y = yi - U.lr * mh / (sqrt(vh) + U.eps);
  ynext[2u * i] = y.x; ynext[2u * i + 1u] = y.y;
  adam[4u * i] = m.x; adam[4u * i + 1u] = m.y; adam[4u * i + 2u] = v.x; adam[4u * i + 3u] = v.y;
}`;

// Spectral initialisation: the two leading non-trivial eigenvectors of the normalised fuzzy graph D^-1/2 W D^-1/2, by
// power iteration on (I + D^-1/2 W D^-1/2) / 2 with the trivial vector D^1/2 1 projected out and the pair
// re-orthonormalised every iteration (umap-learn solves the same eigenproblem with a sparse eigensolver).
const WGSL_SPMV = /* wgsl */`
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> ew: array<f32>;
@group(0) @binding(4) var<storage, read> isd: array<f32>;   // 1 / sqrt(degree)
@group(0) @binding(5) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> z: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  var acc = vec2<f32>(0.0, 0.0);
  for (var e: u32 = off[i]; e < off[i + 1u]; e = e + 1u) { let j = idx[e]; acc = acc + ew[e] * isd[j] * vec2<f32>(x[2u * j], x[2u * j + 1u]); }
  let r = 0.5 * (vec2<f32>(x[2u * i], x[2u * i + 1u]) + isd[i] * acc);
  z[2u * i] = r.x; z[2u * i + 1u] = r.y;
}`;
const WGSL_DOTS = /* wgsl */`
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> z: array<f32>;
@group(0) @binding(2) var<storage, read> u0: array<f32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;   // z1.u0, z2.u0, z1.z1, z2.z2, z1.z2
var<workgroup> part: array<array<f32, 5>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var a = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  for (var i: u32 = l.x; i < U.n; i = i + 256u) { let z1 = z[2u * i]; let z2 = z[2u * i + 1u]; let u = u0[i]; a[0] += z1 * u; a[1] += z2 * u; a[2] += z1 * z1; a[3] += z2 * z2; a[4] += z1 * z2; }
  part[l.x] = a; workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st = st >> 1u) { if (l.x < st) { for (var c: u32 = 0u; c < 5u; c++) { part[l.x][c] += part[l.x + st][c]; } } workgroupBarrier(); }
  if (l.x == 0u) { for (var c: u32 = 0u; c < 5u; c++) { s[c] = part[0][c]; } }
}`;
const WGSL_ORTH = /* wgsl */`
struct P { n: u32, mode: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read_write> z: array<f32>;
@group(0) @binding(2) var<storage, read> u0: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  if (U.mode == 0u) { z[2u * i] -= s[0] * u0[i]; z[2u * i + 1u] -= s[1] * u0[i]; return; }   // drop the trivial direction
  let n1 = sqrt(max(s[2], 1e-30)); let c = s[4] / max(s[2], 1e-30); let n2 = sqrt(max(s[3] - s[4] * c, 1e-30));
  let z1 = z[2u * i]; let z2 = z[2u * i + 1u];
  x[2u * i] = z1 / n1; x[2u * i + 1u] = (z2 - c * z1) / n2;
}`;

// Topic colours: k-means on the GPU (assignment and centroid means; k-means++ seeding picks on the CPU from GPU distances).
const WGSL_ASSIGN = /* wgsl */`
struct P { n: u32, dim: u32, kc: u32, pad: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> cent: array<f32>;
@group(0) @binding(3) var<storage, read_write> lab: array<u32>;
@group(0) @binding(4) var<storage, read_write> dmin: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  var best: u32 = 0u; var bd: f32 = ${BIG};
  for (var c: u32 = 0u; c < U.kc; c = c + 1u) {
    var acc: f32 = 0.0; for (var d: u32 = 0u; d < U.dim; d = d + 1u) { let t = feat[i * U.dim + d] - cent[c * U.dim + d]; acc = acc + t * t; }
    if (acc < bd) { bd = acc; best = c; }
  }
  lab[i] = best; dmin[i] = bd;
}`;
const WGSL_CENT = /* wgsl */`
struct P { n: u32, dim: u32, kc: u32, pad: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> lab: array<u32>;
@group(0) @binding(3) var<storage, read_write> cent: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x; if (t >= U.kc * U.dim) { return; }
  let c = t / U.dim; let d = t % U.dim; var s: f32 = 0.0; var cnt: u32 = 0u;
  for (var i: u32 = 0u; i < U.n; i = i + 1u) { if (lab[i] == c) { s = s + feat[i * U.dim + d]; cnt = cnt + 1u; } }
  if (cnt > 0u) { cent[t] = s / f32(cnt); }
}`;

// Principal axes for the PCA start (engine.js pcaAxes, the live map's start): the two matrix-vector products of each
// power-iteration round run on the GPU; the dim-sized normalisation and deflation stay on the CPU.
const WGSL_MEAN_COLS = /* wgsl */`
struct P { n: u32, dim: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read_write> mean: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let d = gid.x; if (d >= U.dim) { return; } var s: f32 = 0.0; for (var i: u32 = 0u; i < U.n; i = i + 1u) { s = s + feat[i * U.dim + d]; } mean[d] = s / f32(U.n); }`;
const WGSL_PROJ = /* wgsl */`
struct P { n: u32, dim: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> u: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let i = gid.x; if (i >= U.n) { return; } var s: f32 = 0.0; for (var d: u32 = 0u; d < U.dim; d = d + 1u) { s = s + (feat[i * U.dim + d] - mean[d]) * v[d]; } u[i] = s; }`;
const WGSL_BACK = /* wgsl */`
struct P { n: u32, dim: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> w: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { let d = gid.x; if (d >= U.dim) { return; } var s: f32 = 0.0; for (var i: u32 = 0u; i < U.n; i = i + 1u) { s = s + (feat[i * U.dim + d] - mean[d]) * u[i]; } w[d] = s; }`;

// Alignment of a new fit onto the map on screen (orthogonal Procrustes with scale): one workgroup sums the centred
// cross-products, the 2 x 2 rotation or reflection is solved on the CPU, a second kernel moves every row.
const WGSL_ALIGN_SUMS = /* wgsl */`
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;   // reference (the map shown)
@group(0) @binding(2) var<storage, read> b: array<f32>;   // the new fit
@group(0) @binding(3) var<storage, read_write> s: array<f32>;   // ma.x, ma.y, mb.x, mb.y, Sxx, Sxy, Syx, Syy, |b|^2, |a|^2
var<workgroup> part: array<array<f32, 6>, 256>;
var<workgroup> mean: array<f32, 4>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var m = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var i: u32 = l.x; i < U.n; i = i + 256u) { m[0] += a[2u * i]; m[1] += a[2u * i + 1u]; m[2] += b[2u * i]; m[3] += b[2u * i + 1u]; }
  part[l.x] = m; workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st = st >> 1u) { if (l.x < st) { for (var c: u32 = 0u; c < 4u; c++) { part[l.x][c] += part[l.x + st][c]; } } workgroupBarrier(); }
  if (l.x == 0u) { for (var c: u32 = 0u; c < 4u; c++) { mean[c] = part[0][c] / f32(U.n); s[c] = mean[c]; } }
  workgroupBarrier();
  var q = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var i: u32 = l.x; i < U.n; i = i + 256u) {
    let a0 = a[2u * i] - mean[0]; let a1 = a[2u * i + 1u] - mean[1]; let b0 = b[2u * i] - mean[2]; let b1 = b[2u * i + 1u] - mean[3];
    q[0] += b0 * a0; q[1] += b0 * a1; q[2] += b1 * a0; q[3] += b1 * a1; q[4] += b0 * b0 + b1 * b1; q[5] += a0 * a0 + a1 * a1;
  }
  part[l.x] = q; workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st = st >> 1u) { if (l.x < st) { for (var c: u32 = 0u; c < 6u; c++) { part[l.x][c] += part[l.x + st][c]; } } workgroupBarrier(); }
  if (l.x == 0u) { for (var c: u32 = 0u; c < 6u; c++) { s[4u + c] = part[0][c]; } }
}`;
const WGSL_ALIGN_APPLY = /* wgsl */`
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32, r00: f32, r01: f32, r10: f32, r11: f32, ax: f32, ay: f32, bx: f32, by: f32 };
@group(0) @binding(0) var<uniform> U: P;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= U.n) { return; }
  let x = b[2u * i] - U.bx; let y = b[2u * i + 1u] - U.by;
  b[2u * i] = U.ax + U.r00 * x + U.r01 * y; b[2u * i + 1u] = U.ay + U.r10 * x + U.r11 * y;
}`;

// Exact lists of the km nearest other rows of the first n rows in featBuf (squared Euclidean summed in dimension order,
// ties to the lower row, sqrt distances): the tune mode's search (WGSL_D2Q, then WGSL_SELECT_PAR with its exact recheck)
// as a standalone call, for the base fit of a static map (StreamingEngine.fitBase). Same entries as the streaming
// engine's list refresh (WGSL_DISTANCE / TOPK_PARTIAL / TOPK_MERGE) with every row alive, in a fraction of the time.
const searchPipes = new WeakMap();
export async function exactLists(device, featBuf, n, dim, km, onProgress = null) {
  let P = searchPipes.get(device);
  if (!P) {
    P = {}; for (const [k, code] of Object.entries({ d2q: WGSL_D2Q, selectPar: WGSL_SELECT_PAR, selectRecheck: WGSL_SELECT_RECHECK })) P[k] = device.createComputePipeline({ label: k, layout: 'auto', compute: { module: device.createShaderModule({ code, label: k }), entryPoint: 'main' } });
    searchPipes.set(device, P);
  }
  const buf = (size, usage) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage }), S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_DST, R = GPUBufferUsage.COPY_SRC;
  const bind = (p, bufs) => device.createBindGroup({ layout: P[p].getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const budget = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize, 256 << 20);
  const Q = Math.max(16, Math.min(n, Math.floor(budget / (4 * n)) & ~15));
  const d2 = buf(Q * n * 4, S), IBuf = buf(n * km * 4, S | R), DBuf = buf(n * km * 4, S | R), flags = buf(n * 4, S | R), uD = buf(16, GPUBufferUsage.UNIFORM | C), uS = buf(16, GPUBufferUsage.UNIFORM | C);
  const gD = bind('d2q', [uD, featBuf, d2]), gS = bind('selectPar', [uS, d2, IBuf, DBuf, flags]), gR = bind('selectRecheck', [uS, d2, IBuf, DBuf, flags]);
  for (let q0 = 0; q0 < n; q0 += Q) {
    const qn = Math.min(Q, n - q0);
    device.queue.writeBuffer(uD, 0, new Uint32Array([n, dim, q0, qn])); device.queue.writeBuffer(uS, 0, new Uint32Array([n, qn, q0, km]));
    const enc = device.createCommandEncoder();
    let p = enc.beginComputePass(); p.setPipeline(P.d2q); p.setBindGroup(0, gD); p.dispatchWorkgroups(Math.ceil(n / 64), Math.ceil(qn / 64)); p.end();
    p = enc.beginComputePass(); p.setPipeline(P.selectPar); p.setBindGroup(0, gS); p.dispatchWorkgroups(qn); p.end();
    p = enc.beginComputePass(); p.setPipeline(P.selectRecheck); p.setBindGroup(0, gR); p.dispatchWorkgroups(qn); p.end();
    device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
    onProgress?.(Math.min(n, q0 + qn), n);
  }
  const read = async (b, bytes, Ctor) => {
    const st = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | C });
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(b, 0, st, 0, bytes); device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ); const out = new Ctor(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return out;
  };
  const I = await read(IBuf, n * km * 4, Uint32Array), D = await read(DBuf, n * km * 4, Float32Array);
  for (const b of [d2, IBuf, DBuf, flags, uD, uS]) b.destroy();
  return { I, D };
}

// The 2 x 2 solve of an alignment from the GPU's sums (WGSL_ALIGN_SUMS), and WGSL_ALIGN_APPLY's uniforms for it
function solveAlign([ax, ay, bx, by, sxx, sxy, syx, syy, bb, aa], n, { scale = true, reflect = true } = {}) {
  let best = null;
  for (const refl of reflect ? [1, -1] : [1]) {   // y' = refl * (sin x + cos y); maximise trace(R^T C)
    const th = Math.atan2(refl * sxy - syx, sxx + refl * syy), c = Math.cos(th), sn = Math.sin(th), tr = sxx * c - syx * sn + refl * (sxy * sn + syy * c);
    if (!best || tr > best.tr) best = { tr, c, sn, refl };
  }
  // scale by spread, not least squares: a fit from other start positions matches the map only loosely, and the
  // least-squares scale then shrinks it toward its centre (to a quarter of the size in one test)
  const sc = !scale ? 1 : bb > 0 && aa > 0 ? Math.sqrt(aa / bb) : 1, { c, sn, refl } = best, uni = new ArrayBuffer(48), u = new Uint32Array(uni), f = new Float32Array(uni);
  u[0] = n; f.set([sc * c, -sc * sn, refl * sc * sn, refl * sc * c, ax, ay, bx, by], 4);
  return { uni, out: { scale: sc, angle: Math.atan2(sn, c), flipped: refl < 0, shift: Math.hypot(ax - bx, ay - by) } };
}
// The same alignment on two plain position buffers, outside a fit: buf (n rows of x, y) is turned, mirrored if that fits
// better, scaled and shifted onto refBuf row for row, in place on the GPU (the model compare lines up two models' maps)
const alignPipes = new WeakMap();
export async function alignBuffers(device, refBuf, buf, n, opts = {}) {
  let P = alignPipes.get(device);
  if (!P) { P = {}; for (const [k, code] of Object.entries({ sums: WGSL_ALIGN_SUMS, apply: WGSL_ALIGN_APPLY })) P[k] = device.createComputePipeline({ label: 'align ' + k, layout: 'auto', compute: { module: device.createShaderModule({ code, label: 'align ' + k }), entryPoint: 'main' } }); alignPipes.set(device, P); }
  const bind = (p, bufs) => device.createBindGroup({ layout: P[p].getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const U = GPUBufferUsage, uS = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }), uA = device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST });
  const sb = device.createBuffer({ size: 64, usage: U.STORAGE | U.COPY_SRC }), st = device.createBuffer({ size: 40, usage: U.MAP_READ | U.COPY_DST });
  device.queue.writeBuffer(uS, 0, new Uint32Array([n, 0, 0, 0]));
  let enc = device.createCommandEncoder(), p = enc.beginComputePass(); p.setPipeline(P.sums); p.setBindGroup(0, bind('sums', [uS, refBuf, buf, sb])); p.dispatchWorkgroups(1); p.end();
  enc.copyBufferToBuffer(sb, 0, st, 0, 40); device.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ); const sums = new Float32Array(st.getMappedRange().slice(0)); st.unmap();
  const { uni, out } = solveAlign(sums, n, opts); device.queue.writeBuffer(uA, 0, uni);
  enc = device.createCommandEncoder(); p = enc.beginComputePass(); p.setPipeline(P.apply); p.setBindGroup(0, bind('apply', [uA, buf])); p.dispatchWorkgroups(Math.ceil(n / 64)); p.end(); device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone(); for (const b of [uS, uA, sb, st]) b.destroy();
  return out;
}

function rng32(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

export class LiveFit {
  constructor(device, feat, n, dim) {
    this.device = device; this.feat = feat; this.n = n; this.dim = dim; this.km = Math.min(K_MAX, n - 1);
    const S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_DST, R = GPUBufferUsage.COPY_SRC;
    this.featBuf = this.buf(feat.byteLength, S | C); device.queue.writeBuffer(this.featBuf, 0, feat);
    this.posBuf = this.buf(n * 8, S | C | R | GPUBufferUsage.VERTEX);
    this.nextPosBuf = this.buf(n * 8, S | C | R);
    this.adamBuf = this.buf(n * 16, S | C);
    this.pipes = {};
    for (const [k, code] of Object.entries({ d2: WGSL_D2, select: WGSL_SELECT, d2q: WGSL_D2Q, selectPar: WGSL_SELECT_PAR, selectRecheck: WGSL_SELECT_RECHECK, sigma: WGSL_SIGMA, sgd: WGSL_SGD, adam: WGSL_ADAM, spmv: WGSL_SPMV, dots: WGSL_DOTS, orth: WGSL_ORTH, assign: WGSL_ASSIGN, cent: WGSL_CENT, meanCols: WGSL_MEAN_COLS, proj: WGSL_PROJ, back: WGSL_BACK, alignSums: WGSL_ALIGN_SUMS, alignApply: WGSL_ALIGN_APPLY })) {
      this.pipes[k] = device.createComputePipeline({ label: k, layout: 'auto', compute: { module: device.createShaderModule({ code, label: k }), entryPoint: 'main' } });
    }
    this.uni = {}; for (const k of ['d2', 'select', 'sigma', 'sgd', 'adam', 'spec', 'orth0', 'orth1', 'km', 'pca', 'align', 'alignApply']) this.uni[k] = this.buf(64, GPUBufferUsage.UNIFORM | C);
    this.epoch = 0; this.step = 0; this.graph = null; this.groups = 1;
  }
  buf(size, usage) { return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage }); }
  bind(pipe, bufs) { return this.device.createBindGroup({ layout: this.pipes[pipe].getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) }); }
  async read(buf, bytes, Ctor = Float32Array) {
    const st = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, st, 0, bytes); this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ); const out = new Ctor(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return out;
  }
  run(pipe, group, x, y = 1, z = 1) { const enc = this.device.createCommandEncoder(), p = enc.beginComputePass(); p.setPipeline(this.pipes[pipe]); p.setBindGroup(0, group); p.dispatchWorkgroups(x, y, z); p.end(); this.device.queue.submit([enc.finish()]); }

  // Lists already computed elsewhere (the live map's exact 15 nearest rows of its base, Euclidean distances like these):
  // k up to their length needs no search of its own (an all-pairs search of 182,000 rows takes about a minute). A larger k
  // asks ensureK for the full lists.
  useLists(idx, dist, k) {
    const { n, device } = this;
    for (let i = 0; i < n * k; i++) if (idx[i] < 0 || idx[i] >= n) return false;   // lists that leave the rows: not usable here
    this.km = k; this.knnI = Uint32Array.from(idx.subarray(0, n * k)); this.knnD = Float32Array.from(dist.subarray(0, n * k));
    this.knnIBuf?.destroy(); this.knnDBuf?.destroy();
    this.knnIBuf = this.buf(n * k * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST); device.queue.writeBuffer(this.knnIBuf, 0, this.knnI);
    this.knnDBuf = this.buf(n * k * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST); device.queue.writeBuffer(this.knnDBuf, 0, this.knnD);
    this.msKnn = 0; this.borrowed = true; return true;
  }
  // the full lists (K_MAX), once: a k above the borrowed lists' waits for them; prefetch starts them in the background
  // noSearch: the lists came with the dataset (arXiv, 30 per row from the full vectors); k stays within them, nothing is searched
  async ensureK(k, onProgress) { if (k <= this.km || this.noSearch) return; if (!this.fullP) this.fullP = this.neighbours(onProgress, Math.min(K_MAX, this.n - 1)); await this.fullP; }
  prefetchLists() { if (this.borrowed && !this.fullP && !this.noSearch) this.fullP = this.neighbours(null, Math.min(K_MAX, this.n - 1)); }
  // Exact lists of the K_MAX nearest other rows (squared Euclidean, ties to the lower row), once per dataset.
  // The lists are built in buffers of their own and swapped in at the end, so lists already in use (borrowed ones) stay
  // valid while a search runs in the background.
  async neighbours(onProgress, km = this.km) {
    const t0 = performance.now(), { n, device } = this;
    const budget = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize, 256 << 20);
    const Q = Math.max(16, Math.min(n, Math.floor(budget / (4 * n)) & ~15));
    const d2 = this.buf(Q * n * 4, GPUBufferUsage.STORAGE);
    const IBuf = this.buf(n * km * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC), DBuf = this.buf(n * km * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const par = !this.onePerQuery, pD = par ? 'd2q' : 'd2', pS = par ? 'selectPar' : 'select';   // onePerQuery: the earlier selection, kept to check against
    const flags = par ? this.buf(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
    const gD = this.bind(pD, [this.uni.d2, this.featBuf, d2]), gS = this.bind(pS, par ? [this.uni.select, d2, IBuf, DBuf, flags] : [this.uni.select, d2, IBuf, DBuf]);
    const gR = par ? this.bind('selectRecheck', [this.uni.select, d2, IBuf, DBuf, flags]) : null;
    for (let q0 = 0; q0 < n; q0 += Q) {
      const qn = Math.min(Q, n - q0);
      device.queue.writeBuffer(this.uni.d2, 0, new Uint32Array([n, this.dim, q0, qn]));
      device.queue.writeBuffer(this.uni.select, 0, new Uint32Array([n, qn, q0, km]));
      const enc = device.createCommandEncoder();
      let p = enc.beginComputePass(); p.setPipeline(this.pipes[pD]); p.setBindGroup(0, gD); p.dispatchWorkgroups(Math.ceil(n / (par ? 64 : 16)), Math.ceil(qn / (par ? 64 : 16))); p.end();
      p = enc.beginComputePass(); p.setPipeline(this.pipes[pS]); p.setBindGroup(0, gS); p.dispatchWorkgroups(par ? qn : Math.ceil(qn / 64)); p.end();
      if (par) { p = enc.beginComputePass(); p.setPipeline(this.pipes.selectRecheck); p.setBindGroup(0, gR); p.dispatchWorkgroups(qn); p.end(); }
      device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
      onProgress?.(Math.min(n, q0 + qn), n);
    }
    d2.destroy();
    if (flags) { const fl = await this.read(flags, n * 4, Uint32Array); let c = 0; for (let i = 0; i < n; i++) c += fl[i]; this.knnRechecked = c; flags.destroy(); }
    const I = await this.read(IBuf, n * km * 4, Uint32Array), D = await this.read(DBuf, n * km * 4);
    this.knnIBuf?.destroy(); this.knnDBuf?.destroy(); this.knnIBuf = IBuf; this.knnDBuf = DBuf; this.knnI = I; this.knnD = D; this.km = km; this.borrowed = false;
    this.msKnn = performance.now() - t0; return this.msKnn;
  }

  // Fuzzy graph for k: per-row weights on the GPU, then the symmetric fuzzy union w + w' - w w' (CPU bookkeeping, one
  // stamp array, no sort), edges below wmax / epochs dropped as umap-learn does. Uploads CSR (Adam, spectral) and the
  // edge list with fresh sample clocks (SGD).
  async setK(k, epochs) {
    const t0 = performance.now(), { n, km, device, knnI: I } = this; k = Math.max(2, Math.min(km, k | 0));   // the lists as they are now (a background search may swap in new ones meanwhile)
    const wBuf = this.buf(n * k * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    device.queue.writeBuffer(this.uni.sigma, 0, new Uint32Array([n, km, k, 0]));
    this.run('sigma', this.bind('sigma', [this.uni.sigma, this.knnDBuf, wBuf]), Math.ceil(n / 64));
    const W = await this.read(wBuf, n * k * 4); wBuf.destroy();
    // the bookkeeping below is about 170 ms of CPU at 91,000 rows and k 30: it runs in blocks of rows and gives the page
    // its turn every 8 ms, so tune mode keeps drawing frames while k changes (same result: the rows go in the same order)
    const yieldFrame = () => new Promise((r) => setTimeout(r, 0));
    const blocks = async (body) => { let t = performance.now(); for (let i0 = 0; i0 < n; i0 += 2048) { body(i0, Math.min(n, i0 + 2048)); if (performance.now() - t > 8) { await yieldFrame(); t = performance.now(); } } };
    const inCnt = new Uint32Array(n + 1);
    await blocks((i0, i1) => { for (let i = i0; i < i1; i++) for (let c = 0; c < k; c++) inCnt[I[i * km + c] + 1]++; });
    for (let i = 0; i < n; i++) inCnt[i + 1] += inCnt[i];
    const inSrc = new Uint32Array(n * k), inW = new Float32Array(n * k), fill = inCnt.slice(0, n);
    await blocks((i0, i1) => { for (let i = i0; i < i1; i++) for (let c = 0; c < k; c++) { const j = I[i * km + c], p = fill[j]++; inSrc[p] = i; inW[p] = W[i * k + c]; } });
    const off = new Uint32Array(n + 1), idx = new Uint32Array(2 * n * k), ew = new Float32Array(2 * n * k), stamp = new Int32Array(n).fill(-1), at = new Uint32Array(n);
    let E = 0;
    await blocks((i0, i1) => { for (let i = i0; i < i1; i++) {
      off[i] = E;
      for (let c = 0; c < k; c++) { const j = I[i * km + c]; stamp[j] = i; at[j] = E; idx[E] = j; ew[E] = W[i * k + c]; E++; }
      for (let p = inCnt[i]; p < inCnt[i + 1]; p++) { const j = inSrc[p], w = inW[p]; if (stamp[j] === i) { const e = at[j]; ew[e] = ew[e] + w - ew[e] * w; } else { idx[E] = j; ew[E] = w; E++; } }
    } });
    off[n] = E;
    // each row's edges in neighbour-row order, as in umap-learn's sorted COO matrix: SGD applies a row's edges in turn,
    // and with early steps near the clip the last ones decide where a row lands, so the order matters (on 15,144 Bluesky
    // posts, own neighbours first then incoming edges kept 0.12 of the 15 nearest after 200 epochs; row order kept 0.165)
    await blocks((i0, i1) => { for (let i = i0; i < i1; i++) for (let e = off[i] + 1; e < E && e < off[i + 1]; e++) { const j = idx[e], wj = ew[e]; let q = e - 1; while (q >= off[i] && idx[q] > j) { idx[q + 1] = idx[q]; ew[q + 1] = ew[q]; q--; } idx[q + 1] = j; ew[q + 1] = wj; } });
    let wmax = 0; for (let e = 0; e < E; e++) if (ew[e] > wmax) wmax = ew[e];
    const cut = wmax / Math.max(1, epochs), o2 = new Uint32Array(n + 1), i2 = new Uint32Array(E), w2 = new Float32Array(E), deg = new Float64Array(n);
    let m = 0; await blocks((i0, i1) => { for (let i = i0; i < i1; i++) { o2[i] = m; for (let e = off[i]; e < off[i + 1]; e++) if (ew[e] >= cut) { i2[m] = idx[e]; w2[m] = ew[e]; deg[i] += ew[e]; m++; } } }); o2[n] = m;
    this.freeGraph();
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, up = (arr) => { const b = this.buf(arr.byteLength, S); device.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength); return b; };
    const isd = new Float32Array(n), u0 = new Float32Array(n); let un = 0; for (let i = 0; i < n; i++) { isd[i] = deg[i] > 0 ? 1 / Math.sqrt(deg[i]) : 0; un += deg[i]; } un = Math.sqrt(un) || 1; for (let i = 0; i < n; i++) u0[i] = Math.sqrt(deg[i]) / un;
    const nextS = new Float32Array(m); for (let e = 0; e < m; e++) nextS[e] = this.epoch + wmax / w2[e];
    this.graph = { k, epochs, E: m, wmax, wCpu: w2, off: up(o2), idx: up(i2), w: up(w2), next: up(nextS), isd: up(isd), u0: up(u0) };
    this.bindLayout();
    this.msGraph = performance.now() - t0; return this.graph;
  }
  freeGraph() { if (!this.graph) return; for (const k of ['off', 'idx', 'w', 'next', 'isd', 'u0']) this.graph[k].destroy(); this.graph = null; }
  bindLayout() {
    const G = this.graph;
    this.gSgd = this.bind('sgd', [this.uni.sgd, G.off, G.idx, G.w, G.next, this.posBuf, this.nextPosBuf]);
    this.gAdam = this.bind('adam', [this.uni.adam, G.off, G.idx, G.w, this.posBuf, this.nextPosBuf, this.adamBuf]);
  }
  // Back to epoch 0 with every sample clock at its first due epoch, so a refit with the same settings repeats exactly.
  // clocks = false (Adam, which samples no edges by clock): the clocks are left as they are.
  rewind(clocks = true) { const G = this.graph; this.epoch = 0; this.resetAdam(); if (!G || !clocks) return; if (!G.next0) { G.next0 = new Float32Array(G.E); for (let e = 0; e < G.E; e++) G.next0[e] = G.wmax / G.wCpu[e]; } this.device.queue.writeBuffer(G.next, 0, G.next0); }
  setPositions(xy) { this.device.queue.writeBuffer(this.posBuf, 0, xy); this.device.queue.writeBuffer(this.nextPosBuf, 0, xy); this.resetAdam(); }
  resetAdam() { const enc = this.device.createCommandEncoder(); enc.clearBuffer(this.adamBuf); this.device.queue.submit([enc.finish()]); this.step = 0; }
  randomInit(seed = 1) { const r = rng32(seed), y = new Float32Array(this.n * 2); for (let i = 0; i < y.length; i++) y[i] = r() * 20 - 10; return y; }
  // engine.js pcaInit on the GPU: same start vector (mulberry32(7)), 25 rounds, deflated second axis, scaled to sd per axis
  async pcaInit(sd = 3.0) {
    const { n, dim, device } = this, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const mean = this.buf(dim * 4, S), v = this.buf(dim * 4, S), u = this.buf(n * 4, S), w = this.buf(dim * 4, S);
    device.queue.writeBuffer(this.uni.pca, 0, new Uint32Array([n, dim, 0, 0]));
    this.run('meanCols', this.bind('meanCols', [this.uni.pca, this.featBuf, mean]), Math.ceil(dim / 64));
    const gP = this.bind('proj', [this.uni.pca, this.featBuf, mean, v, u]), gB = this.bind('back', [this.uni.pca, this.featBuf, mean, u, w]);
    const rng = rng32(7), comps = [], out = new Float32Array(n * 2);
    for (let c = 0; c < 2; c++) {
      let vv = new Float64Array(dim); for (let d = 0; d < dim; d++) vv[d] = rng() - 0.5;
      for (let it = 0; it < 25; it++) {
        device.queue.writeBuffer(v, 0, Float32Array.from(vv)); this.run('proj', gP, Math.ceil(n / 64)); this.run('back', gB, Math.ceil(dim / 64));
        const ww = Float64Array.from(await this.read(w, dim * 4));
        if (c === 1) { const p = comps[0]; let dot = 0; for (let d = 0; d < dim; d++) dot += ww[d] * p[d]; for (let d = 0; d < dim; d++) ww[d] -= dot * p[d]; }
        let nrm = 0; for (let d = 0; d < dim; d++) nrm += ww[d] * ww[d]; nrm = Math.sqrt(nrm) || 1; for (let d = 0; d < dim; d++) vv[d] = ww[d] / nrm;
      }
      comps.push(vv); device.queue.writeBuffer(v, 0, Float32Array.from(vv)); this.run('proj', gP, Math.ceil(n / 64));
      const uu = await this.read(u, n * 4); let m = 0, m2 = 0; for (let i = 0; i < n; i++) { m += uu[i]; m2 += uu[i] * uu[i]; } m /= n; const sdc = Math.sqrt(Math.max(1e-12, m2 / n - m * m));
      for (let i = 0; i < n; i++) out[2 * i + c] = (uu[i] / sdc) * sd;
    }
    for (const b of [mean, v, u, w]) b.destroy();
    return out;
  }
  async spectralInit(iters = 400, seed = 1) {
    const { n, device } = this, G = this.graph, r = rng32(seed);
    const x0 = new Float32Array(n * 2); for (let i = 0; i < x0.length; i++) x0[i] = r() - 0.5;
    const xb = this.buf(n * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC), zb = this.buf(n * 8, GPUBufferUsage.STORAGE), sb = this.buf(32, GPUBufferUsage.STORAGE);
    device.queue.writeBuffer(xb, 0, x0);
    for (const k of ['spec', 'orth0']) device.queue.writeBuffer(this.uni[k], 0, new Uint32Array([n, 0, 0, 0])); device.queue.writeBuffer(this.uni.orth1, 0, new Uint32Array([n, 1, 0, 0]));
    const gM = this.bind('spmv', [this.uni.spec, G.off, G.idx, G.w, G.isd, xb, zb]), gD = this.bind('dots', [this.uni.spec, zb, G.u0, sb]);
    const g0 = this.bind('orth', [this.uni.orth0, zb, G.u0, sb, xb]), g1 = this.bind('orth', [this.uni.orth1, zb, G.u0, sb, xb]), wg = Math.ceil(n / 64);
    for (let it = 0; it < iters; it += 50) {
      const enc = device.createCommandEncoder();
      for (let s = it; s < Math.min(iters, it + 50); s++) {
        const p = enc.beginComputePass();
        p.setPipeline(this.pipes.spmv); p.setBindGroup(0, gM); p.dispatchWorkgroups(wg);
        p.setPipeline(this.pipes.dots); p.setBindGroup(0, gD); p.dispatchWorkgroups(1);
        p.setPipeline(this.pipes.orth); p.setBindGroup(0, g0); p.dispatchWorkgroups(wg);
        p.setPipeline(this.pipes.dots); p.setBindGroup(0, gD); p.dispatchWorkgroups(1);
        p.setPipeline(this.pipes.orth); p.setBindGroup(0, g1); p.dispatchWorkgroups(wg);
        p.end();
      }
      device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
    }
    const y = await this.read(xb, n * 8); xb.destroy(); zb.destroy(); sb.destroy();
    let mx = 0; for (let i = 0; i < y.length; i++) mx = Math.max(mx, Math.abs(y[i]));
    const s = 10 / (mx || 1); for (let i = 0; i < y.length; i++) y[i] = y[i] * s + (r() - 0.5) * 2e-4;   // umap-learn: largest coordinate 10, plus a little noise
    return y;
  }

  // One SGD epoch (umap-learn) or one Adam step (this page's base-fit gradient). p: { a, b, alpha, gamma, negRate, seed }.
  // groups > 1: rows split into that many fixed groups (by hash) updated one after another within the epoch, each group
  // reading the positions the groups before it left; groups = 1 moves every row at once from the previous epoch.
  sgdEpoch(p, groups = this.groups) {
    const G = this.graph, wg = Math.ceil(this.n / 64);
    for (let g = 0; g < groups; g++) {
      const buf = new ArrayBuffer(48), u = new Uint32Array(buf), f = new Float32Array(buf);
      u[0] = this.n; u[1] = p.seed >>> 0; u[2] = groups; u[3] = g; f[4] = this.epoch; f[5] = p.alpha; f[6] = p.a; f[7] = p.b; f[8] = p.gamma; f[9] = p.negRate; f[10] = G.wmax;
      this.device.queue.writeBuffer(this.uni.sgd, 0, buf);
      const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass(); pass.setPipeline(this.pipes.sgd); pass.setBindGroup(0, this.gSgd); pass.dispatchWorkgroups(wg); pass.end();
      enc.copyBufferToBuffer(this.nextPosBuf, 0, this.posBuf, 0, this.n * 8); this.device.queue.submit([enc.finish()]);
    }
    this.epoch++;
  }
  adamStep(p) {
    const buf = new ArrayBuffer(48), u = new Uint32Array(buf), f = new Float32Array(buf);
    u[0] = this.n; u[1] = this.step; u[2] = p.seed >>> 0; u[3] = Math.max(0, Math.round(p.negRate)); f[4] = p.a; f[5] = p.b; f[6] = p.alpha; f[7] = p.gamma; f[8] = 0.9; f[9] = 0.999; f[10] = 1e-8; f[11] = 4;
    this.device.queue.writeBuffer(this.uni.adam, 0, buf);
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass(); pass.setPipeline(this.pipes.adam); pass.setBindGroup(0, this.gAdam); pass.dispatchWorkgroups(Math.ceil(this.n / 64)); pass.end();
    enc.copyBufferToBuffer(this.nextPosBuf, 0, this.posBuf, 0, this.n * 8); this.device.queue.submit([enc.finish()]);
    this.step++; this.epoch++;
  }

  // k-means topic labels (k-means++ seeding with mulberry32(seed), then Lloyd rounds), all distances on the GPU.
  async kmeans(kc, iters = 15, seed = 1) {
    const { n, dim, device } = this, r = rng32(seed);
    const cent = new Float32Array(kc * dim), cBuf = this.buf(kc * dim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const lBuf = this.buf(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC), dBuf = this.buf(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const gA = this.bind('assign', [this.uni.km, this.featBuf, cBuf, lBuf, dBuf]), gC = this.bind('cent', [this.uni.km, this.featBuf, lBuf, cBuf]);
    const feat = this.feat;
    const pick = (row, c) => cent.set(feat.subarray(row * dim, (row + 1) * dim), c * dim);
    pick(Math.floor(r() * n), 0);
    for (let c = 1; c < kc; c++) {
      device.queue.writeBuffer(cBuf, 0, cent.subarray(0, c * dim)); device.queue.writeBuffer(this.uni.km, 0, new Uint32Array([n, dim, c, 0]));
      this.run('assign', gA, Math.ceil(n / 64)); const dm = await this.read(dBuf, n * 4);
      let tot = 0; for (let i = 0; i < n; i++) tot += dm[i]; let t = r() * tot, row = n - 1; for (let i = 0; i < n; i++) { t -= dm[i]; if (t <= 0) { row = i; break; } }
      pick(row, c);
    }
    device.queue.writeBuffer(cBuf, 0, cent); device.queue.writeBuffer(this.uni.km, 0, new Uint32Array([n, dim, kc, 0]));
    let lab = null;
    for (let it = 0; it < iters; it++) {
      this.run('assign', gA, Math.ceil(n / 64)); const nl = await this.read(lBuf, n * 4, Uint32Array);
      let changed = 0; if (lab) for (let i = 0; i < n; i++) if (lab[i] !== nl[i]) changed++; lab = nl;
      if (it > 0 && !changed) break;
      this.run('cent', gC, Math.ceil((kc * dim) / 64));
    }
    cBuf.destroy(); lBuf.destroy(); dBuf.destroy();
    const cnt = new Int32Array(kc); for (let i = 0; i < n; i++) cnt[lab[i]]++;
    const order = [...Array(kc).keys()].sort((a, b) => cnt[b] - cnt[a]), remap = new Uint16Array(kc); order.forEach((c, i) => { remap[c] = i; });
    const out = new Uint16Array(n); for (let i = 0; i < n; i++) out[i] = remap[lab[i]]; return out;
  }
  // Rotate (or reflect), scale and shift the current positions onto refBuf's, row for row: a fit from new start positions
  // then lands where the shown map is instead of turned or mirrored. Returns the scale used.
  async alignTo(refBuf, { scale = true, reflect = true } = {}) {
    const { n, device } = this, sb = this.buf(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    device.queue.writeBuffer(this.uni.align, 0, new Uint32Array([n, 0, 0, 0]));
    this.run('alignSums', this.bind('alignSums', [this.uni.align, refBuf, this.posBuf, sb]), 1);
    const sums = await this.read(sb, 40); sb.destroy();
    const { uni, out } = solveAlign(sums, n, { scale, reflect });
    device.queue.writeBuffer(this.uni.alignApply, 0, uni);
    const enc = device.createCommandEncoder(), p = enc.beginComputePass(); p.setPipeline(this.pipes.alignApply); p.setBindGroup(0, this.bind('alignApply', [this.uni.alignApply, this.posBuf])); p.dispatchWorkgroups(Math.ceil(n / 64)); p.end();
    enc.copyBufferToBuffer(this.posBuf, 0, this.nextPosBuf, 0, n * 8); device.queue.submit([enc.finish()]);
    return out;
  }
  // the positions now, kept as the reference a move is pinned to (alignTo(this.refBuf, { scale: false }))
  markReference() { if (!this.refBuf) this.refBuf = this.buf(this.n * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC); const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(this.posBuf, 0, this.refBuf, 0, this.n * 8); this.device.queue.submit([enc.finish()]); }
  async positions() { return this.read(this.posBuf, this.n * 8); }
  destroy() { this.freeGraph(); for (const b of [this.refBuf, this.featBuf, this.posBuf, this.nextPosBuf, this.adamBuf, this.knnIBuf, this.knnDBuf, ...Object.values(this.uni)]) b?.destroy(); }
}

// ---- one fit with fixed settings, the same sequence the tuner runs on "refit from the start", so the live map can use a
// tuned base (?base=tuned&tuned=...) and get the map the tuner showed for those rows and settings.
// settings: { k, minDist, spread, optimizer: 'sgd' | 'adam', init: 'spectral' | 'pca' | 'random', lr, anneal, epochs, gamma, negRate }
export function epochParamsFor(s, ab, done) {
  const alpha = s.lr * (s.anneal ? Math.max(0, 1 - done / s.epochs) : 1);
  return { a: ab.a, b: ab.b, alpha, gamma: s.gamma, negRate: s.negRate, seed: 1 + done };
}
export const abFor = (s) => findAB(Math.min(s.minDist, s.spread), s.spread);
export async function startFor(fit, init) { return init === 'spectral' ? fit.spectralInit() : init === 'pca' ? fit.pcaInit(3.0) : fit.randomInit(1); }
export async function tunedLayout(device, feat, n, dim, s, onProgress = null) {
  const t0 = performance.now(), fit = new LiveFit(device, feat, n, dim);
  try {
    onProgress?.('neighbour lists'); await fit.neighbours();
    onProgress?.('fuzzy graph'); await fit.setK(s.k, s.epochs);
    onProgress?.(`${s.init} start`); fit.setPositions(await startFor(fit, s.init)); fit.rewind();
    const ab = abFor(s);
    onProgress?.(`${s.epochs} ${s.optimizer === 'sgd' ? 'epochs' : 'steps'}`);
    for (let done = 0; done < s.epochs; done++) { const p = epochParamsFor(s, ab, done); if (s.optimizer === 'sgd') fit.sgdEpoch(p); else fit.adamStep(p); if (done % 100 === 99) await device.queue.onSubmittedWorkDone(); }
    const xy = await fit.positions();
    for (let i = 0; i < xy.length; i++) if (!Number.isFinite(xy[i])) throw new Error('non-finite position in the tuned fit');
    return { xy, edges: fit.graph.E, ab, ms: performance.now() - t0 };
  } finally { fit.destroy(); }
}
// settings <-> a short URL value, e.g. k15_md0.1_sp1_sgd_spectral_lr1_an1_ep200_g1_ng5
export function tunedToParam(s) { return `k${s.k}_md${s.minDist}_sp${s.spread}_${s.optimizer}_${s.init}_lr${+s.lr.toPrecision(3)}_an${s.anneal ? 1 : 0}_ep${s.epochs}_g${s.gamma}_ng${s.negRate}`; }
export function tunedFromParam(v) {
  const t = String(v || '').split('_'), num = (p) => { const x = t.find((y) => y.startsWith(p) && /^-?[\d.]+$/.test(y.slice(p.length))); if (!x) throw new Error(`tuned base: no ${p} in "${v}"`); return +x.slice(p.length); };
  const optimizer = t.includes('adam') ? 'adam' : t.includes('sgd') ? 'sgd' : null, init = ['spectral', 'pca', 'random'].find((x) => t.includes(x));
  if (!optimizer || !init) throw new Error(`tuned base: optimiser or start missing in "${v}"`);
  const s = { k: num('k'), minDist: num('md'), spread: num('sp'), optimizer, init, lr: num('lr'), anneal: num('an') === 1, epochs: num('ep'), gamma: num('g'), negRate: num('ng') };
  if (s.k < 2 || s.k > K_MAX || s.epochs < 1 || s.spread <= 0) throw new Error(`tuned base: out of range in "${v}"`);
  return s;
}
