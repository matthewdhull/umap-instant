// A static dataset as a source (datasets.js): this worker fetches the whole table, orders its rows, gives every row a
// vector and posts the whole window to the page in the recording worker's protocol (ready, frames, backfilled), so the
// first half of the rows becomes the base map and the rest the stream. Nothing arrives after the window.
// Vectors: text through MiniLM-L6 (mean pooling, 128 tokens, L2-normalised: the live feeds' recipe), images through
// DINOv2-small (the class token) or CLIP ViT-B/32 (?model=clip: its projected image vector), table rows as standardised numbers (log for long-tailed positive columns, one-hot for
// categories), ASRS through its packed TF-IDF features. Text and images are embedded on this machine's GPU through the
// page server (serve.py POST /embed, torch), as Embedding Atlas embeds its examples ahead of time, when it answers;
// otherwise in the browser (transformers.js on WebGPU). Embedded windows are kept in IndexedDB, so a second visit to the
// same dataset skips the download and the encoder.
// Time: a table with dates is placed on a real time axis (the window maps linearly onto the dates, rows on the same date
// spread evenly up to the next date); the page labels it in the dataset's unit.
import { DATASETS, TEXT_MODELS, IMAGE_MODELS, modelsFor, defaultModel } from './datasets.js';
import { uploadDataset, getFile, readRows, vectorOf } from './upload.js';

let TM = TEXT_MODELS.minilm, TM_KEY = 'minilm', IM = IMAGE_MODELS.dinov2, IM_KEY = 'dinov2';
const TF = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3', HYP = 'https://cdn.jsdelivr.net/npm/hyparquet@1.31.2/+esm', HYPC = 'https://cdn.jsdelivr.net/npm/hyparquet-compressors@1.1.1/+esm';
const FRAME = 2048, CACHE_DB = 'jetstream-static-datasets';
let running = false;
const stats = { events: 0, post_creates: 0, kept: 0, embedded: 0, embed_ms: 0, batches: 0, not_english: 0, too_short: 0, link_only: 0, duplicate: 0 };
const note = (msg) => postMessage({ type: 'note', msg }), log = (msg) => postMessage({ type: 'log', msg });
const mb = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0) + ' MB';

// ---- Files served with the site (static-data/ next to the page on the public site; the local data folder in development):
// ASRS's packed rows and vectors, and the precomputed vectors of the text and image datasets, so a visitor's browser loads
// vectors instead of embedding a whole table. Precomputed vectors are used only when their fingerprint of the rows (in the
// final order) matches this load's rows; anything else embeds as before.
const BASES = ['./static-data/', '../data/jetstream_demo/static/'];
async function findStatic(key, file) {
  for (const b of BASES) { try { const r = await fetch(`${b}${key}/${file}`, { cache: 'no-store' }); if (r.ok) return { base: `${b}${key}/`, json: await r.json() }; } catch { /* next */ } }
  return null;
}
const fetchBuf = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.arrayBuffer(); });
async function fetchParts(base, parts) { const bufs = await Promise.all(parts.map((p) => fetchBuf(base + p))); const n = bufs.reduce((a, b) => a + b.byteLength, 0), out = new Uint8Array(n); let o = 0; for (const b of bufs) { out.set(new Uint8Array(b), o); o += b.byteLength; } return out.buffer; }
function fnv(strings) { let h = 0x811c9dc5; for (const s of strings) { for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; } h ^= 10; h = Math.imul(h, 16777619) >>> 0; } return (h >>> 0).toString(16).padStart(8, '0') + ':' + strings.length; }
// float16 <-> float32 (IEEE half): the precomputed vectors are stored as halves (wine's would be 252 MB as float32)
function toHalf(f) { const b = new DataView(new ArrayBuffer(4)); b.setFloat32(0, f); const x = b.getUint32(0), s = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff; if (e <= 0) { if (e < -10) return s; m |= 0x800000; const t = 14 - e; return s | ((m + (1 << (t - 1))) >> t); } if (e >= 31) return s | 0x7c00; const h = s | (e << 10) | (m >> 13); return (m & 0x1000) ? h + 1 : h; }
function fromHalves(u16) { const out = new Float32Array(u16.length); for (let i = 0; i < u16.length; i++) { const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); } return out; }
// the default model's vectors are vectors.json (MiniLM for text, DINOv2 for images, as before), another model's
// vectors.<model>.json
const modelKey = (kind) => (kind === 'text' ? TM_KEY : kind === 'image' ? IM_KEY : null);
const vecTag = (kind) => (modelKey(kind) && modelKey(kind) !== defaultModel(kind) ? modelKey(kind) : '');
const vecFile = (kind) => (vecTag(kind) ? `vectors.${vecTag(kind)}.json` : 'vectors.json');
async function precomputed(key, rowsId, kind = 'text') {
  const f = await findStatic(key, vecFile(kind)); if (!f || f.json.rowsId !== rowsId || f.json.model !== (kind === 'text' ? TM : IM).name) return null;
  note(`Reading ${DATASETS[key].label}: precomputed vectors…`); const v = fromHalves(new Uint16Array(await fetchParts(f.json.base || f.base, f.json.parts)));   // base: the parts' own host (Hugging Face on the site)
  return { vecs: v, dim: f.json.dim, model: f.json.model, hf: f.json.model, device: 'precomputed', loadMs: 0 };
}
// ?export=1 on this machine's page: the vectors just embedded go to serve.py's /save, in parts under 10 MB, for the site
async function exportVectors(key, w) {
  const PART = 8 << 20, half = new Uint16Array(w.vecs.length); for (let i = 0; i < half.length; i++) half[i] = toHalf(w.vecs[i]);
  const bytes = new Uint8Array(half.buffer), parts = [];
  const kind = DATASETS[key].kind, tag = vecTag(kind) ? `${vecTag(kind)}_` : '';
  for (let o = 0, j = 0; o < bytes.length; o += PART, j++) { const name = `vectors_${tag}${j}.f16`; await fetch(`/save?path=${key}/${name}`, { method: 'POST', body: bytes.subarray(o, o + PART) }); parts.push(name); }
  await fetch(`/save?path=${key}/${vecFile(kind)}`, { method: 'POST', body: JSON.stringify({ n: w.posts.length, dim: w.dim, model: w.model, rowsId: w.rowsId, parts }) });
  log(`${key}: exported ${w.posts.length} vectors (${parts.length} parts)`);
}

// ---- fetching: whole files with progress and a SHA-256 check; Parquet by columns over range requests
async function fetchWhole(url, sha256, label) {
  const res = await fetch(url); if (!res.ok) throw new Error(`${label}: HTTP ${res.status}`);
  const total = +res.headers.get('content-length') || 0, reader = res.body.getReader(), parts = []; let got = 0, last = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; if (performance.now() - last > 200) { last = performance.now(); note(`Downloading ${label}: ${mb(got)}${total ? ' of ' + mb(total) : ''}`); } }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  if (sha256) {
    const hex = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buf))].map((b) => b.toString(16).padStart(2, '0')).join('');
    if (hex !== sha256) throw new Error(`${label}: the file has changed since it was pinned (sha256 ${hex.slice(0, 12)}…)`);
  }
  return buf;
}
function parseCsv(text) {   // RFC 4180: quoted fields, doubled quotes, newlines inside quotes
  const rows = []; let row = [], f = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true; else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(f); f = ''; if (row.length > 1 || row[0] !== '') rows.push(row); row = []; }
    else f += c;
  }
  if (f !== '' || row.length) { row.push(f); rows.push(row); }
  const head = rows.shift(); return rows.map((r) => Object.fromEntries(head.map((h, j) => [h, r[j] ?? ''])));
}
async function readTable(D, label) {
  if (D.upload) { note(`Reading ${label}…`); return readRows(await getFile(), D.format); }   // your own file, from this browser's storage
  if (D.format === 'parquet') {
    const { asyncBufferFromUrl, parquetMetadataAsync, parquetReadObjects } = await import(HYP), { compressors } = await import(HYPC);
    // a file up to 100 MB comes down in one request (much faster than a range request per row group); a larger one
    // (ScienceQA's, with its images) is read by range requests, only the columns used here
    let file = await asyncBufferFromUrl({ url: D.url });
    if (file.byteLength <= 100 * 1048576) file = (await fetchWhole(D.url, D.sha256, label)).buffer; else note(`Reading ${label} (only the columns used)…`);
    const metadata = await parquetMetadataAsync(file), n = Number(metadata.num_rows), out = [];
    const columns = D.columns || [...(D.numeric || []), ...(D.categorical || [])];
    for (let lo = 0; lo < n; lo += 20000) {   // in slices, so the status moves
      out.push(...await parquetReadObjects({ file, metadata, compressors, columns, utf8: D.kind !== 'image', rowStart: lo, rowEnd: Math.min(n, lo + 20000) }));
      note(`Reading ${label}: ${Math.min(n, lo + 20000).toLocaleString()} of ${n.toLocaleString()} rows`);
    }
    return out;
  }
  const text = new TextDecoder().decode(await fetchWhole(D.url, D.sha256, label));
  if (D.format === 'csv') return parseCsv(text);
  if (D.format === 'jsonl') return text.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
  return JSON.parse(text);
}

// ---- order and sample: by the table's date when it has one, otherwise a fixed shuffle; all rows unless `cap` is set
function mulberry(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function pick(n, cap, orderKey) {
  const idx = Array.from({ length: n }, (_, i) => i), rnd = mulberry(20261017);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const take = cap > 0 ? idx.slice(0, Math.min(cap, n)) : idx;
  if (orderKey) take.sort((a, b) => (orderKey[a] - orderKey[b]) || (a - b));
  return take;
}

// ---- vectors
// the page server's GPU encoder (serve.py), if it answers
let gpuInfo, useGpu = true, exporting = false;   // ?gpu=0 on the page: embed in the browser even when the server's GPU encoder answers
async function gpu() {
  if (gpuInfo !== undefined) return gpuInfo;
  try { const r = await fetch(new URL('/embed', self.location.href), { signal: AbortSignal.timeout(2000) }); const j = r.ok ? await r.json() : null; gpuInfo = j && j.ok ? j : null; } catch { gpuInfo = null; }
  return gpuInfo;
}
async function gpuEmbed(body) {
  body = { ...body, model: body.kind === 'text' ? TM_KEY : IM_KEY };   // serve.py embeds with the same model (sentence-transformers, transformers)
  const r = await fetch(new URL('/embed', self.location.href), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`GPU encoder: HTTP ${r.status} ${r.statusText}`);
  return { v: new Float32Array(await r.arrayBuffer()), dim: +r.headers.get('X-Dim') };
}
// While embedding, the page is shown what comes out: a thinned sample of the vectors (at most about 40,000 dots in all, its
// growing preview cloud) and, per batch, a few rows with their tokens (or picture) and vector for the ticker. Sent without
// waiting, so it never slows the embedding.
let tokenizer = null;
async function tokensOf(text) {
  try { tf = tf || await import(TF); tf.env.allowLocalModels = false; tokenizer = tokenizer || await tf.AutoTokenizer.from_pretrained(TM.id); return tokenizer.tokenize(text).slice(0, 24); }
  catch { return text.split(/\s+/).slice(0, 24); }
}
let peeked = false;
async function peek(out, dim, ids, items, kind, total) {
  peeked = true;
  const every = Math.max(1, Math.ceil(total / 40000)), keep = ids.filter((i) => i % every === 0), v = new Float32Array(keep.length * dim);
  keep.forEach((i, r) => v.set(out.subarray(i * dim, (i + 1) * dim), r * dim));
  const picks = []; for (let k = 0; k < 4 && ids.length && kind !== 'none'; k++) picks.push(ids[Math.floor((k + 0.5) / 4 * ids.length)]);
  const samples = await Promise.all(picks.map(async (i) => ({ vec: out.slice(i * dim, (i + 1) * dim), ...(kind === 'text' ? { text: items[i], tokens: await tokensOf(items[i]) } : { img: 'data:image/jpeg;base64,' + items[i] }) })));
  const rows = Uint32Array.from(keep);   // each sampled vector's row in the table (the page seeds the final fit from the preview by them)
  postMessage({ type: 'peek', dim, vecs: v, rows, samples, done: ids.length }, [v.buffer, rows.buffer]);
}
async function embedOnGpu(items, kind, label, info) {
  const t0 = performance.now(), step = kind === 'text' ? 2048 : 128; let out = null, dim = 0;
  for (let lo = 0; lo < items.length && running; lo += step) {
    const part = items.slice(lo, lo + step), { v, dim: d } = await gpuEmbed(kind === 'text' ? { kind, texts: part } : { kind, images: part });
    if (!out) { dim = d; out = new Float32Array(items.length * dim); } out.set(v, lo * dim);
    peek(out, dim, Array.from({ length: part.length }, (_, k) => lo + k), items, kind, items.length);
    note(`Embedding ${label}: ${Math.min(items.length, lo + step).toLocaleString()} of ${items.length.toLocaleString()} on this machine's GPU (${info.device})`);
  }
  stats.embed_ms = performance.now() - t0;
  return { vecs: out, dim, model: kind === 'text' ? TM.name : IM.name, hf: kind === 'text' ? TM.name : IM.name, device: `this machine's GPU (torch ${info.device})`, loadMs: 0 };
}
let tf = null;
async function encoder(task, id) {
  tf = tf || await import(TF); tf.env.allowLocalModels = false;
  for (const device of ['webgpu', 'wasm']) {
    try { const t0 = performance.now(); const ex = await tf.pipeline(task, id, { device, dtype: 'fp32', progress_callback: (p) => { if (p.status === 'progress' && p.file?.endsWith('.onnx')) postMessage({ type: 'progress', pct: p.progress }); } }); return { ex, device, loadMs: performance.now() - t0 }; }
    catch (e) { log(`${device} backend failed for ${id}: ${e.message || e}`); }
  }
  throw new Error(`no backend could load ${id}`);
}
async function embedTexts(texts, label) {
  const { ex, device, loadMs } = await encoder('feature-extraction', TM.id), dim = 384, out = new Float32Array(texts.length * dim), t0 = performance.now();
  // the live feeds' recipe: at most 128 tokens, chunks of 32; a non-finite row (the WebGPU backend returns one now and
  // then) is embedded again alone, then on the wasm backend
  const run = async (e, xs) => { const t = await e(xs, { pooling: TM.pool, normalize: true, truncation: true, max_length: 128 }); const v = Float32Array.from(t.data); t.dispose?.(); return v; };
  const finite = (v, j) => { for (let k = 0; k < dim; k++) if (!Number.isFinite(v[j * dim + k])) return false; return true; };
  // batches are padded to their longest text, so texts go through in length order and come back to their rows
  let fallback = null, pend = []; const byLen = texts.map((t, i) => i).sort((a, b) => texts[a].length - texts[b].length), tmp = new Float32Array(32 * dim);
  for (let lo = 0; lo < texts.length && running; lo += 32) {
    const ids = byLen.slice(lo, lo + 32), chunk = ids.map((i) => texts[i]), c = await run(ex, chunk); tmp.set(c);
    for (let j = 0; j < chunk.length; j++) {
      let one = tmp.subarray(j * dim, (j + 1) * dim);
      if (!finite(one, 0)) {
        one = await run(ex, [chunk[j]]);
        if (!finite(one, 0)) { fallback = fallback || await tf.pipeline('feature-extraction', TM.id, { device: 'wasm', dtype: 'fp32' }); one = await run(fallback, [chunk[j]]); stats.nan_repaired_wasm = (stats.nan_repaired_wasm || 0) + 1; }
        else stats.nan_repaired = (stats.nan_repaired || 0) + 1;
      }
      out.set(one, ids[j] * dim);
    }
    pend.push(...ids); if (pend.length >= 512) { peek(out, dim, pend, texts, 'text', texts.length); pend = []; }
    if (lo % 512 === 0) note(`Embedding ${label}: ${Math.min(texts.length, lo + 32).toLocaleString()} of ${texts.length.toLocaleString()} on ${device}`);
  }
  stats.embed_ms = performance.now() - t0; return { vecs: out, dim, model: TM.name, hf: TM.id, device, loadMs };
}
async function embedImages(blobs, label, b64s = null) {
  const { ex, device, loadMs } = await encoder('image-feature-extraction', IM.id); let dim = 0, out = null; const t0 = performance.now();
  for (let i = 0; i < blobs.length && running; i += 16) {
    const imgs = await Promise.all(blobs.slice(i, i + 16).map((b) => tf.RawImage.fromBlob(b)));
    const t = await ex(imgs);   // DINOv2: [batch, tokens, dim], the class token is the image's vector; CLIP: [batch, dim], already one vector
    const [B, T, H] = t.dims.length === 2 ? [t.dims[0], 1, t.dims[1]] : t.dims; if (!out) { dim = H; out = new Float32Array(blobs.length * dim); }
    for (let b = 0; b < B; b++) { const v = t.data.subarray(b * T * H, b * T * H + H); let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; for (let k = 0; k < H; k++) out[(i + b) * dim + k] = v[k] / s; }
    if (b64s) peek(out, dim, Array.from({ length: B }, (_, k) => i + k), b64s, 'image', blobs.length);
    note(`Embedding ${label}: ${Math.min(blobs.length, i + 16).toLocaleString()} of ${blobs.length.toLocaleString()} images on ${device}`);
  }
  stats.embed_ms = performance.now() - t0; return { vecs: out, dim, model: IM.name, hf: IM.id, device, loadMs };
}
function tableVectors(rows, D) {   // standardised numbers; log1p for long-tailed positive columns; one-hot categories (top 24) at weight 0.7
  const cols = [];
  for (const c of D.numeric || []) {
    const raw = rows.map((r) => { const v = r[c]; return v === null || v === '' || v === undefined ? NaN : Number(v); }), ok = raw.filter(Number.isFinite).sort((a, b) => a - b);
    if (!ok.length) continue; const med = ok[ok.length >> 1], logIt = ok[0] >= 0 && ok[ok.length - 1] > 50 * Math.max(med, 1e-9);
    const v = raw.map((x) => (Number.isFinite(x) ? (logIt ? Math.log1p(x) : x) : NaN)), fin = v.filter(Number.isFinite);
    const mean = fin.reduce((a, b) => a + b, 0) / fin.length, sd = Math.sqrt(fin.reduce((a, b) => a + (b - mean) ** 2, 0) / fin.length) || 1;
    cols.push(v.map((x) => (Number.isFinite(x) ? (x - mean) / sd : 0)));
  }
  for (const c of D.categorical || []) {
    const counts = new Map(); for (const r of rows) { const v = r[c]; if (v !== null && v !== undefined && v !== '' && v !== '?') counts.set(v, (counts.get(v) || 0) + 1); }
    for (const [val] of [...counts].sort((a, b) => b[1] - a[1]).slice(0, 24)) cols.push(rows.map((r) => (r[c] === val ? 0.7 : 0)));
  }
  const dim = cols.length, out = new Float32Array(rows.length * dim);
  for (let i = 0; i < rows.length; i++) for (let k = 0; k < dim; k++) out[i * dim + k] = cols[k][i];
  return { vecs: out, dim, model: `table-${dim}`, hf: 'standardised columns', device: 'cpu', loadMs: 0 };
}
// naming words for a table row (the page names topic clusters from them): each category value, and for a numeric column a
// word for the bottom or top third of the sample (datasets.js `levels`, else "low <column>" / "high <column>"); null
// separates phrases so no two-word name crosses columns
function tableTerms(rows, D) {
  const cut = {}, split = (x) => String(x).toLowerCase().match(/[a-z][a-z']+/g) || [];
  for (const c of D.numeric || []) { const v = rows.map((r) => Number(r[c])).filter(Number.isFinite).sort((a, b) => a - b); if (v.length) cut[c] = [v[Math.floor(v.length / 3)], v[Math.floor(2 * v.length / 3)]]; }
  return rows.map((r) => {
    const t = [];
    for (const c of D.categorical || []) { const v = r[c]; if (v && v !== '?') t.push(...split(v), null); }
    for (const c of D.numeric || []) {
      const x = Number(r[c]); if (!cut[c] || r[c] === null || r[c] === '' || !Number.isFinite(x) || (x > cut[c][0] && x < cut[c][1])) continue;
      const lv = D.levels?.[c]; if (D.levels && !lv) continue;   /* with `levels`, only the columns it names get words */
      t.push(...split(x <= cut[c][0] ? (lv ? lv[0] : `low ${c}`) : (lv ? lv[1] : `high ${c}`)), null);
    }
    return t;
  });
}

// ---- the cache of embedded windows (IndexedDB, this browser only). A whole table is hundreds of MB, which Chrome refuses
// as one value (UnknownError), so it is kept in slices of 16,384 rows (rows and vectors), with a small index entry written
// last: a save cut short leaves no index and is never read. Entries of other cache versions are deleted.
const SLICE = 16384, CACHE_V = 'v6';
function db() { return new Promise((res, rej) => { const q = indexedDB.open(CACHE_DB, 1); q.onupgradeneeded = () => q.result.createObjectStore('w'); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); }); }
const idbGet = (d, k) => new Promise((res) => { const q = d.transaction('w').objectStore('w').get(k); q.onsuccess = () => res(q.result ?? null); q.onerror = () => res(null); });
const idbPut = (d, k, v) => new Promise((res, rej) => { const t = d.transaction('w', 'readwrite'); t.objectStore('w').put(v, k); t.oncomplete = res; t.onerror = t.onabort = () => rej(t.error || new Error('aborted')); });
async function cacheGet(key) {
  try {
    const d = await db(), meta = await idbGet(d, key + '|index'); if (!meta) return null;
    const posts = [], vecs = new Float32Array(meta.n * meta.dim);
    for (let i = 0; i < meta.slices; i++) { const [p, v] = await Promise.all([idbGet(d, `${key}|p${i}`), idbGet(d, `${key}|v${i}`)]); if (!p || !v) return null; posts.push(...p); vecs.set(v, i * SLICE * meta.dim); }
    return { ...meta.w, posts, vecs };
  } catch { return null; }
}
async function cachePut(key, w) {
  try {
    const d = await db(), n = w.posts.length, slices = Math.ceil(n / SLICE);
    for (let i = 0; i < slices; i++) { await idbPut(d, `${key}|p${i}`, w.posts.slice(i * SLICE, (i + 1) * SLICE)); await idbPut(d, `${key}|v${i}`, w.vecs.slice(i * SLICE * w.dim, Math.min(n, (i + 1) * SLICE) * w.dim)); }
    const { posts, vecs, ...rest } = w; await idbPut(d, key + '|index', { n, dim: w.dim, slices, w: rest });
    log(`cached ${n} rows in ${slices} slices`);
  } catch (e) { log(`not cached (${e.name || e}: ${e.message || ''})`); }
}
async function cachePrune() {   // entries written by other versions of this cache
  try { const d = await db(), keys = await new Promise((res) => { const q = d.transaction('w').objectStore('w').getAllKeys(); q.onsuccess = () => res(q.result); q.onerror = () => res([]); });
    const old = keys.filter((k) => !String(k).includes(`|${CACHE_V}|`)); if (old.length) await new Promise((res) => { const t = d.transaction('w', 'readwrite'), st = t.objectStore('w'); for (const k of old) st.delete(k); t.oncomplete = t.onerror = t.onabort = res; });
  } catch { /* nothing to prune */ }
}

async function load(key, cap) {
  const D = DATASETS[key]; if (!D) throw new Error(`no dataset called ${key}`);
  const label = D.label;
  if (D.kind === 'lists') {   // arXiv (arxiv/build_arxiv.py): rows, 64-number vectors and 30-neighbour lists, all in parts
    const f = await findStatic(key, 'info.json'); if (!f) throw new Error(`${label} is not available here (no static-data/${key}/info.json)`);
    const info = f.json, base = info.base || f.base, n = info.n; note(`Reading ${label}: ${n.toLocaleString()} papers…`);
    // every file counts, the titles (rows_*.json, a quarter of the bytes) too: the count once stopped at 944 MB, all the binary
    // parts, while the titles were still arriving, which read as a stall. MB here are millions of bytes, as the files are listed.
    // Bytes count as they arrive. A file with no new bytes for 20 s is cancelled and fetched again past the browser's cache
    // (up to 3 tries; then the load fails with its name): in the investigator's Chrome the 39 title files once never arrived
    // (more than 5 min at "991 of 1318 MB, 120 of 159 files") while the same files took 4 s in a clean worker here.
    const files = info.rowParts.length + info.vectorParts.length + info.knnIdxParts.length + info.knnDistParts.length, total = info.bytes || 0; let got = 0, done = 0, retried = 0;
    const show = () => note(`Reading ${label}: ${(got / 1e6).toFixed(0)}${total ? ` of ${(total / 1e6).toFixed(0)}` : ''} MB, ${done} of ${files} files${retried ? ` (${retried} fetched again after stalling)` : ''}`);
    const steady = async (url, cache, tries = 3) => {
      const ac = new AbortController(); let last = Date.now(), mine = 0; const dog = setInterval(() => { if (Date.now() - last > 20000) ac.abort(); }, 2000);
      try {
        const r = await fetch(url, { cache, signal: ac.signal }); if (!r.ok) throw new Error(`${url.split('/').pop()}: ${r.status}`);
        const rd = r.body.getReader(), chunks = [];
        for (;;) { const { done: end, value } = await rd.read(); if (end) break; chunks.push(value); mine += value.byteLength; got += value.byteLength; last = Date.now(); show(); }
        const out = new Uint8Array(mine); let o = 0; for (const c of chunks) { out.set(c, o); o += c.byteLength; } return out.buffer;
      } catch (e) {
        got -= mine; if (!ac.signal.aborted || tries <= 1) throw ac.signal.aborted ? new Error(`${url.split('/').pop()} stalled three times`) : e;
        retried++; log(`${url.split('/').pop()}: no bytes for 20 s, fetching again`); return steady(url, 'no-store', tries - 1);
      } finally { clearInterval(dog); }
    };
    const part = (p) => steady(base + p, 'default').then((b) => { done++; show(); return b; });
    const cat = async (parts) => { const bufs = await Promise.all(parts.map(part)), out = new Uint8Array(bufs.reduce((a, b) => a + b.byteLength, 0)); let o = 0; for (const b of bufs) { out.set(new Uint8Array(b), o); o += b.byteLength; } return out.buffer; };
    const dec = new TextDecoder(), rowsOf = (p) => part(p).then((b) => JSON.parse(dec.decode(b)));
    const [rowsArr, vbuf, ibuf, dbuf] = await Promise.all([Promise.all(info.rowParts.map(rowsOf)).then((a) => a.flat()), cat(info.vectorParts), cat(info.knnIdxParts), cat(info.knnDistParts)]);
    note(`Reading ${label}: all ${files} files in (${(got / 1e6).toFixed(0)} MB); setting out ${n.toLocaleString()} papers…`);
    const field = (c) => String(info.categories[c] || '').split('.')[0];   // the archive (cs, math, hep-th, ...) colours the map
    return { posts: rowsArr.map((r) => ({ text: r[0], kind: `${info.categories[r[1]]} · ${r[2]}`, url: `https://arxiv.org/abs/${r[3]}`, cat: field(r[1]) })), dates: rowsArr.map(D.date),
      vecs: fromHalves(new Uint16Array(vbuf)), dim: info.dim, lists: { idx: new Int32Array(ibuf), dist: fromHalves(new Uint16Array(dbuf)), k: info.k, n }, model: info.model || 'sentence-transformers/all-MiniLM-L6-v2 (64 principal components)', hf: 'precomputed', device: 'precomputed', loadMs: 0, total: n };
  }
  if (D.kind === 'vectors') {   // packed locally (ASRS): info.json, rows.json, vectors as float32
    const f = await findStatic(key, 'info.json'); if (!f) throw new Error(`${label} is not packed here: run python3 browser_jetstream/pack_asrs.py`);
    const info = f.json, base = info.base || f.base; note(`Reading ${label}…`);   // base: the parts' own host (Hugging Face on the site)
    const [rows, vbuf] = await Promise.all([info.rowParts ? Promise.all(info.rowParts.map((p) => fetch(base + p).then((r) => r.json()))).then((a) => a.flat()) : fetch(base + 'rows.json').then((r) => r.json()), info.vectorParts ? fetchParts(base, info.vectorParts) : fetchBuf(base + info.vectors)]);
    const all = new Float32Array(vbuf), dim = info.dim, take = pick(rows.length, cap, rows.map(D.date)), vecs = new Float32Array(take.length * dim);
    take.forEach((i, r) => vecs.set(all.subarray(i * dim, (i + 1) * dim), r * dim));
    return { posts: take.map((i) => ({ text: rows[i].text, kind: rows[i].show, cat: D.category(rows[i]) })), dates: take.map((i) => D.date(rows[i])), colors: info.colors, vecs, dim, model: info.model, hf: info.model, device: 'packed', loadMs: 0, total: rows.length };
  }
  let rows = await readTable(D, label); const total = rows.length;
  if (D.dedupe) { const seen = new Set(); rows = rows.filter((r) => { const k = D.dedupe(r); if (!k || seen.has(k)) return false; seen.add(k); return true; }); stats.duplicate = total - rows.length; }
  if (D.kind === 'text') rows = rows.filter((r) => (D.text(r) || '').trim());
  if (D.kind === 'upload-vectors') rows = rows.filter((r) => vectorOf(r[D.embedding]));
  if (D.date) rows = rows.filter((r) => D.date(r) != null);   // a dated table: rows without a date have no place on the axis
  const take = pick(rows.length, cap, D.date ? rows.map(D.date) : null), sel = take.map((i) => rows[i]);
  const show = (r) => (D.show ? String(D.show(r)) : ''), cat = (r) => (D.category ? String(D.category(r) ?? '') || '(none)' : undefined);
  const extra = { total, dates: D.date ? sel.map(D.date) : null }, server = D.kind === 'table' || !useGpu ? null : await gpu();
  if (D.kind === 'text') {
    const texts = sel.map((r) => String(D.text(r)).replace(/\s+/g, ' ').trim()), rowsId = fnv(texts);
    const v = (!exporting && await precomputed(key, rowsId)) || (server ? await embedOnGpu(texts, 'text', label, server) : await embedTexts(texts, label));
    return { ...v, ...extra, rowsId, posts: sel.map((r, i) => ({ text: texts[i], kind: show(r), url: D.link ? D.link(r) : undefined, cat: cat(r) })) };
  }
  if (D.kind === 'upload-vectors') {   // your file's own vectors: used as they are (L2-normalised), nothing embedded
    const first = vectorOf(sel[0][D.embedding]), dim = first.length, vecs = new Float32Array(sel.length * dim); let bad = 0;
    sel.forEach((r, i) => { const v = vectorOf(r[D.embedding]); if (!v || v.length !== dim) { bad++; return; } let q = 0; for (const x of v) q += x * x; q = Math.sqrt(q) || 1; for (let k = 0; k < dim; k++) vecs[i * dim + k] = v[k] / q; });
    if (bad) log(`${bad} rows had no vector of ${dim} numbers: placed at the origin of the vector space`);
    return { ...extra, vecs, dim, model: `${D.embedding} (your file's vectors)`, hf: 'your file', device: 'your file', loadMs: 0, posts: sel.map((r) => ({ text: D.text(r) || show(r), kind: show(r), cat: cat(r) })) };
  }
  if (D.kind === 'image') {
    const b64 = (bytes) => { let s = ''; for (let i = 0; i < bytes.length; i += 32768) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 32768)); return btoa(s); }, enc = sel.map((r) => b64(r.image.bytes));
    const rowsId = fnv(sel.map((r, i) => `${show(r)}|${r.image.bytes.length}|${enc[i].slice(0, 64)}`));
    const v = (!exporting && await precomputed(key, rowsId, 'image')) || (server ? await embedOnGpu(enc, 'image', label, server) : await embedImages(sel.map((r) => new Blob([r.image.bytes], { type: 'image/jpeg' })), label, enc));
    return { ...v, ...extra, rowsId, posts: sel.map((r, i) => ({ text: show(r), kind: show(r), img: 'data:image/jpeg;base64,' + enc[i], cat: cat(r) })) };
  }
  const v = tableVectors(sel, D), terms = tableTerms(sel, D);
  return { ...v, ...extra, posts: sel.map((r, i) => ({ text: [show(r), terms[i].filter(Boolean).join(' ')].filter(Boolean).join(' · '), kind: show(r), terms: terms[i], cat: cat(r) })) };
}

onmessage = async (ev) => {
  const m = ev.data;
  if (m.type === 'start') {
    running = true;
    try {
      useGpu = m.gpu !== false; exporting = !!m.export;
      if (m.model && TEXT_MODELS[m.model]) { TM_KEY = m.model; TM = TEXT_MODELS[m.model]; }
      if (m.model && IMAGE_MODELS[m.model]) { IM_KEY = m.model; IM = IMAGE_MODELS[m.model]; }
      if (m.upload) DATASETS.upload = { ...uploadDataset(m.upload), embedding: m.upload.embedding };   // your own file (upload.js)
      const key = m.relay, D = DATASETS[key], cap = m.rows || 0, ck = `${key}|${cap || 'all'}|${CACHE_V}${vecTag(D.kind) ? '|' + vecTag(D.kind) : ''}${D.upload ? '|' + m.upload.id : ''}`, t0 = performance.now(); cachePrune();
      const worth = D.kind === 'text' || D.kind === 'image';   /* packed vectors and tables load in seconds: not cached */
      let w = worth && !exporting ? await cacheGet(ck) : null; const cached = !!w;
      if (!w) { w = await load(key, cap); if (!running) return; }
      if (exporting && worth && w.rowsId) await exportVectors(key, w);
      const n = w.posts.length, dim = w.dim, span = m.liveStartUs - m.cursorUs;
      if (!peeked) await peek(w.vecs, dim, Array.from({ length: n }, (_, i) => i), null, 'none', n);   // nothing was embedded here (packed, a table, or cached): the whole cloud at once, so the map still glides out of it
      let bad = 0; for (let i = 0; i < w.vecs.length; i++) if (!Number.isFinite(w.vecs[i])) { w.vecs[i] = 0; bad++; } if (bad) log(`${bad} non-finite numbers set to 0`);
      // stream times: dated rows on a linear map of [first date's unit start, end of the last date's unit) onto the window,
      // rows sharing a date spread evenly up to the next date; undated rows evenly spaced. The base map is the first half
      // of the rows (the stream starts at the middle row's time) only on request: by default every row is the base map (one
      // fit of the whole table, nothing streamed); untilMs (a rewind on the timeline) makes the base the rows dated up to
      // then, and the rest the stream that Play places.
      const T = new Float64Array(n); let timeAxis = null;
      if (w.dates) {
        const unitStart = (ms) => { const d = new Date(ms); return D.time === 'year' ? Date.UTC(d.getUTCFullYear(), 0) : D.time === 'month' ? Date.UTC(d.getUTCFullYear(), d.getUTCMonth()) : Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()); };
        const unitNext = (ms) => { const d = new Date(unitStart(ms)); return D.time === 'year' ? Date.UTC(d.getUTCFullYear() + 1, 0) : D.time === 'month' ? Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1) : ms + 86400000; };
        const d0 = unitStart(w.dates[0]), d1 = unitNext(w.dates[n - 1]), toT = (ms) => (ms - d0) / (d1 - d0) * span;
        for (let i = 0; i < n;) { let j = i; while (j < n && w.dates[j] === w.dates[i]) j++; const a = toT(w.dates[i]), b = toT(j < n ? w.dates[j] : unitNext(w.dates[i])); for (let k = i; k < j; k++) T[k] = a + (k - i + 0.5) / (j - i) * (b - a); i = j; }
        timeAxis = { d0, d1, unit: D.time };
      } else for (let i = 0; i < n; i++) T[i] = (i + 0.5) / n * span;
      let baseEndUs = m.liveStartUs + 1;   // past every row: all of them are the base map
      if (m.untilMs && w.dates) { let k = 0; while (k < n && w.dates[k] <= m.untilMs) k++; if (k > 0 && k < n) baseEndUs = Math.round(m.cursorUs + T[k]); }
      // the model lesson: how many of each item's 15 nearest neighbours the other models agree on with the default model
      // (MiniLM for text, DINOv2 for images; model_overlap.py)
      let agreement = null; const models = modelsFor(D.kind);
      if (models && !D.upload) { agreement = {}; for (const k of Object.keys(models)) { if (k === defaultModel(D.kind)) continue; const f = await findStatic(key, `vectors.${k}.json`); const v = f?.json?.sharedWithDefault ?? f?.json?.sharedWithMinilm; if (v != null) agreement[k] = v; } }
      postMessage({ type: 'ready', agreement, rowsId: w.rowsId || null, modelKey: modelKey(D.kind), model: w.model, hf: w.hf, device: cached ? 'this browser\'s cache' : w.device, dim, loadMs: performance.now() - t0, cursorUs: m.cursorUs, liveStartUs: m.liveStartUs, stream: key, rows: n, total: w.total, baseEndUs, timeAxis, colors: w.colors || null, categoryName: D.categoryName || null });
      if (w.lists) postMessage({ type: 'lists', ...w.lists }, [w.lists.idx.buffer, w.lists.dist.buffer]);   // shipped neighbour lists: the page searches nothing
      const posts = w.posts.map((p, i) => ({ uri: `${key}/${i}`, time_us: Math.round(m.cursorUs + T[i]), did: '', rkey: '', text: p.text || '', tags: [], reply: 0, embed: 0, kind: p.kind, url: p.url, img: p.img, terms: p.terms, cat: p.cat }));
      stats.events = stats.post_creates = w.total || n; stats.kept = stats.embedded = n;
      for (let lo = 0; lo < n && running; lo += FRAME) {
        const hi = Math.min(n, lo + FRAME), v = w.vecs.slice(lo * dim, hi * dim); stats.batches++;
        postMessage({ type: 'frame', hdr: { n: hi - lo, dim, model: w.model, encoder: w.model, source: 'static', stream: key, device: 'static', embed_ms: 0, posts: posts.slice(lo, hi), cursorUs: m.cursorUs, liveStartUs: m.liveStartUs, pendingAfter: n - hi, backfilling: true }, vecs: v }, [v.buffer]);
        await new Promise((r) => setTimeout(r, 0));
      }
      postMessage({ type: 'backfilled', liveStartUs: m.liveStartUs, cursorUs: m.cursorUs });
      if (!cached && worth) cachePut(ck, w);   /* after the rows are on their way: a slow or refused save never holds up the map */
      postMessage({ type: 'stats', stats: { ...stats, pending: 0 } });
      log(`${key}: ${n} rows (${cached ? 'cached' : 'fetched and embedded'}) in ${Math.round(performance.now() - t0)} ms`);
    } catch (e) { postMessage({ type: 'error', msg: `${DATASETS[m.relay]?.label || m.relay} did not load (${e.message || e})` }); }
  } else if (m.type === 'stop') { running = false; postMessage({ type: 'stopped' }); }
};
