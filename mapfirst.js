// arXiv map first (2026-10-26): the finished map arrives before anything else, as dtour.dev shows its 3 million papers
// from 115 MB of precomputed positions. What the first screen needs, about 33 MB in all (arxiv/build_mapfirst.py):
// (every file ends in .bin, which the Hugging Face repository stores as large files and serves from its file network;
// a plain file comes from huggingface.co itself at about 0.3 MB/s in a browser; the manifest names them)
//   map.f32          every paper's finished 2D position (float32 x, y; the page's own fit of the full table, exported once)
//   cat.u8, year.u8  its primary category (index into info.categories) and first-version year (offset from info.years[0])
//   regions.json     the regions' names at every grain the page offers, named from every title when the map was exported
// and, fetched only for a paper someone clicks, by HTTP byte range:
//   titles.txt       "arXiv id<TAB>title" per paper, one line each, in row order; titles_index.u32 gives the byte offset
//                    of every 1,024th line (plus the end), so one paper's title is one request of about 90 kB
//   knn_idx_*.i32    the shipped 30-neighbour lists (build_arxiv.py), read 120 bytes at a time
// The vectors (412 MB) and the whole neighbour lists (579 MB) load only when a control needs them: the page reloads with
// ?full=1, which reads everything and starts from this same map (no new fit).
const BASES = ['./static-data/', '../data/jetstream_demo/static/'];

export async function findInfo(key) {
  for (const b of BASES) { try { const r = await fetch(`${b}${key}/info.json`, { cache: 'no-store' }); if (r.ok) return { base: `${b}${key}/`, json: await r.json() }; } catch { /* next */ } }
  return null;
}
// a whole file, its bytes counted as they arrive
async function counted(url, onBytes) {
  const r = await fetch(url); if (!r.ok) throw new Error(`${url.split('/').pop()}: ${r.status}`);
  if (!r.body) { const b = await r.arrayBuffer(); onBytes(b.byteLength); return b; }
  const reader = r.body.getReader(), chunks = []; let n = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); n += value.byteLength; onBytes(value.byteLength); }
  const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.byteLength; } return out.buffer;
}
async function range(url, a, b) {   // bytes [a, b)
  const r = await fetch(url, { headers: { Range: `bytes=${a}-${b - 1}` } }); if (!r.ok) throw new Error(`${url.split('/').pop()}: ${r.status}`);
  const buf = await r.arrayBuffer(); return r.status === 206 ? buf : buf.slice(a, b);   // a host that ignores Range sends the whole file
}

// One paper's title (and arXiv id), a block of 1,024 lines per request; the last 256 blocks read are kept.
class Titles {
  constructor(url, index, block) { this.url = url; this.index = index; this.block = block; this.cache = new Map(); this.wait = new Map(); }
  get(p) { const L = this.cache.get(Math.floor(p / this.block)); return L ? L[p % this.block] : null; }
  async load(p) {
    const b = Math.floor(p / this.block); if (this.cache.has(b)) return this.get(p);
    if (!this.wait.has(b)) this.wait.set(b, range(this.url, this.index[b], this.index[b + 1]).then((buf) => {
      const lines = new TextDecoder().decode(buf).split('\n').map((l) => { const t = l.indexOf('\t'); return { id: l.slice(0, t), title: l.slice(t + 1) }; });
      this.cache.set(b, lines); if (this.cache.size > 256) this.cache.delete(this.cache.keys().next().value); this.wait.delete(b);
    }).catch((e) => { this.wait.delete(b); throw e; }));
    await this.wait.get(b); return this.get(p);
  }
}
// One paper's shipped neighbour list (k = 30, nearest first) by byte range on the list part that holds it.
class Lists {
  constructor(base, info) { this.base = base; this.parts = info.knnIdxParts; this.per = info.rowsPerPart.knnIdx; this.k = info.k; }
  async of(p) { const part = Math.floor(p / this.per), o = (p % this.per) * this.k * 4; return new Int32Array(await range(this.base + this.parts[part], o, o + this.k * 4)); }
}

// The first screen's files, or null when this host has none (the page then loads everything, as before).
export async function loadMapFirst(key, note) {
  const f = await findInfo(key), M = f?.json?.mapFirst; if (!M) return null;
  const info = f.json, base = M.base || info.base || f.base, n = info.n, total = M.bytes || 0; let got = 0;
  const tick = (d) => { got += d; note(`Reading the finished map of ${n.toLocaleString()} papers: ${(got / 1e6).toFixed(0)}${total ? ` of ${(total / 1e6).toFixed(0)}` : ''} MB`); };
  const [xy, cat, year, idx, regions] = await Promise.all([counted(base + M.map, tick), counted(base + M.cat, tick), counted(base + M.year, tick), counted(base + M.titleIndex, tick), counted(base + M.regions, tick)]);
  const out = { info, n, xy: new Float32Array(xy), cat: new Uint8Array(cat), year: new Uint8Array(year), regions: JSON.parse(new TextDecoder().decode(regions)), rowsId: M.rowsId || info.rowsId,
    titles: new Titles(base + M.titles, new Uint32Array(idx), M.titleBlock), lists: new Lists(info.base || f.base, info), bytes: got };
  if (out.xy.length !== n * 2 || out.cat.length !== n || out.year.length !== n) throw new Error(`the finished map of ${key} does not match its ${n.toLocaleString()} rows`);
  return out;
}

// The shipped name of each region just computed: the shipped region at this grain with the same rank and size (the same
// map and code give the same regions), else the nearest shipped centre that is still free. Regions computed with
// settings that were never shipped keep "region k" only when no shipped region is near.
export function nameRegions(R, shipped, sigma) {
  const keys = Object.keys(shipped.bySigma).map(Number), s = keys.includes(sigma) ? sigma : keys.reduce((a, b) => (Math.abs(b - sigma) < Math.abs(a - sigma) ? b : a), keys[0]);
  const S = shipped.bySigma[s] || [], used = new Set(); let exact = 0;
  for (const r of R.regions) {
    let e = S[r.k] && S[r.k].countBase === r.countBase && !used.has(r.k) ? r.k : -1; if (e >= 0) exact++;
    if (e < 0) { let bd = Infinity; S.forEach((x, j) => { if (used.has(j)) return; const d = (x.cx - r.cx) ** 2 + (x.cy - r.cy) ** 2; if (d < bd) { bd = d; e = j; } }); }
    if (e < 0) continue; used.add(e); const x = S[e]; Object.assign(r, { name: x.name, terms: x.terms, termInfo: x.termInfo || [], rep: x.rep ?? -1, repText: x.repText });
  }
  return { sigma: s, exact, of: R.regions.length };
}
