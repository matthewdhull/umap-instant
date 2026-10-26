// Finished maps of a static dataset, one per model (the model picker and the model compare): each map is its rows' 2D
// positions, 8 bytes a row (float32 x, y), in the dataset's row order. A map is used only for the rows it was fitted on:
// its rowsId (the worker's fingerprint of the rows in their final order) and row count must match this load's.
//   on the site    static-data/<dataset>/map.<model>.json + map_<model>.f32, exported from this machine's page with
//                  ?source=<dataset>&model=<model>&exportmap=1 (a fresh fit, then serve.py /save), shipped by build_site.py
//   this browser   IndexedDB, written after every fresh fit of a whole table (no ?until=, no tuned base), so a model with no
//                  map on the site is fitted once and is instant after that
// Nothing here is ever sent anywhere but the local page server's /save on export.
const BASES = ['./static-data/', '../data/jetstream_demo/static/'], DB = 'jetstream-maps', STORE = 'maps';

const keyOf = (source, model, rowsId) => `${source}|${model}|${rowsId}`;
function db() { return new Promise((res) => { if (!('indexedDB' in self)) { res(null); return; } const q = indexedDB.open(DB, 1); q.onupgradeneeded = () => q.result.createObjectStore(STORE); q.onsuccess = () => res(q.result); q.onerror = q.onblocked = () => res(null); }); }

// the site's map of this dataset under this model, or null (no file, or fitted on other rows)
async function siteMap(source, model, rowsId, n) {
  for (const b of BASES) {
    try {
      const r = await fetch(`${b}${source}/map.${model}.json`, { cache: 'no-store' }); if (!r.ok) continue;
      const m = await r.json(); if (m.rowsId !== rowsId || m.n !== n) return null;
      const bufs = await Promise.all(m.parts.map((p) => fetch(`${b}${source}/${p}`).then((x) => { if (!x.ok) throw new Error(`${p}: ${x.status}`); return x.arrayBuffer(); })));
      const xy = new Float32Array(n * 2); let o = 0; for (const buf of bufs) { xy.set(new Float32Array(buf), o); o += buf.byteLength / 4; }
      return o === n * 2 ? { xy, from: 'site', meta: m } : null;
    } catch { /* next base */ }
  }
  return null;
}
async function browserMap(source, model, rowsId, n) {
  const d = await db(); if (!d) return null;
  return new Promise((res) => { try { const q = d.transaction(STORE).objectStore(STORE).get(keyOf(source, model, rowsId)); q.onsuccess = () => { const v = q.result; res(v && v.n === n ? { xy: new Float32Array(v.xy), from: 'browser', meta: v.meta } : null); }; q.onerror = () => res(null); } catch { res(null); } });
}
// the finished map of this dataset under this model: the site's, else this browser's, else null
export async function findMap(source, model, rowsId, n) {
  if (!rowsId) return null;
  return (await siteMap(source, model, rowsId, n)) || (await browserMap(source, model, rowsId, n));
}
// keep a map in this browser; resolves once it is written (a reload may follow at once)
export async function keepMap(source, model, rowsId, xy, meta = {}) {
  const d = await db(); if (!d || !rowsId) return;
  await new Promise((res) => { try { const t = d.transaction(STORE, 'readwrite'); t.objectStore(STORE).put({ n: xy.length / 2, xy: xy.slice().buffer, meta: { ...meta, at: Date.now() } }, keyOf(source, model, rowsId)); t.oncomplete = res; t.onerror = t.onabort = () => { console.warn('[maps] not kept', t.error); res(); }; } catch (e) { console.warn('[maps] not kept', e); res(); } });
}
// the map that was on screen when the page reloaded itself for a model's own vectors (main.js reloadWith): this browser only
export const keepOnScreen = (source, model, rowsId, xy) => keepMap(source, `${model}.onscreen`, rowsId, xy);
export const onScreenMap = (source, model, rowsId, n) => browserMap(source, `${model}.onscreen`, rowsId, n);
// ?exportmap=1 on this machine's page: the map just fitted goes to serve.py's /save for the site (one part: 1.5 MB at 182,000 rows)
export async function exportMap(source, model, rowsId, xy, meta = {}) {
  const part = `map_${model}.f32`, n = xy.length / 2;
  await fetch(`/save?path=${source}/${part}`, { method: 'POST', body: xy.slice().buffer });
  await fetch(`/save?path=${source}/map.${model}.json`, { method: 'POST', body: JSON.stringify({ n, rowsId, model, parts: [part], ...meta }) });
  console.info(`[maps] exported ${source} ${model}: ${n} rows`);
}
