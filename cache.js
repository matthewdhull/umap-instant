// Browser-side cache of embedded posts (IndexedDB), so a reload within the hour reuses what was already fetched and
// embedded and only the minutes since the last visit are fetched again. One record per post, keyed by [time_us, uri]
// so a time range is one bulk read and expiring old records is one range delete (a first version keyed by uri alone
// pruned record by record through a cursor, which took 10 to 15 s with nothing on screen). The 384-dim vector is
// stored as bytes. Nothing here is ever sent anywhere.
const DB = 'jetstream-live-map', STORE = 'posts_by_time', VERSION = 2;
// One database per stream source (the Bluesky one keeps the original name), so an hour of edits never mixes with an hour of posts.
export function openCache(source = 'jetstream') {
  return new Promise((resolve) => {
    if (!('indexedDB' in window)) { resolve(null); return; }
    const req = indexedDB.open(source === 'jetstream' ? DB : `${DB}-${source}`, VERSION);
    req.onupgradeneeded = () => { const db = req.result; for (const name of Array.from(db.objectStoreNames)) db.deleteObjectStore(name); db.createObjectStore(STORE, { keyPath: 'key' }); };
    req.onsuccess = () => resolve(req.result); req.onerror = () => resolve(null); req.onblocked = () => resolve(null);
  });
}
const range = (fromUs, toUs) => IDBKeyRange.bound([fromUs], [toUs + 1]);   // array keys compare element-wise: [toUs + 1] is past every [toUs, uri]
// Records with time_us in [fromUs, toUs], in time order, in one bulk read.
export function readRange(db, fromUs, toUs = Number.MAX_SAFE_INTEGER) {
  return new Promise((resolve) => {
    if (!db) { resolve([]); return; }
    try { const r = db.transaction(STORE, 'readonly').objectStore(STORE).getAll(range(fromUs, toUs)); r.onsuccess = () => resolve(r.result || []); r.onerror = () => resolve([]); }
    catch (e) { console.warn('[cache] read failed', e); resolve([]); }
  });
}
// Store one embedded frame (fire and forget).
export function writeFrame(db, hdr, vecs) {
  if (!db) return;
  try {
    const tx = db.transaction(STORE, 'readwrite'), st = tx.objectStore(STORE), dim = hdr.dim;
    hdr.posts.forEach((p, i) => { if (!p.uri) return; st.put({ key: [p.time_us, p.uri], uri: p.uri, time_us: p.time_us, did: p.did, rkey: p.rkey, created_at: p.created_at, text: p.text, tags: p.tags || [], reply: p.reply || 0, embed: p.embed || 0, url: p.url, kind: p.kind, score: p.score ?? null, descendants: p.descendants ?? null, vec: vecs.slice(i * dim, (i + 1) * dim).buffer }); });
    tx.onerror = (e) => console.warn('[cache] write failed', e.target?.error);
  } catch (e) { console.warn('[cache] write failed', e); }
}
// Delete everything older than beforeUs: one range delete.
export function prune(db, beforeUs) {
  return new Promise((resolve) => {
    if (!db) { resolve(); return; }
    try { const tx = db.transaction(STORE, 'readwrite'); tx.objectStore(STORE).delete(IDBKeyRange.upperBound([beforeUs])); tx.oncomplete = () => resolve(); tx.onerror = () => resolve(); }
    catch (e) { console.warn('[cache] prune failed', e); resolve(); }
  });
}
export function countAll(db) { return new Promise((resolve) => { if (!db) { resolve(0); return; } try { const r = db.transaction(STORE, 'readonly').objectStore(STORE).count(); r.onsuccess = () => resolve(r.result); r.onerror = () => resolve(0); } catch { resolve(0); } }); }
