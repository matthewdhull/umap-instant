// The base map passed between the live map and the base map tuner, in this browser only (IndexedDB, one record).
// The live map writes its base (rows, vectors, positions, topic clusters, time window) when you open the tuner, so the
// tuner shows that same map; the tuner writes the tuned positions back when you press Use in live map, and the live map
// opened with ?base=handoff uses them as its base instead of fitting one (its 15-neighbour lists are recomputed from the
// same vectors). Nothing here is sent anywhere.
const DB = 'jetstream-base-handoff', STORE = 'handoff', KEY = 'current';
function open() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) { reject(new Error('this browser has no IndexedDB')); return; }
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error || new Error('IndexedDB open failed'));
  });
}
// h: { source, recKey, t0us, baseEndUs, nBase, dim, model, posts, timesUs, feat, xy, labelS, labels, labelColors, fit, savedAt }
export async function saveHandoff(h) {
  const db = await open();
  await new Promise((resolve, reject) => { const tx = db.transaction(STORE, 'readwrite'); tx.objectStore(STORE).put({ ...h, savedAt: Date.now() }, KEY); tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); });
  db.close();
}
export async function loadHandoff() {
  const db = await open();
  const h = await new Promise((resolve, reject) => { const r = db.transaction(STORE, 'readonly').objectStore(STORE).get(KEY); r.onsuccess = () => resolve(r.result || null); r.onerror = () => reject(r.error); });
  db.close(); return h;
}

// A picture of the map for the page change itself: the leaving page saves its canvas (sessionStorage, this tab only), the
// next page shows it at the same place on screen from its first paint (the inline script in its <head>) and fades it out
// once its own map is drawn, so the map never disappears while the other page starts its GPU work.
export function saveSnapshot(canvas, toPage) {
  try {
    const r = canvas.getBoundingClientRect();
    sessionStorage.setItem('jet.snap', JSON.stringify({ url: canvas.toDataURL('image/webp', 0.92), rect: { left: r.left, top: r.top, width: r.width, height: r.height }, to: toPage, t: Date.now() }));
  } catch (e) { console.warn('[handoff] no snapshot', e); }
}
export function dropSnapshot() {
  const el = document.getElementById('snapOverlay'); if (!el) return;
  requestAnimationFrame(() => requestAnimationFrame(() => { window.__snapDroppedAt = performance.now(); el.style.opacity = '0'; setTimeout(() => el.remove(), 350); }));
}
