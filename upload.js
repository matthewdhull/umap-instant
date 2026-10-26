// Your own file as a dataset (?source=upload): a CSV, TSV, JSON, JSONL or Parquet file chosen in the data menu, read in this
// browser and never sent anywhere. The file's bytes wait in this browser's IndexedDB and the chosen columns in
// sessionStorage, so the page reloads onto it like any other dataset (static_worker.js reads the bytes back).
//   text column       what is embedded (MiniLM or the picked model), unless
//   embedding column  numbers per row already (a JSON or Parquet list, or "[0.1, 0.2, ...]" in a CSV cell): used as they are,
//                     L2-normalised, nothing embedded
//   label column      the topic colours and legend (as a dataset's category)
//   date column       a time axis: rows in date order, rewind and Play as with the other dated tables
const HYP = 'https://cdn.jsdelivr.net/npm/hyparquet@1.31.2/+esm', HYPC = 'https://cdn.jsdelivr.net/npm/hyparquet-compressors@1.1.1/+esm';
const DB = 'umap-instant-upload', STORE = 'files';

function idb() { return new Promise((res, rej) => { const r = indexedDB.open(DB, 1); r.onupgradeneeded = () => r.result.createObjectStore(STORE); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); }); }
export async function putFile(bytes) { const db = await idb(); await new Promise((res, rej) => { const t = db.transaction(STORE, 'readwrite'); t.objectStore(STORE).put(bytes, 'current'); t.oncomplete = res; t.onerror = () => rej(t.error); }); db.close(); }
export async function getFile() { const db = await idb(); const v = await new Promise((res, rej) => { const q = db.transaction(STORE).objectStore(STORE).get('current'); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); }); db.close(); if (!v) throw new Error('the uploaded file is no longer in this browser: open it again'); return v; }

export const formatOf = (name) => { const e = name.toLowerCase().split('.').pop(); return e === 'parquet' || e === 'pq' ? 'parquet' : e === 'jsonl' || e === 'ndjson' ? 'jsonl' : e === 'json' ? 'json' : e === 'tsv' || e === 'tab' ? 'tsv' : 'csv'; };

function parseDelimited(text, sep) {   // RFC 4180: quoted fields, doubled quotes, newlines inside quotes
  const rows = []; let row = [], f = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true; else if (c === sep) { row.push(f); f = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(f); f = ''; if (row.length > 1 || row[0] !== '') rows.push(row); row = []; }
    else f += c;
  }
  if (f !== '' || row.length) { row.push(f); rows.push(row); }
  const head = rows.shift() || []; return rows.map((r) => Object.fromEntries(head.map((h, j) => [h, r[j] ?? ''])));
}

// every row as an object, whatever the format
export async function readRows(bytes, format) {
  if (format === 'parquet') {
    const { parquetReadObjects } = await import(HYP), { compressors } = await import(HYPC);
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return parquetReadObjects({ file: buf, compressors, utf8: true });
  }
  const text = new TextDecoder().decode(bytes);
  if (format === 'jsonl') return text.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
  if (format === 'json') { const j = JSON.parse(text); return Array.isArray(j) ? j : Array.isArray(j.data) ? j.data : Array.isArray(j.rows) ? j.rows : []; }
  return parseDelimited(text, format === 'tsv' ? '\t' : ',');
}

// an embedding cell: an array (or typed array) of numbers, its text form, or an object holding one (a Parquet struct)
export function vectorOf(v) {
  if (v == null) return null;
  if (ArrayBuffer.isView(v) || Array.isArray(v)) { const a = Array.from(v, Number); return a.length > 1 && a.every(Number.isFinite) ? a : null; }
  if (typeof v === 'string' && /^\s*\[/.test(v)) { try { return vectorOf(JSON.parse(v)); } catch { return null; } }
  if (typeof v === 'object') { for (const x of Object.values(v)) { const a = vectorOf(x); if (a) return a; } }   // nested one level (Parquet struct: { model: [...] })
  return null;
}
const dateOf = (v) => { if (v == null || v === '') return null; if (v instanceof Date) return v.getTime(); if (typeof v === 'number' || typeof v === 'bigint') { const x = Number(v); return x > 1e11 ? x : x > 1e9 ? x * 1000 : x >= 1000 && x <= 3000 ? Date.UTC(x, 0) : null; } const t = Date.parse(v); return Number.isFinite(t) ? t : null; };

// the columns and a guess at each role, from the first rows
export function inspect(rows) {
  const sample = rows.slice(0, 500), cols = [...new Set(sample.flatMap((r) => Object.keys(r)))], info = {};
  for (const c of cols) {
    const vals = sample.map((r) => r[c]).filter((v) => v != null && v !== ''), strs = vals.filter((v) => typeof v === 'string');
    const vecs = vals.map(vectorOf).filter(Boolean), dates = vals.map(dateOf).filter((x) => x != null), distinct = new Set(vals.map(String)).size;
    info[c] = { filled: vals.length, avgLen: strs.length ? strs.reduce((a, s) => a + s.length, 0) / strs.length : 0, vecDim: vecs.length > vals.length * 0.9 && vals.length ? vecs[0].length : 0, dateShare: vals.length ? dates.length / vals.length : 0, distinct };
  }
  const by = (f) => cols.filter(f).sort((a, b) => info[b].avgLen - info[a].avgLen);
  const text = by((c) => info[c].avgLen >= 20 && !info[c].vecDim)[0] || null;
  const embedding = cols.find((c) => info[c].vecDim > 1) || null;
  const label = cols.filter((c) => c !== text && !info[c].vecDim && info[c].distinct >= 2 && info[c].distinct <= 40 && info[c].avgLen < 40).sort((a, b) => info[a].distinct - info[b].distinct)[0] || null;
  const date = cols.find((c) => /date|day|time|year|month|created|published/i.test(c) && info[c].dateShare > 0.9) || null;
  return { cols, info, guess: { text, embedding, label, date }, rows: rows.length };
}

// the dataset entry (datasets.js shape) for an uploaded file and its chosen columns
export function uploadDataset(spec) {
  const D = {
    label: spec.name, what: `${spec.name} (your file)`, noun: 'rows', one: 'row', kind: spec.embedding ? 'upload-vectors' : 'text', upload: true, format: spec.format,
    text: (r) => (spec.text ? String(r[spec.text] ?? '') : ''),
    show: (r) => [spec.label && r[spec.label], spec.date && r[spec.date]].filter((x) => x != null && x !== '').join(' · '),
  };
  if (spec.label) { D.category = (r) => r[spec.label]; D.categoryName = spec.label; }
  if (spec.date) { D.date = (r) => dateOf(r[spec.date]); D.time = spec.unit || 'day'; }
  return D;
}
export const SPEC_KEY = 'jet.upload';
export function savedSpec() { try { return JSON.parse(sessionStorage.getItem(SPEC_KEY) || 'null'); } catch { return null; } }
