// In-browser live source: this Web Worker opens a public stream directly, applies a quality filter, embeds each
// batch with MiniLM-L6 through transformers.js (ONNX on WebGPU, fp32, mean pooling, L2-normalised: the same recipe
// as the relay's torch encoder) and posts frames to the page in the relay's frame shape. No server is involved;
// nothing is recorded (use the relay for that). Sources, chosen by the page's `source` field on start:
//   jetstream  Bluesky's Jetstream firehose (websocket, cursor backfill): posts, the relay's filter (English by tag
//              and by text, at least 8 words after removing links and mentions, not link-only, deduplicated).
//   wikipedia  Wikimedia EventStreams recentchange (server-sent events, `since=` backfill): English Wikipedia article
//              edits by people (no bots, main namespace), text = title and edit summary, at least 5 words, deduplicated.
//   nostr      public Nostr relays (websocket; REQ with since/until/limit pages the stored notes back, then the same
//              subscription streams live): kind-1 text notes, the English test, at least 8 words, deduplicated by id.
//   hackernews the Hacker News Firebase API (REST, public, no key): backfill walks item ids back from maxitem, live polls
//              maxitem every 15 s; comments and stories (title plus text), at least 8 words. Slow: about 850 items an hour.
// Every source produces the same frame: {uri, time_us, text, tags, reply, vec}.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

env.allowLocalModels = false;

const JETSTREAM = ['wss://jetstream2.us-east.bsky.network/subscribe', 'wss://jetstream1.us-east.bsky.network/subscribe',
                   'wss://jetstream2.us-west.bsky.network/subscribe', 'wss://jetstream1.us-west.bsky.network/subscribe'];
const HF_ID = 'Xenova/all-MiniLM-L6-v2', TORCH_ID = 'sentence-transformers/all-MiniLM-L6-v2', ENCODER = 'minilm', DIM = 384;
const URL_RE = /https?:\/\/\S+|\bwww\.\S+/gi, MENTION_RE = /(?<![\p{L}\p{N}_])@[\p{L}\p{N}_.\-]+/gu, TAG_RE = /(?<![\p{L}\p{N}_])#([\p{L}\p{N}_]+)/gu;
const DEDUPE_WINDOW = 20000;
const SOURCES = { jetstream: { minWords: 8, englishTest: true }, wikipedia: { minWords: 5, englishTest: false }, nostr: { minWords: 8, englishTest: true }, hackernews: { minWords: 8, englishTest: true } };
const NOSTR_RELAYS = ['wss://relay.primal.net', 'wss://nos.lol', 'wss://relay.damus.io'];
const HN = 'https://hacker-news.firebaseio.com/v0';
let SOURCE = 'jetstream';
const WIKI_STREAM = 'https://stream.wikimedia.org/v2/stream/recentchange';

const EN_WORDS = new Set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'for', 'on', 'with', 'this', 'i', 'be', 'are', 'was', 'have', 'not', 'but', 'my', 'they', 'at', 'so', 'we', 'what', 'if', 'me', 'your', 'can', 'all', 'just', 'like', 'about', 'out', 'up', 'get', 'one', 'do', 'when', 'there', 'would', 'will', 'from', 'or', 'as', 'an', 'by', 'has', 'had', 'his', 'her', 'him', 'who', 'more', 'no', 'new', 'now', 'been', 'were', 'some', 'them', 'than', 'then', 'their', 'into', 'over', 'only', 'very', 'also', 'today']), OTHER_WORDS = [['que', 'não', 'nao', 'para', 'com', 'uma', 'por', 'mais', 'como', 'mas', 'isso', 'agora', 'nossa', 'nosso', 'nossos', 'nossas', 'programa', 'mesmo', 'mesma', 'coisa', 'coisas', 'bom', 'boa', 'dia', 'ano', 'anos', 'vida', 'fazer', 'faz', 'fez', 'vez', 'vezes', 'sempre', 'nunca', 'então', 'entao', 'assim', 'aqui', 'ali', 'lá', 'la', 'esse', 'essa', 'isto', 'pelo', 'pela', 'pelos', 'pelas', 'numa', 'num', 'ele', 'eles', 'elas', 'você', 'vocês', 'vcs', 'vc', 'tá', 'ta', 'né', 'ne', 'obrigado', 'obrigada', 'e', 'o', 'os', 'as', 'do', 'no', 'na', 'nos', 'nas', 'você', 'voce', 'ele', 'ela', 'está', 'esta', 'muito', 'pra', 'dos', 'das', 'meu', 'minha', 'eu', 'tem', 'foi', 'ser', 'também', 'tambem', 'já', 'ja', 'bem', 'quando', 'aqui', 'ainda', 'nada', 'tudo', 'gente', 'até', 'ate', 'vai', 'vou', 'sim', 'são', 'sao', 'todos', 'todo', 'porque', 'pode', 'hoje', 'dia', 'de', 'em', 'da', 'um', 'é', 'ao', 'aos', 'pelo', 'pela', 'três', 'sem', 'seu', 'sua', 'quem', 'onde', 'depois', 'sobre', 'entre', 'contra'], ['que', 'para', 'con', 'una', 'por', 'más', 'mas', 'como', 'pero', 'esto', 'usted', 'él', 'ella', 'está', 'muy', 'los', 'las', 'del', 'mi', 'yo', 'tiene', 'fue', 'ser', 'también', 'ya', 'bien', 'cuando', 'aquí', 'todo', 'todos', 'nada', 'hoy', 'porque', 'puede', 'gracias', 'hola', 'bueno', 'ahora', 'siempre', 'nunca', 'donde', 'de', 'en', 'el', 'es', 'un', 'se', 'le', 'lo', 'al', 'sin', 'sus', 'este', 'esta', 'ese', 'esa'], ['que', 'pour', 'avec', 'une', 'par', 'plus', 'comme', 'mais', 'est', 'très', 'les', 'des', 'du', 'mon', 'je', 'vous', 'nous', 'ils', 'elle', 'dans', 'pas', 'sur', 'ce', 'cette', 'son', 'sa', 'ses', 'tout', 'tous', 'rien', 'parce', 'de', 'en', 'le', 'la', 'et', 'un', 'au', 'aux', 'être', 'avoir', 'fait', 'bien'], ['und', 'der', 'die', 'das', 'ist', 'nicht', 'ich', 'du', 'wir', 'ihr', 'sie', 'mit', 'für', 'auf', 'von', 'zu', 'ein', 'eine', 'auch', 'noch', 'aber', 'wie', 'wenn', 'dann', 'doch', 'mal', 'schon', 'sehr', 'nur', 'dem', 'den', 'des', 'im', 'am'], ['che', 'per', 'con', 'una', 'più', 'come', 'però', 'questo', 'lei', 'lui', 'è', 'molto', 'gli', 'del', 'della', 'nel', 'nella', 'sono', 'anche', 'già', 'bene', 'quando', 'qui', 'tutto', 'tutti', 'niente', 'oggi', 'perché', 'di', 'il', 'la', 'le', 'un', 'non'], ['yang', 'dan', 'di', 'ini', 'itu', 'aku', 'kamu', 'tidak', 'ada', 'untuk', 'dengan', 'dari', 'ke', 'juga', 'saya', 'bisa', 'sudah', 'akan', 'kalau', 'apa', 'mau'], ['en', 'het', 'de', 'een', 'van', 'is', 'niet', 'ik', 'je', 'we', 'zij', 'met', 'voor', 'op', 'te', 'ook', 'nog', 'maar', 'hoe', 'als', 'dan', 'wel', 'al', 'heel', 'alleen']].map(l => new Set(l));
// True when the text reads as English: letters mostly Latin script, at least one English function word, and English function
// words at least as frequent as those of any one other language. The post's own language tag is whatever the app set, often "en" by default.
function looksEnglish(text) {
  const letters = (text.match(/\p{L}/gu) || []).length; if (letters && (text.match(/[A-Za-z\u00C0-\u024F]/g) || []).length < 0.6 * letters) return false;
  const words = (text.toLowerCase().match(/\p{L}+/gu) || []); let en = 0; for (const w of words) if (EN_WORDS.has(w)) en++;
  let other = 0; for (const S of OTHER_WORDS) { let c = 0; for (const w of words) if (S.has(w)) c++; if (c > other) other = c; }
  return en >= 1 && other <= en;
}
const cleanText = (t) => t.replace(URL_RE, ' ').replace(MENTION_RE, ' ').replace(TAG_RE, '$1').replace(/\s+/g, ' ').trim();

let resumeUs = 0;
let extractor = null, device = null, ws = null, es = null, running = false, timer = null, urlK = 0, batchSeconds = 0.25;
let cursorUs = 0, liveStartUs = 0, backfilling = false, lastUs = 0, baseEndUs = 0, CHUNK = 32;   // backfill: replay from cursorUs, then live from liveStartUs on the same socket
let pending = [], recent = new Map();
const stats = { events: 0, post_creates: 0, not_english: 0, too_short: 0, link_only: 0, duplicate: 0, kept: 0, embedded: 0, embed_ms: 0, batches: 0 };
const log = (msg) => postMessage({ type: 'log', msg });

async function loadModel(prefer) {
  const tries = prefer === 'wasm' ? [['wasm', 'fp32']] : [['webgpu', 'fp32'], ['wasm', 'fp32']];
  for (const [dev, dtype] of tries) {
    try {
      const t0 = performance.now();
      const ex = await pipeline('feature-extraction', HF_ID, { device: dev, dtype, progress_callback: (p) => { if (p.status === 'progress' && p.file?.endsWith('.onnx')) postMessage({ type: 'progress', pct: p.progress }); } });
      await ex(['warm up'], { pooling: 'mean', normalize: true });
      extractor = ex; device = dev; return performance.now() - t0;
    } catch (e) { log(`${dev} backend failed: ${e.message || e}`); }
  }
  throw new Error('no backend could load the encoder');
}

let fallback = null;   // wasm extractor, loaded only if the WebGPU backend ever returns a non-finite row
async function embedWith(ex, texts) {
  const out = await ex(texts, { pooling: 'mean', normalize: true, truncation: true, max_length: 128 });
  const v = out.data instanceof Float32Array ? Float32Array.from(out.data) : Float32Array.from(out.data); out.dispose?.();
  if (v.length !== texts.length * DIM) throw new Error(`unexpected embedding size ${v.length} for ${texts.length} posts`);
  return v;
}
const rowFinite = (v, j) => { for (let k = 0; k < DIM; k++) if (!Number.isFinite(v[j * DIM + k])) return false; return true; };
// Embed in chunks of at most 32 (the live batches are far smaller). The WebGPU backend has returned an all-NaN row once in a
// 300-post batch; any non-finite row is re-embedded alone, then on the wasm backend if it is still bad.
async function embed(texts) {
  const v = new Float32Array(texts.length * DIM);
  for (let lo = 0; lo < texts.length; lo += CHUNK) {
    const chunk = texts.slice(lo, lo + CHUNK); const c = await embedWith(extractor, chunk); v.set(c, lo * DIM);
    for (let j = 0; j < chunk.length; j++) {
      if (rowFinite(v, lo + j)) continue;
      let one = await embedWith(extractor, [chunk[j]]);
      if (!rowFinite(one, 0)) { if (!fallback) { fallback = await pipeline('feature-extraction', HF_ID, { device: 'wasm', dtype: 'fp32' }); log('loaded the wasm fallback encoder'); } one = await embedWith(fallback, [chunk[j]]); stats.nan_repaired_wasm = (stats.nan_repaired_wasm || 0) + 1; }
      else stats.nan_repaired = (stats.nan_repaired || 0) + 1;
      v.set(one, (lo + j) * DIM);
    }
  }
  return v;
}

// Quality filter shared by every source; returns true when the item was queued for embedding.
function admit(item, text, target = pending) {
  const cfg = SOURCES[SOURCE];
  let clean = cleanText(text); if (!clean) { stats.link_only++; return false; }
  if (clean.split(/\s+/).length < cfg.minWords) { stats.too_short++; return false; }
  if (clean.length > 1200) clean = clean.slice(0, 1200);   // the encoder truncates at 128 tokens anyway; very long notes only cost tokenizer time
  if (cfg.englishTest && !looksEnglish(clean)) { stats.not_english_text = (stats.not_english_text || 0) + 1; return false; }
  const key = clean.toLowerCase(); if (recent.has(key)) { stats.duplicate++; return false; }
  recent.set(key, 1); if (recent.size > DEDUPE_WINDOW) recent.delete(recent.keys().next().value);
  target.push({ ...item, clean }); stats.kept++; return true;
}
// Wikipedia: edit summaries carry wiki markup; keep the words a reader would see.
const wikiComment = (c) => (c || '').replace(/\/\*\s*(.*?)\s*\*\//g, '$1:').replace(/\[\[(?:[^\]|]*\|)?([^\]]*)\]\]/g, '$1').replace(/\{\{[^}]*\}\}/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
// Edit timestamps have one-second resolution: events in the same second get an ordinal in microseconds, so order within a
// second is arrival order and the clock never freezes. A replayed event (the stream resumes from the last second on a
// reconnect, and partitions are not strictly ordered) is dropped by its id.
let wikiSec = 0, wikiOrd = 0; const seenIds = new Set(); let seenQueue = [];
function connectWikipedia() {
  if (!running) return;
  const from = lastUs || resumeUs || cursorUs; const url = WIKI_STREAM + (from ? `?since=${new Date(Math.floor(from / 1e6) * 1000).toISOString()}` : '');
  const src = new EventSource(url); es = src; let last = performance.now();
  const reconnect = (why) => { if (src !== es) return; clearInterval(watchdog); src.close(); es = null; if (running) { log(`${why}; reconnecting from ${new Date(Math.floor((lastUs || cursorUs) / 1e6) * 1000).toISOString()}`); setTimeout(connectWikipedia, 1500); } };
  const watchdog = setInterval(() => { if (src === es && performance.now() - last > 60000) reconnect('no events for 60 s'); }, 5000);
  src.onopen = () => log(`connected ${url}`);
  src.onmessage = (ev) => {
    last = performance.now(); stats.events++;
    let e; try { e = JSON.parse(ev.data); } catch { return; }
    const sec = e.timestamp || 0; if (sec !== wikiSec) { wikiSec = sec; wikiOrd = 0; } const tUs = sec * 1e6 + (wikiOrd++); if (tUs > lastUs) lastUs = tUs;
    if (backfilling && tUs >= liveStartUs) { backfilling = false; postMessage({ type: 'backfilled', liveStartUs, cursorUs }); }
    if (e.wiki !== 'enwiki' || (e.type !== 'edit' && e.type !== 'new') || e.namespace !== 0 || e.bot) return;
    const id = `enwiki/${e.revision?.new ?? e.id ?? e.meta?.id}`; if (seenIds.has(id)) { stats.replayed = (stats.replayed || 0) + 1; return; } seenIds.add(id); seenQueue.push(id); if (seenQueue.length > 100000) { seenIds.delete(seenQueue.shift()); }
    stats.post_creates++;
    const comment = wikiComment(e.comment); if (!comment) { stats.link_only++; return; }
    const text = `${e.title}: ${comment}`;
    admit({ uri: id, time_us: tUs, did: '', rkey: '', created_at: e.meta?.dt, text, tags: [], reply: 0, embed: 0, title: e.title, url: e.meta?.uri, kind: e.type, minor: e.minor ? 1 : 0, delta: e.length ? (e.length.new || 0) - (e.length.old || 0) : 0 }, text);
  };
  src.onerror = () => reconnect('stream error or drop');   // never let the browser's own retry replay from the original since=
}
// One-second timestamps get an ordinal in microseconds so order within a second is arrival order.
let ordSec = 0, ordN = 0; const stampUs = (sec) => { if (sec !== ordSec) { ordSec = sec; ordN = 0; } return sec * 1e6 + (ordN++); };
// HTML in Hacker News comments: paragraphs and links become spaces, entities are decoded.
const ENT = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ' };
const decodeHtml = (h) => (h || '').replace(/<p>/gi, ' ').replace(/<[^>]+>/g, ' ').replace(/&#x([0-9a-f]+);/gi, (_, x) => String.fromCodePoint(parseInt(x, 16))).replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(+d)).replace(/&(\w+);/g, (m, n) => ENT[n] ?? m).replace(/\s+/g, ' ').trim();

// Nostr: stored notes are paged back from the live start to the cursor in pages of 500 (relays cap a REQ at about that), each
// page newest first, so the backfill is collected, sorted by time and only then queued for embedding (the page expects time
// order to know when the base window is complete). The paging state lives outside the socket: a relay that cuts the
// subscription (relay.damus.io meters reads at 1 MB per minute per address) hands the same page to the next relay, and ids
// are deduplicated across relays. After the last page the socket carries a live subscription.
let nostrK = 0; const nostrSeen = new Set(); let nostrSeenQ = [];
const nostr = { until: 0, page: 0, backlog: [], sinceSec: 0, started: false };
function connectNostr() {
  if (!running) return;
  const url = NOSTR_RELAYS[nostrK++ % NOSTR_RELAYS.length]; const sock = new WebSocket(url); ws = sock; let last = performance.now();
  if (!nostr.started) { nostr.started = true; nostr.sinceSec = Math.floor((resumeUs || cursorUs) / 1e6); nostr.until = backfilling ? Math.floor(liveStartUs / 1e6) : 0; }
  let got = 0, oldest = Infinity, sub = '';
  const req = () => { nostr.page++; got = 0; oldest = Infinity; sub = 'p' + nostr.page; const f = { kinds: [1], since: nostr.sinceSec, limit: 500 }; if (nostr.until) f.until = nostr.until; sock.send(JSON.stringify(['REQ', sub, f])); };
  const live = () => { sub = 'live'; sock.send(JSON.stringify(['REQ', sub, { kinds: [1], since: Math.floor((lastUs || liveStartUs) / 1e6) }])); };
  const handOver = (why) => { if (sock !== ws) return; log(`${why}; continuing on the next relay`); ws = null; sock.close(); setTimeout(connectNostr, 200); };
  const watchdog = setInterval(() => { if (sock === ws && performance.now() - last > 30000) handOver('no events for 30 s'); }, 5000);
  sock.onopen = () => { log(`connected ${url}`); if (nostr.until) req(); else live(); };
  sock.onmessage = (ev) => {
    last = performance.now(); stats.events++;
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m[0] === 'EOSE' && m[1] === sub) {
      if (nostr.until) {
        sock.send(JSON.stringify(['CLOSE', sub]));
        if (got >= 400 && oldest > nostr.sinceSec) { nostr.until = oldest - 1; req(); return; }   // an earlier page of stored notes
        nostr.until = 0; nostr.backlog.sort((a, b) => a.time_us - b.time_us); for (const p of nostr.backlog) pending.push(p); log(`backfill complete: ${nostr.backlog.length} notes kept over ${nostr.page} pages`); nostr.backlog = [];
        backfilling = false; postMessage({ type: 'backfilled', liveStartUs, cursorUs }); live();
      }
      return;
    }
    if (m[0] === 'CLOSED' || m[0] === 'NOTICE') { log(`relay says ${JSON.stringify(m).slice(0, 160)}`); if (m[0] === 'CLOSED' && m[1] === sub) { nostr.page--; handOver('subscription closed by the relay'); } return; }
    if (m[0] !== 'EVENT' || m[1] !== sub) return; const e = m[2]; if (!e || e.kind !== 1 || typeof e.content !== 'string') return;
    if (nostrSeen.has(e.id)) { stats.replayed = (stats.replayed || 0) + 1; got++; if (e.created_at < oldest) oldest = e.created_at; return; } nostrSeen.add(e.id); nostrSeenQ.push(e.id); if (nostrSeenQ.length > 200000) nostrSeen.delete(nostrSeenQ.shift());
    got++; if (e.created_at < oldest) oldest = e.created_at;
    const tUs = stampUs(e.created_at); if (tUs > lastUs) lastUs = tUs; stats.post_creates++;
    const tags = []; for (const t of e.tags || []) if (t[0] === 't' && t[1]) { tags.push(String(t[1])); if (tags.length >= 8) break; }
    const reply = (e.tags || []).some(t => t[0] === 'e') ? 1 : 0;
    admit({ uri: 'nostr/' + e.id, time_us: tUs, did: '', rkey: '', created_at: new Date(e.created_at * 1000).toISOString(), text: e.content, tags, reply, embed: 0, url: 'https://njump.me/' + e.id, kind: 'note' }, e.content, nostr.until ? nostr.backlog : pending);
  };
  sock.onclose = () => { clearInterval(watchdog); if (sock !== ws) return; ws = null; if (running) { log('relay closed; next relay in 2 s'); if (nostr.until) nostr.page--; setTimeout(connectNostr, 2000); } };
  sock.onerror = () => {};
}

// Hacker News: item ids are issued in time order, so the backfill walks back from maxitem, 24 items at a time, until most of
// a batch is older than the cursor; live polls maxitem every 15 s and fetches the new ids. Comments carry reply: 0 here on
// purpose: they are the bulk of the stream and would otherwise be left out of the base map by the originals-only rule.
let hnMax = 0, hnTimer = null, hnBusy = false;
const hnItem = async (id) => { try { const r = await fetch(`${HN}/item/${id}.json`); return r.ok ? await r.json() : null; } catch { return null; } };
function hnAdmit(it, target) {
  if (!it || it.deleted || it.dead || !it.time) return; let text;
  if (it.type === 'comment') text = decodeHtml(it.text); else if (it.type === 'story' || it.type === 'job' || it.type === 'poll') text = (it.title || '') + (it.text ? ': ' + decodeHtml(it.text) : ''); else return;
  stats.post_creates++; const tUs = stampUs(it.time); if (tUs > lastUs) lastUs = tUs;
  admit({ uri: 'hn/' + it.id, time_us: tUs, did: '', rkey: '', created_at: new Date(it.time * 1000).toISOString(), text, tags: [], reply: 0, embed: 0, url: `https://news.ycombinator.com/item?id=${it.id}`, kind: it.type === 'comment' ? 'comment' : it.type, score: it.score ?? null, descendants: it.descendants ?? null }, text, target);
}
async function connectHackerNews() {
  if (!running) return;
  try { hnMax = await (await fetch(`${HN}/maxitem.json`)).json(); } catch (e) { log('maxitem failed; retrying in 5 s'); setTimeout(connectHackerNews, 5000); return; }
  log(`connected ${HN} (maxitem ${hnMax})`);
  if (backfilling) {
    const sinceSec = Math.floor((resumeUs || cursorUs) / 1e6), backlog = []; let id = hnMax, stop = false, fetched = 0;
    while (running && !stop) {
      const ids = []; for (let k = 0; k < 24 && id > 0; k++) ids.push(id--);
      const items = await Promise.all(ids.map(hnItem)); fetched += ids.length; let old = 0;
      for (const it of items) { if (!it || !it.time) continue; if (it.time < sinceSec) { old++; continue; } hnAdmit(it, backlog); }
      if (old >= 12) stop = true;
      if (fetched % 240 === 0) postMessage({ type: 'stats', stats: { ...stats, pending: backlog.length, fetched } });
    }
    if (!running) return;
    backlog.sort((a, b) => a.time_us - b.time_us); for (const p of backlog) pending.push(p);
    backfilling = false; postMessage({ type: 'backfilled', liveStartUs, cursorUs }); log(`backfill: ${fetched} items fetched, ${backlog.length} kept`);
  }
  hnTimer = setInterval(async () => {
    if (hnBusy || !running) return; hnBusy = true;
    try { const m = await (await fetch(`${HN}/maxitem.json`)).json(); const ids = []; for (let i = hnMax + 1; i <= m; i++) ids.push(i); hnMax = m;
      for (let lo = 0; lo < ids.length; lo += 24) { const items = await Promise.all(ids.slice(lo, lo + 24).map(hnItem)); for (const it of items) hnAdmit(it, pending); } }
    catch (e) { log('poll failed: ' + (e.message || e)); } finally { hnBusy = false; }
  }, 15000);
}
function connect() {
  if (SOURCE === 'wikipedia') return connectWikipedia();
  if (SOURCE === 'nostr') return connectNostr();
  if (SOURCE === 'hackernews') return connectHackerNews();
  if (!running) return;
  const from = lastUs || resumeUs || cursorUs; const url = JETSTREAM[urlK++ % JETSTREAM.length] + '?wantedCollections=app.bsky.feed.post' + (from ? `&cursor=${from}` : '');
  let last = performance.now(); const sock = new WebSocket(url); ws = sock;
  const watchdog = setInterval(() => { if (sock === ws && performance.now() - last > 30000) { log('no events for 30 s, reconnecting'); sock.close(); } }, 5000);
  sock.onopen = () => log(`connected ${url}`);
  sock.onmessage = (ev) => {
    last = performance.now(); stats.events++;
    let e; try { e = JSON.parse(ev.data); } catch { return; }
    if (e.time_us) { lastUs = e.time_us; if (backfilling && e.time_us >= liveStartUs) { backfilling = false; postMessage({ type: 'backfilled', liveStartUs, cursorUs }); } }
    const c = e.commit; if (!c || c.operation !== 'create' || c.collection !== 'app.bsky.feed.post') return;
    stats.post_creates++;
    const r = c.record || {}, text = r.text || '';
    if (backfilling && baseEndUs && e.time_us < baseEndUs && r.reply) { stats.base_replies_skipped = (stats.base_replies_skipped || 0) + 1; return; }   // never used: the base map is originals only
    if (!(r.langs || []).includes('en')) { stats.not_english++; return; }
    const tags = []; for (const m of text.matchAll(TAG_RE)) { tags.push(m[1]); if (tags.length >= 8) break; }
    admit({ uri: e.did + '/' + c.rkey, time_us: e.time_us, did: e.did, rkey: c.rkey, created_at: r.createdAt, text, tags, reply: r.reply ? 1 : 0, embed: r.embed ? 1 : 0 }, text);
  };
  sock.onclose = () => { clearInterval(watchdog); if (sock !== ws) return; ws = null; if (running) { log('connection closed; retrying in 3 s'); setTimeout(connect, 3000); } };
  sock.onerror = () => {};
}

async function flushBatch() {
  if (!pending.length || !extractor) return;
  const batch = pending.splice(0, 512);   // at most 512 per frame, so a backfill shows progress and never lands as one block
  const t0 = performance.now(); let vecs;
  try { vecs = await embed(batch.map(p => p.clean)); } catch (e) { postMessage({ type: 'error', msg: 'embedding failed: ' + (e.message || e) }); return; }
  const ms = performance.now() - t0; stats.embedded += batch.length; stats.embed_ms += ms; stats.batches++;
  if (ms > Math.max(300, 4 * batch.length)) { stats.slow_embeds = (stats.slow_embeds || 0) + 1; log(`slow embed: ${ms.toFixed(0)} ms for ${batch.length} posts (longest ${Math.max(...batch.map(p => p.clean.length))} chars)`); }   // over 4 ms per post is slow; a 512-post backfill batch normally takes ~0.7 s
  for (const p of batch) delete p.clean;
  const hdr = { n: batch.length, dim: DIM, model: TORCH_ID, encoder: ENCODER, source: 'browser', stream: SOURCE, device, embed_ms: ms, posts: batch, cursorUs, liveStartUs, pendingAfter: pending.length, backfilling };
  postMessage({ type: 'frame', hdr, vecs }, [vecs.buffer]);
}

let busy = false;
async function tick() { if (busy) return; busy = true; try { await flushBatch(); } finally { busy = false; } postMessage({ type: 'stats', stats: { ...stats, pending: pending.length } }); }

onmessage = async (ev) => {
  const m = ev.data;
  if (m.type === 'start') {
    batchSeconds = m.batchSeconds || 0.25; SOURCE = SOURCES[m.source] ? m.source : 'jetstream';
    if (m.backfillMinutes > 0) { cursorUs = m.cursorUs || (Date.now() - m.backfillMinutes * 60000) * 1000; resumeUs = m.resumeUs || 0; liveStartUs = Date.now() * 1000; backfilling = true; lastUs = 0; baseEndUs = m.baseMinutes ? cursorUs + m.baseMinutes * 60e6 : 0; }
    if (m.embedChunk) CHUNK = m.embedChunk;
    try {
      if (!extractor) { postMessage({ type: 'loading' }); const ms = await loadModel(m.device); postMessage({ type: 'ready', model: TORCH_ID, hf: HF_ID, device, loadMs: ms, cursorUs, liveStartUs, stream: SOURCE }); }
      running = true; connect(); timer = setInterval(tick, batchSeconds * 1000);
    } catch (e) { postMessage({ type: 'error', msg: e.message || String(e) }); }
  } else if (m.type === 'batch') {   // the page changed its flush interval: re-arm the batch timer
    batchSeconds = m.batchSeconds || 0.25; if (timer) { clearInterval(timer); timer = setInterval(tick, batchSeconds * 1000); }
  } else if (m.type === 'stop') {
    running = false; clearInterval(timer); timer = null; if (hnTimer) { clearInterval(hnTimer); hnTimer = null; } if (ws) { const s = ws; ws = null; s.close(); } if (es) { const s = es; es = null; s.close(); } pending = []; postMessage({ type: 'stopped' });
  } else if (m.type === 'bench') {   // embedding throughput at several chunk sizes, for choosing CHUNK
    try { if (!extractor) await loadModel(m.device); const out = []; for (const c of (m.chunks || [16, 32, 64, 128])) { const saved = CHUNK; CHUNK = c; await embed(m.texts.slice(0, c)); const t0 = performance.now(); await embed(m.texts); out.push({ chunk: c, ms: performance.now() - t0, perPost: (performance.now() - t0) / m.texts.length }); CHUNK = saved; } postMessage({ type: 'bench', out, device }); }
    catch (e) { postMessage({ type: 'error', msg: e.message || String(e) }); }
  } else if (m.type === 'parity') {   // embed given texts and return the vectors, for checking against the relay's torch vectors
    try { if (!extractor) { const ms = await loadModel(m.device); postMessage({ type: 'ready', model: TORCH_ID, hf: HF_ID, device, loadMs: ms }); } const vecs = await embed(m.texts.map(cleanText)); postMessage({ type: 'parity', vecs, device }, [vecs.buffer]); }
    catch (e) { postMessage({ type: 'error', msg: e.message || String(e) }); }
  }
};
