import { loadData, StreamingEngine, JACOBI_ITERS, FLUSH_CAP, kmeans, pcaAxes, pcaProject } from './engine.js';
import { openCache, readRange, writeFrame, prune, countAll } from './cache.js';
import { PointRenderer, heatRgb, HEAT_PALETTES } from './render.js';
import { tunedLayout, tunedFromParam, tunedToParam } from './livefit.js';
import { saveHandoff, loadHandoff, saveSnapshot, dropSnapshot } from './handoff.js';
import { TuneMode } from './tunemode.js';
import { Cast3D } from './cast3d.js';
import { DATASETS } from './datasets.js';
import { PALETTE, computeRegions, computeRegionsSteps, docsSteps, runSliced, nameClusters, docsOf, setNsfwWords, setNsfwFilter, nsfwFilterOn , setNamingMinShare, setNamingSlots } from './regions.js';

const $ = (id) => document.getElementById(id);
const K = 15;
const params = new URLSearchParams(location.search);
// Opt-in tuned base (?base=tuned&tuned=..., written by the base map tuner): the base layout is the tuner's fit for those
// settings instead of fitBase. Only the layout changes: the 15-neighbour lists, rho and sigma that Batched RevAdopt and the
// NodeGardener quiet rebuilds read, and their a and b (spread 1.2, min distance 0.03), stay the live map's, so a tuned k or
// min distance shapes the base map but not how arrivals are placed. Without the parameter nothing here runs.
let TUNED = null, TUNED_ERR = null;
if (params.get('base') === 'tuned') { try { TUNED = tunedFromParam(params.get('tuned')); } catch (e) { TUNED_ERR = e; } }
// Opt-in handed-over base (?base=handoff, from the tuner's Use in live map): the base rows, vectors, positions and topic
// clusters come from handoff.js exactly as the tuner showed them; only the 15-neighbour lists are recomputed (fitBase with
// no layout steps), and the stream window keeps the base's start, so arrivals are the posts after that base.
let HANDOFF = null;
// Stream sources the in-browser worker can read on a fresh start (recordings are always Bluesky). Switching source is a
// fresh start on the other stream's last hour: the base map is fitted on one source's vectors.
const SOURCES = {
  jetstream: { key: 'jetstream', what: 'Bluesky posts', noun: 'posts', one: 'post', feed: 'Jetstream', h1: 'each new post lands beside its nearest neighbours as it arrives', minWords: 8, filter: 'English only, at least eight words after removing links and mentions, replies left out of the base map' },
  wikipedia: { key: 'wikipedia', what: 'English Wikipedia edits', noun: 'edits', one: 'edit', feed: 'Wikimedia EventStreams', h1: 'each new edit (title and summary) lands beside its nearest neighbours as it happens', minWords: 5, filter: 'article edits by people (no bots), at least five words of title and summary' },
  nostr: { key: 'nostr', what: 'Nostr notes', noun: 'notes', one: 'note', feed: 'public Nostr relays', h1: 'each new note lands beside its nearest neighbours as it arrives', minWords: 8, filter: 'English by text, at least eight words after removing links and mentions, replies left out of the base map; the default window is two hours', fresh: 120 },
  opensky: { key: 'opensky', what: 'aircraft states (OpenSky ADS-B)', noun: 'samples', one: 'sample', feed: 'OpenSky through the local relay', h1: 'each new aircraft state lands beside its nearest neighbours as it arrives · local relay', minWords: 0, filter: 'every aircraft with a position in the relay\'s box, one sample per new position report; features, not text', fresh: 60, numeric: true, dim: 48, model: 'opensky-state-v3', worker: './opensky_worker.js' },
  hackernews: { key: 'hackernews', what: 'Hacker News comments and stories', noun: 'items', one: 'item', feed: 'the Hacker News API', h1: 'each new comment or story lands beside its nearest neighbours as it is posted', minWords: 8, filter: 'comments and stories, at least eight words; about 850 items an hour, so the default window is three hours', fresh: 180 },
};
// Static datasets (datasets.js): a finished table, loaded and embedded by static_worker.js, played like a recording (the
// first half of its rows is the base map, the rest arrives as the stream); nothing arrives after it.
for (const [key, D] of Object.entries(DATASETS)) SOURCES[key] = { key, what: D.what, noun: D.noun, one: D.one, feed: D.label, h1: `${D.label}: ${params.get('until') && D.time ? `the map up to a date; Play places the rest in date order` : `one map of every ${D.one}; similar ${D.noun} sit together`}`, minWords: 0, filter: `${D.what}; ${D.time ? 'in date order on a time axis by ' + D.time : 'in a fixed random order, with no time axis'}`, fresh: 60, static: true, kind: D.kind, rows: +(params.get('rows') || 0), worker: './static_worker.js' };
// A packaged copy (a shared artifact) carries recordings in rec/ and cannot reach the live feeds: the dataset comes from the
// link's #token (artifacts pass no query string), each source reads its recording through recorded_worker.js.
const RECORDED = await fetch('./rec/index.json', { cache: 'no-store' }).then(r => (r.ok ? r.json() : null)).catch(() => null);
const REC_KEY = RECORDED ? (RECORDED.sources[location.hash.slice(1)] ? location.hash.slice(1) : RECORDED.default) : null;
const BAD_SOURCE = params.get('source') && !SOURCES[params.get('source')] ? params.get('source') : null;
const SOURCE = RECORDED ? RECORDED.sources[REC_KEY].source : SOURCES[params.get('source')] ? params.get('source') : 'jetstream', SRC = SOURCES[SOURCE];
if (RECORDED) { const r = RECORDED.sources[REC_KEY]; Object.assign(SRC, { recorded: `./rec/${REC_KEY}/`, worker: './recorded_worker.js', fresh: r.minutes, feed: 'a recording', h1: SRC.h1.replace(' · local relay', '') + ' · recorded ' + r.when }); addEventListener('hashchange', () => location.reload()); }
if (SRC.numeric) setNamingMinShare(0.25);
// Fresh start (the default): the last FRESH minutes of the firehose are fetched and embedded in the browser, the base map
// is fitted from the first half, the second half is the playable stream, then live. `?capture=` loads a recording instead.
const FRESH = params.has('capture') || params.get('fresh') === '0' ? 0 : +(params.get('fresh') || SRC.fresh || 60);
const CAPTURE = FRESH ? 'fresh' : (params.get('capture') || 'capture_20261012_0945pdt'), BASE = FRESH ? 'fresh' : (params.get('base') || 'base_30min_originals');
const BUDGET = +(params.get('budget') || 0);   // living arrivals kept on the map (0 = unlimited); older ones retire
const LIVE = FRESH > 0 || params.get('live') === '1', LIVE_PORT = +(params.get('port') || 8788), LIVE_SRC = FRESH ? (SOURCE === 'jetstream' ? 'browser' : SOURCE) : (params.get('src') || 'browser');
const CAP = +(params.get('cap') || (SRC.static && FRESH ? 256000 : FRESH ? (BUDGET ? Math.ceil(BUDGET * 1.15) + 65536 : 220000) : LIVE ? (BUDGET ? Math.ceil(BUDGET * 1.15) + 16384 : 150000) : 0));   // row slots reserved for live arrivals
const CAPTURE_URL = `../data/jetstream_demo/${CAPTURE}/`, RELAY = params.get('relay') || 'http://127.0.0.1:8790';   // RELAY: the OpenSky relay
const state = { playing: false, live: false, ws: null, speed: 10, interval: 0.5, clock: 0, lastStats: null, cumulativeMs: 0, log: [], window: 30, autoThreshold: true, manualKept: 4, baseQ5: 0, baseMedian: 0.4, settleMs: 500, inspect: null, inspectMode: 'both', achieved: 0, liveShift: null, clockWall: 0, animOn: false, frames: 0, worker: null, liveSource: LIVE_SRC, liveStats: null, budget: BUDGET, countFrom: 0, placeBins: new Map(), lastWave: null, arriveBins: new Map(), ledgerLog: [],
  autoRebuild: true, rebuildMinutes: 0.5, rebuildArrivals: 0, playedMs: 0, playWall: 0, lastRebuildPlayed: 0, leash: 30, morphMs: 3000, rebuilding: false, morphing: false, flushBusy: false, lastRebuildClock: 0, rebuildMarks: [], rebuildRequested: false };
function countArrivals(from, to) { for (let p = from; p < to; p++) { const b = Math.floor(data.tS[p] / 10); state.arriveBins.set(b, (state.arriveBins.get(b) || 0) + 1); } }
let tuner = null, cast = null, data, engine, renderer, device, scores, regions = null, regionS = null, regionPrevS = null, prevRegions = null;
const rstate = { on: true, chips: true, sigma: 4, soft: 6, tint: 'topic', tau: 0.1, focus: -1 };
state.heat = 'off'; state.rates = null; state.histMinute = -1; state.rateWin = 60;

const events = [];
function logEvent(kind, detail) { const e = { t: Math.round(performance.now()), played: Math.round(playedNow() / 1000), clock: Math.round(state.clock), kind, ...detail }; events.push(e); if (events.length > 2000) events.shift(); console.info('[jet]', kind, JSON.stringify(e)); }
// the status note sits on the map's lower right and fades 7 s after the last message, unless it is an error or work in progress (…)
function status(msg, isError = false) { const el = $('status'); el.textContent = msg; el.classList.toggle('error', isError); el.classList.remove('quiet'); clearTimeout(status.t); if (!isError && !/…\s*$/.test(msg)) status.t = setTimeout(() => el.classList.add('quiet'), 7000); if (isError) document.getElementById('snapOverlay')?.remove(); }
const median = (arr) => { if (!arr.length) return NaN; const a = Float64Array.from(arr).sort(); const m = a.length >> 1; return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]); };
const quantile = (arr, q) => { if (!arr.length) return NaN; const a = Float64Array.from(arr).sort(); return a[Math.min(a.length - 1, Math.floor(q * a.length))]; };
const fmt = (x, d = 1) => Number.isFinite(x) ? x.toFixed(d) : '—';
const mmss = (s) => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, ss = s % 60; return (h ? h + ':' : '') + String(m).padStart(h ? 2 : 1, '0') + ':' + String(ss).padStart(2, '0'); };
const wall = (s) => data.t0us ? new Date(data.t0us / 1000 + s * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
// A dated static dataset: the worker maps its dates linearly onto the window, so stream time reads back as a date in the
// dataset's unit (month, year, day); a dataset without dates has no time axis at all.
const YEAR_MS = 365.2425 * 86400000;
const dateAt = (t) => SRC.timeAxis.d0 + t / SRC.timeSpan * (SRC.timeAxis.d1 - SRC.timeAxis.d0), tOfDate = (ms) => (ms - SRC.timeAxis.d0) / (SRC.timeAxis.d1 - SRC.timeAxis.d0) * SRC.timeSpan;
function tLabel(t) {
  if (!SRC.timeAxis) return mmss(t);
  const d = new Date(dateAt(t)), u = SRC.timeAxis.unit, o = { timeZone: 'UTC', year: 'numeric' };
  return u === 'year' ? String(d.getUTCFullYear()) : d.toLocaleDateString([], u === 'month' ? { ...o, month: 'short' } : { ...o, month: 'short', day: 'numeric' });
}
const noTime = () => SRC.static && !SRC.timeAxis;
// A static dataset is one map of every row, fitted in one go (no base and stream halves, no Batched RevAdopt). Dragging the
// timeline back offers to fit the map of the rows dated up to then (?until=, a date in ms); Play then places the rest
// with Batched RevAdopt, and Full map returns to the whole table.
const UNTIL = SRC.static ? +(params.get('until') || 0) : 0;
const loadedEnd = () => data.n > data.nBase ? data.tS[data.n - 1] : data.baseSeconds;

function scoreOf(p) { const W = state.window, b = p * K; let c = 0; for (let k = 0; k < K; k++) if (engine.ranks[b + k] < W) c++; return c / K; }
function recomputeScores(from = 0) { for (let p = from; p < engine.n; p++) scores[p] = scoreOf(p); renderer.setScores(scores); }
function threshold() { return state.autoThreshold ? (state.baseQ5 + 0.5) / K : (state.manualKept - 0.5) / K; }
function applyThreshold() { renderer.threshold = threshold(); $('thrv').textContent = state.autoThreshold ? `stranded at ${state.baseQ5} or fewer of ${K} kept (base map's 5th percentile)` : `stranded below ${state.manualKept} of ${K} kept`; }
function updateBaseP5() { const s = []; for (let p = 0; p < data.nBase; p++) s.push(scores[p]); state.baseQ5 = Math.round(quantile(s, 0.05) * K); state.baseMedian = median(s); renderer.fitMid = state.baseMedian; }
// At-placement stranding, accumulated per 10 s of stream time as rows are placed (so retirement and compaction cannot disturb it)
function recordPlacement(st) { const thr = threshold(); for (let r = 0; r < st.w; r++) { const p = st.start + r, b = Math.floor(data.tS[p] / 10); const e = state.placeBins.get(b) || [0, 0]; e[1]++; if (scores[p] < thr) e[0]++; state.placeBins.set(b, e); } }
function strandedCurve(binSec) { const out = new Map(); for (const [b10, e] of state.placeBins) { const b = Math.floor(b10 * 10 / binSec); const o = out.get(b) || [0, 0]; o[0] += e[0]; o[1] += e[1]; out.set(b, o); } return new Map([...out].map(([b, e]) => [b, e[0] / e[1]])); }

function updateReadouts(st) {
  $('r-time').textContent = SRC.static ? (SRC.timeAxis ? `up to ${tLabel(state.clock)}` : `${engine.n.toLocaleString()} of ${data.n.toLocaleString()} placed`) : `${mmss(state.clock)} of stream time · latest post ${wall(engine.n > data.nBase ? data.wallS[engine.n - 1] : data.baseSeconds)} local`;
  $('s-time').innerHTML = noTime() ? '' : `<b>${tLabel(state.clock)}</b>`; $('s-map').innerHTML = `<b>${(engine.n - engine.deadPlaced).toLocaleString()}</b> ${SRC.noun}`;
  $('s-arr').innerHTML = st ? `<b class="arr">+${st.w.toLocaleString()}</b> placed` : '—'; $('s-ms').innerHTML = st ? `<b>${fmt(st.msTotal, 0)}</b> ms` : '—';
  if (!st) return; const thr = threshold(); const kept = [], settle = []; let stranded = 0;
  for (let r = 0; r < st.w; r++) { const p = st.start + r; kept.push(scores[p] * K); settle.push(engine.settleDist[p]); if (scores[p] < thr) stranded++; }
  $('r-mapsize').textContent = st.n.toLocaleString(); $('r-arrivals').textContent = `${st.w.toLocaleString()} posts (${mmss(st.tFirst)} → ${mmss(st.tLast)})`;
  $('r-fit').textContent = `median ${fmt(median(kept))} of ${K} meaning-neighbours are within the ${state.window} nearest on the map (base map: ${fmt(state.baseMedian * K)})`;
  $('r-stranded').textContent = `${stranded} of ${st.w} (${fmt(100 * stranded / st.w)} %)`;
  $('r-settle').textContent = `median ${fmt(median(settle), 3)} map units from barycentre start to final (map is ${fmt(renderer.fitExtent)} across)`;
  $('r-adopters').textContent = `${st.adoptersChanged.toLocaleString()} history rows changed (meaning), ${(st.mapAdopters ?? 0).toLocaleString()} (map)`;
  { const L = st.ledger || engine.ledger, placed = Math.max(1, engine.placedTotal); $('r-ledger').textContent = `${L.net.toLocaleString()} entries now (${(L.net / Math.max(1, engine.n - engine.deadPlaced)).toFixed(2)} per row; base map ${(L.base ?? 0).toLocaleString()}) · +${L.created} −${L.removed} this flush · ${L.gross.toLocaleString()} created since the base (${(L.gross / placed).toFixed(1)} per arrival)`; }
  $('r-iters').textContent = `${st.iters} of ${JACOBI_ITERS} · final mean movement ${st.movement.toExponential(2)}`;
  $('r-ms').textContent = `search ${fmt(st.msSearch)} · graph ${fmt(st.msGraph)} · weights ${fmt(st.msWeights)} · solve+rank ${fmt(st.msSolve)} · total ${fmt(st.msTotal)} ms`;
  { const w = state.lastWave; $('r-history').textContent = `${(engine.n - data.nBase - engine.deadPlaced).toLocaleString()} living arrivals` + (state.budget ? ` of a budget of ${state.budget.toLocaleString()}` : effectiveBudget() ? ` of an automatic budget of ${effectiveBudget().toLocaleString()} (what the ${CAP.toLocaleString()} reserved row slots can hold; set History budget to choose)` : ' (no budget)') + (state.droppedFull ? ` · ${state.droppedFull.toLocaleString()} dropped with the row store full` : '') + ` · ${engine.retiredTotal.toLocaleString()} retired · ${engine.compactions} compactions` + (w ? ` · last wave: ${w.retired} retired, ${w.refreshed} lists refreshed in ${w.ms.toFixed(0)} ms` + (w.compacted ? `, compacted in ${w.compactMs.toFixed(0)} ms` : '') : ''); }
  { const L = engine.ledger, r = engine.lastRebuild, since = engine.placedTotal - engine.placedAtRebuild; $('r-rebuild').textContent = (r ? `last: ${r.rows.toLocaleString()} rows, ${r.edges.toLocaleString()} edges, ${(r.msTotal / 1000).toFixed(1)} s · history moved median ${r.histMedian.toFixed(2)} / p95 ${r.histP95.toFixed(2)} widths (${r.nHist.toLocaleString()} rows), arrivals re-seated median ${r.arrMedian.toFixed(2)} (${r.nArr.toLocaleString()}) · ledger ${r.ledgerBefore.toLocaleString()} → ${r.ledgerAfter.toLocaleString()}` : 'none yet') + ` · since: ${since.toLocaleString()} arrivals, ${L.since.toLocaleString()} ledger entries, ${mmss(Math.max(0, playedNow() - state.lastRebuildPlayed) / 1000)} played`; }
  $('r-cum').textContent = `${engine.flushes} flushes, ${engine.placedTotal.toLocaleString()} posts placed in ${(state.cumulativeMs / 1000).toFixed(1)} s of GPU+CPU work` + (state.achieved ? ` · playing at ${fmt(state.achieved, 0)}× real time` : '');
  $('r-hash').textContent = st.hash.toString(16).padStart(8, '0');
}

// The footer's time axis: stream time to pixels, shared by the timeline, the lineage view and the scrub.
function timeScale(W) { const tEnd = Math.max(loadedEnd(), data.baseSeconds + 1), yrSec = SRC.timeAxis ? SRC.timeSpan / ((SRC.timeAxis.d1 - SRC.timeAxis.d0) / YEAR_MS) : 0, binSec = SRC.timeAxis ? (SRC.timeSpan / yrSec > 6 ? yrSec : yrSec / 12) : tEnd > 3 * 3600 ? 300 : (tEnd > 1800 ? 60 : 10), nb = Math.ceil(tEnd / binSec) + 1, span = nb * binSec; return { tEnd, binSec, nb, span, bw: W / nb, xOf: (t) => t / span * W, tOf: (x) => x / W * span }; }
// The lineage's own axis: from the base map's end (where the first identities begin) to the loaded end, so the bands fill the width.
function lineageScale(W) { const ts = timeScale(W), t0 = data.baseSeconds, span = Math.max(1, ts.tEnd - t0); return { ...ts, t0, xOf: (t) => (t - t0) / span * W, tOf: (x) => t0 + x / W * span }; }
let footerMode = 'timeline';
function setFooter(mode) { footerMode = mode; $('timeline').hidden = mode !== 'timeline'; $('lineage').hidden = mode !== 'lineage'; $('lineage-btn').classList.toggle('on', mode === 'lineage'); drawTimeline(); }
function drawCursors(ctx, ts, top, H, bot) {
  // rebuilds: a rug of short ticks above the plot, marks closer than 4 px merged into one, so a burst of rebuilds (a replay's
  // preparation, fast play) reads as a dense stretch of rug instead of a wall of lines across the bands
  { ctx.fillStyle = '#59c3a0'; let last = -1e9; for (const m of state.rebuildMarks) { const x = Math.round(ts.xOf(m.clock)); if (x - last < 4) continue; last = x; ctx.fillRect(x - 0.75, top - 8, 1.5, 5); } }
  ctx.fillStyle = '#9fb0c6'; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center';
  if (SRC.timeAxis) {   // calendar years, every 1, 2, 5 or 10 depending on the span
    const y0 = new Date(SRC.timeAxis.d0).getUTCFullYear(), y1 = new Date(SRC.timeAxis.d1).getUTCFullYear(), span = y1 - y0, step = span > 60 ? 10 : span > 15 ? 5 : span > 6 ? 2 : 1;
    for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) { const t = tOfDate(Date.UTC(y, 0)), x = ts.xOf(t); if (t > ts.tEnd || x < 8) continue; ctx.fillText(String(y), x, H - 4); ctx.fillRect(x, H - bot, 1, 4); }
  } else {
    const tickSec = ts.tEnd > 3 * 3600 ? 3600 : (ts.tEnd > 1800 ? 600 : 60);
    for (let t = 0; t <= ts.tEnd; t += tickSec) { const x = ts.xOf(t); if (x < 8) continue; ctx.fillText(mmss(t), x, H - 4); ctx.fillRect(x, H - bot, 1, 4); }
  }
  ctx.fillStyle = '#ffffff'; ctx.fillRect(ts.xOf(state.clock), top, 2, H - top - bot);
  if (view.on) { ctx.fillStyle = '#ffba46'; ctx.fillRect(ts.xOf(view.T) - 1, top, 3, H - top - bot); }
  if (state.rewindT != null) { const x = ts.xOf(state.rewindT); ctx.fillStyle = 'rgba(255,186,70,.16)'; ctx.fillRect(x, top, Math.max(0, ts.xOf(state.clock) - x), H - top - bot); ctx.fillStyle = '#ffba46'; ctx.fillRect(x - 1, top, 3, H - top - bot); }   // a rewind on offer: the rows after it would be played
}
function drawTimeline() {
  if (footerMode === 'lineage') { drawLineage(); return; }
  const c = $('timeline'), ctx = c.getContext('2d'), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight; c.width = W * dpr; c.height = H * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
  const ts = timeScale(W), { binSec, nb, bw, xOf } = ts, counts = new Int32Array(nb);
  for (const [b10, c] of state.arriveBins) counts[Math.min(nb - 1, Math.floor(b10 * 10 / binSec))] += c;
  const maxC = Math.max(1, ...counts), top = 22, bot = 18;
  for (let b = 0; b < nb; b++) { const h = (H - top - bot) * counts[b] / maxC, tb = b * binSec; ctx.fillStyle = tb < data.baseSeconds ? '#4a5a74' : (tb <= state.clock ? '#7d8da6' : '#2a3340'); ctx.fillRect(b * bw, H - bot - h, Math.max(1, bw - 0.5), h); }
  let strandedNote = ''; const curve = strandedCurve(binSec); if (curve.size) { ctx.strokeStyle = '#f0734a'; ctx.lineWidth = 1.5; ctx.beginPath(); let first = true; const maxF = Math.max(0.05, ...curve.values());
    for (const [b, f] of [...curve.entries()].sort((a, bb) => a[0] - bb[0])) { const x = b * bw + bw / 2, y = H - bot - (H - top - bot) * f / maxF; if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y); } ctx.stroke();
    strandedNote = ' · orange: stranded'; }
  // the stretch the rewind can show: from the oldest kept snapshot to the play-head
  if (snaps.length || live) { const lo = snaps.length ? snaps[0].clock : live.clock; ctx.fillStyle = 'rgba(255,186,70,0.35)'; ctx.fillRect(xOf(lo), top - 3, Math.max(1, xOf(state.clock) - xOf(lo)), 2); }
  drawCursors(ctx, ts, top, H, bot);
  ctx.textAlign = 'left'; ctx.fillStyle = '#9fb0c6'; ctx.fillText(`${SRC.noun} per ${SRC.timeAxis ? ((SRC.timeAxis.d1 - SRC.timeAxis.d0) / YEAR_MS > 6 ? 'year' : 'month') : binSec >= 60 ? binSec / 60 + ' min' : binSec + ' s'}${strandedNote}${state.rebuildMarks.length ? ' · green: rebuilds' : ''}`, 6, 12);
}
// ---- lineage view: one band per region identity over stream time, height = living posts (from the biography), colour = the
// region's tint; a new identity is stacked next to the region most of its base rows came from, so a split forks out of its
// parent's edge at the rebuild mark and a vanished identity narrows to nothing where it ended. Bands are drawn once per
// change of the play-head into an offscreen bitmap; cursors go on top every frame.
const lineage = { key: '', bmp: null, cols: null, order: null, W: 0, H: 0, top: 22, bot: 18 };
function lineageSeries() {
  const liveCount = new Map(regions ? regions.regions.map(r => [r.uid, r.count]) : []), series = [], byUid = new Map();
  for (const [uid, e] of regionHist) {
    if (!e.log.length) continue; const pts = e.log.map(x => [x.clock, x.count]), last = e.log[e.log.length - 1], lc = liveCount.get(uid);
    if (lc !== undefined) pts.push([state.clock, lc]); else if (e.ended) pts.push([e.ended.clock, pts[pts.length - 1][1]], [e.ended.clock, 0]);
    const sr = { uid, pts, color: last.color || '#8c99ad', name: last.name, born: e.born, ended: e.ended }; series.push(sr); byUid.set(uid, sr);
  }
  const order = []; series.sort((a, b) => a.uid - b.uid).forEach(sr => { const parent = sr.born && sr.born.from.length ? byUid.get(sr.born.from[0].uid) : null, i = parent ? order.indexOf(parent) : -1; if (i >= 0) order.splice(i + 1, 0, sr); else order.push(sr); });
  return order;
}
function lineageBitmap(W, H, fUid) {
  const dpr = window.devicePixelRatio || 1, ts = lineageScale(W), order = lineageSeries(), nS = order.length, top = lineage.top, bot = lineage.bot, Hb = H - top - bot;
  const w = new Float32Array(nS * W), total = new Float32Array(W), idx = new Int32Array(nS);
  for (let x = 0; x < W; x++) { const t = ts.tOf(x + 0.5); let tot = 0;
    for (let si = 0; si < nS; si++) { const pts = order[si].pts; let j = idx[si]; while (j + 1 < pts.length && pts[j + 1][0] <= t) j++; idx[si] = j;
      let v = 0; if (t >= pts[0][0] && t <= pts[pts.length - 1][0]) { const a = pts[j], b = pts[j + 1]; v = b && b[0] > a[0] ? a[1] + (b[1] - a[1]) * (t - a[0]) / (b[0] - a[0]) : a[1]; }
      w[si * W + x] = v; } }
  { const row = new Float32Array(W), R = 2; for (let si = 0; si < nS; si++) { const o = si * W; row.set(w.subarray(o, o + W)); let acc = 0; for (let x = 0; x < Math.min(W, R); x++) acc += row[x];   // a 5 px box along time: a rebuild's step becomes a short ramp
      for (let x = 0; x < W; x++) { if (x + R < W) acc += row[x + R]; if (x - R - 1 >= 0) acc -= row[x - R - 1]; const n = Math.min(W - 1, x + R) - Math.max(0, x - R) + 1; w[o + x] = row[x] > 0 || acc > 0 ? acc / n : 0; } } }
  for (let x = 0; x < W; x++) { let tot = 0; for (let si = 0; si < nS; si++) tot += w[si * W + x]; total[x] = tot; }
  let maxT = 1; for (let x = 0; x < W; x++) if (total[x] > maxT) maxT = total[x]; const sc = Hb / maxT;
  const cols = new Float32Array((nS + 1) * W);   // stacked tops per column, for drawing and hit tests: cols[si*W+x] = bottom edge of band si
  for (let x = 0; x < W; x++) { let y = top + (Hb - total[x] * sc) / 2; for (let si = 0; si < nS; si++) { cols[si * W + x] = y; y += w[si * W + x] * sc; } cols[nS * W + x] = y; }
  const bmp = document.createElement('canvas'); bmp.width = W * dpr; bmp.height = H * dpr; const ctx = bmp.getContext('2d'); ctx.scale(dpr, dpr);
  for (let si = 0; si < nS; si++) { const sr = order[si]; let x0 = -1, x1 = -1; for (let x = 0; x < W; x++) if (w[si * W + x] > 0) { if (x0 < 0) x0 = x; x1 = x; } if (x0 < 0) continue;
    x0 = Math.max(0, x0 - 1); x1 = Math.min(W - 1, x1 + 1);
    ctx.beginPath(); for (let x = x0; x <= x1; x++) { const y = cols[(si + 1) * W + x]; if (x === x0) ctx.moveTo(x, y); else ctx.lineTo(x, y); } for (let x = x1; x >= x0; x--) ctx.lineTo(x, cols[si * W + x]); ctx.closePath();
    ctx.fillStyle = sr.color; ctx.globalAlpha = 0.85; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = sr.uid === fUid ? '#ffffff' : 'rgba(11,14,19,0.7)'; ctx.lineWidth = sr.uid === fUid ? 1.5 : 0.6; ctx.stroke(); }
  return { bmp, cols, order, nS };
}
function drawLineage() {
  const c = $('lineage'), ctx = c.getContext('2d'), dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight; if (!W || !H) return;
  c.width = W * dpr; c.height = H * dpr; ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
  const fUid = focusUid(), key = `${W}|${H}|${state.clock}|${regionHist.size}|${state.rebuildMarks.length}|${fUid}|${engine.n}`;
  if (key !== lineage.key) { const t0 = performance.now(); Object.assign(lineage, lineageBitmap(W, H, fUid), { key, W, H }); lineage.ms = performance.now() - t0; }
  ctx.drawImage(lineage.bmp, 0, 0, W, H);
  const ts = lineageScale(W); drawCursors(ctx, ts, lineage.top, H, lineage.bot);
  ctx.textAlign = 'left'; ctx.fillStyle = '#9fb0c6'; ctx.font = '11px system-ui, sans-serif'; ctx.fillText(`${lineage.nS} regions over time${state.rebuildMarks.length ? ' · green: rebuilds' : ''}`, 6, 12);
}
function lineageHit(x, y, tol = 0) {   // the band under (x, y); with a tolerance, the nearest band edge within tol px when none is exactly under
  const L = lineage; if (!L.cols || x < 0 || x >= L.W) return null; const xi = Math.floor(x); let best = null, bd = tol + 1e-9;
  for (let si = 0; si < L.nS; si++) { const b = L.cols[si * L.W + xi], t = L.cols[(si + 1) * L.W + xi]; if (t - b <= 0.5) continue; if (y >= b && y <= t) return L.order[si]; const d = y < b ? b - y : y - t; if (d < bd) { bd = d; best = L.order[si]; } }
  return best;
}
// The loupe: a 3x magnified crop of the lineage around the pointer, drawn above the footer, with the band under the pointer
// named; it only shows, it never takes the pointer, so clicks and drags stay on the canvas (clicks use a 4 px tolerance).
const LOUPE = { w: 72, h: 36, mag: 3 };
function showLoupe(el, ev) {
  const lp = $('loupe'), r = el.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * el.clientWidth, y = (ev.clientY - r.top) / r.height * el.clientHeight;
  if (!lineage.bmp || x < 0 || x > el.clientWidth) { lp.hidden = true; return; }
  const dpr = window.devicePixelRatio || 1, W = LOUPE.w * LOUPE.mag, H = LOUPE.h * LOUPE.mag + 16; if (lp.width !== W * dpr || lp.height !== H * dpr) { lp.width = W * dpr; lp.height = H * dpr; lp.style.width = W + 'px'; lp.style.height = H + 'px'; }
  const ctx = lp.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = '#0b0e13'; ctx.fillRect(0, 0, W, H);
  const sx = Math.max(0, Math.min(lineage.W - LOUPE.w, x - LOUPE.w / 2)), sy = Math.max(0, Math.min(lineage.H - LOUPE.h, y - LOUPE.h / 2));
  ctx.imageSmoothingEnabled = false; ctx.drawImage(lineage.bmp, sx * dpr, sy * dpr, LOUPE.w * dpr, LOUPE.h * dpr, 0, 0, W, LOUPE.h * LOUPE.mag);
  const cx = (x - sx) * LOUPE.mag, cy = (y - sy) * LOUPE.mag; ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy); ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8); ctx.stroke();
  const hit = lineageHit(x, y, 4); ctx.fillStyle = '#e6edf6'; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.fillText(hit ? hit.name : '', 6, H - 4);
  const stage = $('map').parentElement.getBoundingClientRect(); lp.style.left = Math.max(4, Math.min(stage.width - W - 4, ev.clientX - stage.left - W / 2)) + 'px'; lp.hidden = false;
}

// Stream clock as displayed: advances smoothly between flushes at the playback rate (1x when live), never past the
// next flush, so arrival emphasis decays continuously instead of stepping with the flush interval.
function displayClock() { const rate = state.live ? 1 : state.speed, ahead = Math.max(0, (performance.now() - state.clockWall) / 1000) * rate; return state.clock + Math.min(state.interval, ahead); }
function setClock(t) { state.clock = t; state.clockWall = performance.now(); }
function drawFrame() { if (view.on) { const more = viewDraw(); state.frames++; return more; }
  const c = displayClock(); renderer.curTime = c; renderer.rate = state.live ? 1 : state.speed; renderer.tMin = data.baseSeconds; renderer.tMax = Math.max(loadedEnd(), data.baseSeconds + 1); renderer.flushStart = state.morphing ? 0 : (state.lastStats ? state.lastStats.start : data.nBase);
  // first row still inside the emphasis window by real time since it appeared (rows appear in order), so the flare passes only touch fresh arrivals
  let lo = data.nBase, hi = engine.n; const since = performance.now() / 1000 - renderer.recentWindow / renderer.rate, ap = renderer.appearS; while (lo < hi) { const mid = (lo + hi) >> 1; if (ap[mid] < since) lo = mid + 1; else hi = mid; }
  renderer.draw(engine.n, lo); state.frames++; return lo < engine.n; }
function render() { drawFrame(); scheduleChips(); }
// Frame loop: runs while playing and while any arrival is still inside its emphasis window or growing in.
function kickAnim() { if (state.animOn) return; state.animOn = true; const tick = () => { const fresh = drawFrame(); const growing = performance.now() / 1000 - renderer.lastAppear < 0.5; const moving = state.playing || state.live || (performance.now() - state.clockWall) / 1000 * (state.live ? 1 : state.speed) < state.interval; if ((fresh && moving) || growing || (view.on && fresh) || renderer.heatMoving) requestAnimationFrame(tick); else state.animOn = false; }; requestAnimationFrame(tick); }

function animateSettle() {
  if (state.morphing) return Promise.resolve();   // the rebuild's glide drives settle for every row
  if (!$('settleOn').checked || state.settleMs <= 0 || document.hidden) { renderer.settle = 1; render(); return Promise.resolve(); }   // a background tab gets no animation frames: placement must not wait for them
  return new Promise((resolve) => { const t0 = performance.now(); const tick = () => { const t = document.hidden ? 1 : Math.min(1, (performance.now() - t0) / state.settleMs); renderer.settle = 1 - (1 - t) * (1 - t) * (1 - t); render(); if (t < 1) frame(tick); else resolve(); }, frame = (f) => { let done = false; const go = () => { if (!done) { done = true; f(); } }; requestAnimationFrame(go); setTimeout(go, 50); }; frame(tick); });
}
// Place every row whose stream time is at or before the clock (in chunks of at most FLUSH_CAP).
async function flushPending() {
  state.flushBusy = true; try { return await flushPendingInner(); } finally { state.flushBusy = false; }
}
async function flushPendingInner() {
  while (state.tuneBusy) await state.tuneWait;   // the base is moving (tune mode): wait for the chunk being applied (milliseconds), not for the next flush, so arrivals keep their rhythm
  let w = engine.pendingUntil(state.clock), any = false;
  // appear times go in before the flush: the rows become drawable inside engine.flush, and a frame drawn before the
  // appear time landed would show them at full size for an instant and then restart the grow-in
  while (w > 0) { const c = Math.min(w, FLUSH_CAP); const since = state.clock - renderer.recentWindow; renderer.updateAppear(engine.n, c, (r) => !state.catchingUp && !state.quietPlace && data.tS[r] >= since); const tw = performance.now(), gap = state.lastFlushWall ? tw - state.lastFlushWall : 0; const st = await engine.flush(c); const wallMs = performance.now() - tw; state.lastFlushWall = performance.now(); if (st && (wallMs > 250 || gap > 1500)) logEvent('slow flush', { rows: c, wallMs: Math.round(wallMs), gpuMs: Math.round(st.msTotal), gapMs: Math.round(gap), rebuilding: state.rebuilding }); if (st) { snapTail(st.start, st.w); state.cumulativeMs += st.msTotal; state.lastStats = st; state.ledgerLog.push([st.tLast, engine.placedTotal, st.ledger.gross, st.ledger.net]); recomputeScores(st.start); recordPlacement(st); assignRegions(st.start, st.w); state.log.push(st); any = true; } w -= c; }
  if (any) { await retireSuperseded(); await enforceBudget(); autoLook(); regionRates(); const minute = Math.floor(state.clock / 60); if (minute !== state.histMinute) { state.histMinute = minute; recordHistory('minute'); } if (rstate.focus >= 0) showRegion(); }
  const due = !state.catchingUp && (state.rebuildRequested || (state.autoRebuild && playedNow() - state.lastRebuildPlayed >= state.rebuildMinutes * 60000) || (state.rebuildArrivals > 0 && engine.placedTotal - engine.placedAtRebuild >= state.rebuildArrivals));
  if (due && !state.rebuilding && !state.tuneMode && !cast?.on && engine.n > data.nBase) doRebuild(state.rebuildRequested ? 'requested' : 'cadence').catch(e => status('Rebuild failed: ' + (e.message || e), true));   // not awaited: arrivals keep flowing
  return any;
}
// Quiet rebuild (NodeGardener v3) and its morph: history leashed, arrivals free; then every dot glides from its old
// position to its new one while the region tints fade out, are recomputed from the new layout, and fade back in.
async function doRebuild(reason = 'requested') {
  if (state.rebuilding) return null; if (view.on) viewExit(); state.rebuilding = true; state.rebuildRequested = false; state.lastRebuildPlayed = playedNow(); const tStart = performance.now(); logEvent('rebuild start', { reason, rows: engine.n, k: state.rebuildMarks.length + 1 });
  try {
    status(`Rebuilding ${(engine.n - engine.deadPlaced).toLocaleString()} rows…`);
    const meanKept = () => { let sum = 0, c = 0; for (let p = 0; p < engine.n; p++) if (engine.alive[p]) { sum += scoreOf(p); c++; } return c ? sum / c * K : 0; };
    await engine.computeRanks(0, engine.n, engine.n, engine.knnIdx.subarray(0, engine.n * K)); const keptBefore = meanKept();
    // Hold every row at its current position for the whole rebuild: the morph mode reads positions from `start`, so the engine's
    // commit of the new layout into `pos` is invisible until the glide below moves settle from 0 to 1.
    engine.start.set(engine.pos.subarray(0, engine.n * 2)); engine.device.queue.writeBuffer(engine.startBuf, 0, engine.start.subarray(0, engine.n * 2));
    state.morphing = true; renderer.settle = 0; drawFrame();
    const preAnch = engine.anchored.slice(0, engine.n), preReg = regionS ? regionS.slice(0, engine.n) : null, preRegions = regions;
    const st = await engine.quietRebuild({ lambda: state.leash }); try { state.lastMovers = await rebuildMovers(preAnch, preReg, preRegions); } catch (e) { console.warn('[jet] mover report failed', e); } snapClose(); snapOpen('rebuild'); const keptAfter = meanKept(); logEvent('rebuild committed', { k: state.rebuildMarks.length + 1, msPrep: Math.round(st.msPrep), msTotal: Math.round(st.msTotal), histP95: +st.histP95.toFixed(2), arrMedian: +st.arrMedian.toFixed(2) });
    // re-cluster on the new layout now (rows are still shown at their old positions), holding the old membership (rmix 0)
    crossfadeToken++; await buildRegions({ crossfade: false }); recomputeScores(0);
    // one glide: positions, tint field, membership blend and chip positions all follow the same eased clock
    await glide(Math.max(0, state.morphMs));
    const mark = { clock: state.clock, reason, keptBefore, keptAfter, ...st }; state.rebuildMarks.push(mark); state.lastRebuildClock = state.clock; logEvent('rebuild morph done', { k: state.rebuildMarks.length, wallMs: Math.round(performance.now() - tStart) });
    updateReadouts(state.lastStats); render(); drawTimeline(); legend();
    status(`Rebuilt in ${(st.msTotal / 1000).toFixed(1)} s · local fit ${keptBefore.toFixed(1)} → ${keptAfter.toFixed(1)} of 15`);
    return mark;
  } catch (e) { logEvent('rebuild failed', { message: String(e && e.message || e), stack: String(e && e.stack || '').slice(0, 400) }); console.error('[jet] rebuild failed', e); throw e; }
  finally { state.rebuilding = false; state.morphing = false; renderer.settle = 1; renderer.rmix = 1; }
}
// The glide: every row from `start` to `pos`, the membership blend and the chip positions follow one eased clock over T ms.
// Next frame, or 50 ms if animation frames are not being delivered (hidden or throttled tab), so a morph can never stall.
async function glide(T, each = null) {
  const t0 = performance.now(), frame = () => new Promise(r => { let done = false; const go = () => { if (!done) { done = true; r(); } }; requestAnimationFrame(go); setTimeout(go, 50); });
  while (T > 0) { const t = Math.min(1, (performance.now() - t0) / T), e = t * t * (3 - 2 * t); renderer.settle = e; renderer.rmix = e; if (each) each(e); drawFrame(); scheduleChips(); if (t >= 1) break; await frame(); }
  if (each) each(1);
  renderer.settle = 1; renderer.rmix = 1; state.morphing = false; if (regionPrevS) { regionPrevS.set(regionS); renderer.updateRegionIds(regionS, 0, engine.n, regionPrevS); } drawFrame(); scheduleChips();
}
async function requestRebuild() { if (view.on) viewExit(); if (state.flushBusy || state.playing) { state.rebuildRequested = true; status('Rebuild queued'); if (!state.playing) { await flushPending(); } } else await doRebuild('requested'); }
// Capped history: when living arrivals exceed the budget, retire the excess from among the oldest living arrivals
// (a pool of four times the excess, older than the emphasis window), densest first by 15th-neighbour radius in feature
// space, so dense cores thin out before sparse rows. Retired rows leave the search, their neighbours' lists are
// refreshed exactly, and the row arrays are compacted once enough rows are dead.
// The budget in force: the History budget control, or, live with no budget set, the largest budget the reserved row
// slots can hold (living rows plus the dead rows allowed before compaction plus two flushes of headroom must fit), so
// a long live session retires instead of running out of slots. Replay with no budget stays unlimited (hash-neutral).
function effectiveBudget() {
  if (state.budget || !CAP || SRC.static) return state.budget;   // a static dataset keeps every row
  const slots = data.N - data.nBase - 2 * FLUSH_CAP;   // slots shared by the capture's streamed rows and live arrivals
  return Math.max(1000, Math.floor(Math.min(slots - 5000, slots / 1.1)));
}
async function enforceBudget() {
  if (state.rebuilding) return;   // row indices must stay fixed while a rebuild is in flight; the next flush catches up
  const budget = effectiveBudget(); if (!budget) return;
  const living = engine.n - data.nBase - engine.deadPlaced, excess = living - budget; if (excess <= 0) return;
  const since = state.clock - renderer.recentWindow, pool = [];
  for (let p = data.nBase; p < engine.n && pool.length < 4 * excess; p++) if (engine.alive[p] && data.tS[p] < since) pool.push(p);
  if (!pool.length) return;
  pool.sort((a, b) => (engine.knnDist[a * K + K - 1] - engine.knnDist[b * K + K - 1]) || (a - b));
  const victims = pool.slice(0, Math.min(excess, pool.length)), t0 = performance.now();
  const res = engine.retire(victims); renderer.markRetired(victims);
  for (const p of victims) { const r = regionS[p]; if (r >= 0 && regions) regions.regions[r].count--; regionS[p] = -1; }
  const refreshed = await engine.refreshLists(res.affected); const t1 = performance.now();
  let compacted = false; if (engine.deadPlaced > Math.max(5000, 0.1 * budget)) { compactAll(); compacted = true; }
  state.lastWave = { retired: victims.length, refreshed, ms: t1 - t0, compactMs: performance.now() - t1, compacted }; logEvent('retire', { retired: victims.length, refreshed, ms: Math.round(t1 - t0), compacted, compactMs: Math.round(performance.now() - t1) });
}
// One dot per aircraft (numeric sources whose rows carry an icao24): a new sample of an aircraft retires its previous streamed
// dot. Base rows are never retired (they are the fitted map the regions are drawn from); the base map itself holds one sample
// per aircraft, its latest in the base window (freshBuild). Waits while a rebuild holds the row indices; the next flush catches up.
async function retireSuperseded() {
  if (!SRC.numeric || state.rebuilding) return;
  const seen = new Set(), victims = [];
  for (let p = engine.n - 1; p >= data.nBase; p--) { if (!engine.alive[p]) continue; const id = data.posts[p] && data.posts[p].icao24; if (!id) continue; if (seen.has(id)) victims.push(p); else seen.add(id); }
  if (!victims.length) return;
  const t0 = performance.now(), res = engine.retire(victims); renderer.markRetired(victims);
  for (const p of victims) { const r = regionS[p]; if (r >= 0 && regions) regions.regions[r].count--; regionS[p] = -1; }
  const refreshed = await engine.refreshLists(res.affected); const t1 = performance.now();
  let compacted = false; if (engine.deadPlaced > 5000) { compactAll(); compacted = true; }
  state.lastWave = { retired: victims.length, refreshed, ms: t1 - t0, compactMs: performance.now() - t1, compacted };
}
function compactAll() {
  const map = engine.compact(); if (!map) return;
  const m = data.n, sc = new Float32Array(data.N), rs = new Int16Array(data.N).fill(-1);
  for (let i = 0; i < map.length; i++) { const j = map[i]; if (j >= 0) { sc[j] = scores[i]; rs[j] = regionS[i]; } }
  scores = sc; regionS = rs; if (state.lastStats) { const j = map[state.lastStats.start]; state.lastStats.start = j >= 0 ? j : engine.n; }
  renderer.remap(map, m, regionS, scores); snapClear(); snapOpen('compacted'); if (state.inspect !== null) clearInspect();
}
async function stepOnce() {
  if (view.on) viewExit(); const end = loadedEnd();
  if (!state.live && state.clock >= end && engine.n >= data.n) { setPlaying(false); status(`Replay complete: ${engine.placedTotal.toLocaleString()} placed`); return false; }
  setClock(state.live ? end : Math.min(end, state.clock + state.interval));
  await flushPending(); updateReadouts(state.lastStats); kickAnim(); await animateSettle(); drawTimeline(); return true;
}
async function playLoop() {
  let tPrev = performance.now(), cPrev = state.clock;
  while (state.playing) {
    const t = performance.now(); if (!(await stepOnce())) break;
    const want = state.live ? 1000 * state.interval : 1000 * state.interval / state.speed, wait = Math.max(0, want - (performance.now() - t));
    if (wait) await new Promise(r => setTimeout(r, wait));
    const now = performance.now(); if (now - tPrev > 1000) { state.achieved = (state.clock - cPrev) / ((now - tPrev) / 1000); tPrev = now; cPrev = state.clock; }
  }
}
async function runTo(T) {
  if (view.on) viewExit(); if (T < state.clock) { status('Already placed; Reset to replay'); return; }
  if (state.playing) setPlaying(false); renderer.settle = 1; let i = 0; const end = loadedEnd();
  while (state.clock < Math.min(T, end)) { setClock(Math.min(end, state.clock + state.interval)); await flushPending(); if (++i % 10 === 0) { updateReadouts(state.lastStats); render(); drawTimeline(); await new Promise(r => setTimeout(r, 0)); } }
  updateReadouts(state.lastStats); render(); drawTimeline();
  if (!state.live && state.clock >= end && engine.n >= data.n) status(`Replay complete: ${engine.placedTotal.toLocaleString()} placed`);
}

// ---- soft regions: computed once from the frozen base map; arrivals acquire membership by landing inside.
// Membership crossfade after regions are recomputed: tints and region-coloured dots blend from the old regions to the new
// ones over 1.5 s, so a region that split off fades in and a recoloured row crosses over instead of flipping.
// Each new region remembers the old region most of its base rows came from (if that is at least half of them), so its chip
// can slide from the old centroid to the new one while the membership blends.
// A new region keeps an old region's identity (uid, chip element, chip position, counts) only when the match is mutual:
// most of the new region's base rows came from that old region AND most of the old region's base rows went to the new
// one. A region that split off or swallowed others is a new identity: its chip appears in place, the old chip fades out.
let uidCounter = 0;
function matchPreviousRegions() {
  if (!regions) return; const R = regions.regions; for (const r of R) { r.prev = null; r.uid = 0; }
  if (prevRegions && regionPrevS) {
    const overlap = new Map();
    for (let p = 0; p < data.nBase; p++) { const n = regionS[p], o = regionPrevS[p]; if (n < 0 || o < 0) continue; const key = n * 65536 + o; overlap.set(key, (overlap.get(key) || 0) + 1); }
    const bestNew = new Map(), bestOld = new Map();
    for (const [key, c] of overlap) { const n = Math.floor(key / 65536), o = key % 65536; if (!bestNew.has(n) || c > bestNew.get(n).c) bestNew.set(n, { o, c }); if (!bestOld.has(o) || c > bestOld.get(o).c) bestOld.set(o, { n, c }); }
    for (const [n, b] of bestNew) { const r = R[n], o = prevRegions.regions[b.o], back = bestOld.get(b.o); if (!r || !o || !back || back.n !== n) continue;
      if (b.c >= 0.5 * r.countBase && b.c >= 0.5 * o.countBase) { r.uid = o.uid; r.prev = { cx: o.cx, cy: o.cy, count: o.count, arr: o.arr }; } }
    // lineage: a new identity records which old regions its base rows came from; a vanished identity records where its rows went
    const fromOf = new Map(), intoOf = new Map();
    for (const [key, c] of overlap) { const n = Math.floor(key / 65536), o = key % 65536, r = R[n], po = prevRegions.regions[o]; if (!r || !po) continue;
      if (c >= 0.1 * r.countBase) { if (!fromOf.has(n)) fromOf.set(n, []); fromOf.get(n).push({ uid: po.uid, name: po.name, share: c / r.countBase }); }
      if (c >= 0.1 * po.countBase) { if (!intoOf.has(o)) intoOf.set(o, []); intoOf.get(o).push({ n, share: c / po.countBase }); } }
    const kept = new Set(R.filter(r => r.uid).map(r => r.uid));
    for (const r of R) if (!r.uid) { r.uid = ++uidCounter; histOf(r.uid).born = { clock: state.clock, from: (fromOf.get(r.k) || []).sort((a, b) => b.share - a.share) }; }
    prevRegions.regions.forEach((po, o) => { if (kept.has(po.uid)) return; const e = regionHist.get(po.uid); if (e && !e.ended) e.ended = { clock: state.clock, into: (intoOf.get(o) || []).map(x => ({ uid: R[x.n].uid, name: R[x.n].name, share: x.share })).sort((a, b) => b.share - a.share) }; });
  }
  for (const r of R) if (!r.uid) r.uid = ++uidCounter;
}
let crossfadeToken = 0;
function crossfadeRegions(ms = 1500) {
  const token = ++crossfadeToken, t0 = performance.now(); renderer.rmix = 0;
  const frame = () => new Promise(r => { let done = false; const go = () => { if (!done) { done = true; r(); } }; requestAnimationFrame(go); setTimeout(go, 50); });
  (async () => { while (true) { const t = Math.min(1, (performance.now() - t0) / ms); if (token !== crossfadeToken) return; renderer.rmix = t * t * (3 - 2 * t); drawFrame(); scheduleChips(); if (t >= 1) break; await frame(); }
    renderer.rmix = 1; if (regionPrevS && regionS) { regionPrevS.set(regionS); renderer.updateRegionIds(regionS, 0, engine.n, regionPrevS); } scheduleChips(); })();
}
// The fade-in played backwards (entering tune mode): each base row's region becomes its previous one and its new one none,
// so the tint splats lose weight over the same eased 1.5 s and every region contracts into its dots; the chips fade with
// it. Arrivals take no region meanwhile; at the end the regions are cleared (buildRegions does nothing in tune mode).
function shrinkRegions(ms = 1500) {
  if (!regions || !regionS) { buildRegions(); render(); return; }
  const token = ++crossfadeToken, t0 = performance.now(), chips = $('chips'); state.regionsLeaving = true;
  renderer.setRegions(regions, tintPx(), rstate.soft, regions);
  regionPrevS = regionS.slice(); renderer.updateRegionIds(new Int16Array(regionS.length).fill(-1), 0, engine.n, regionPrevS); renderer.rmix = 0;
  chips.style.transition = `opacity ${ms}ms ease`; chips.style.opacity = '0';
  const frame = () => new Promise(r => { let done = false; const go = () => { if (!done) { done = true; r(); } }; requestAnimationFrame(go); setTimeout(go, 50); });
  (async () => {
    while (token === crossfadeToken) { const t = Math.min(1, (performance.now() - t0) / ms); renderer.rmix = t * t * (3 - 2 * t); drawFrame(); if (t >= 1) break; await frame(); }
    state.regionsLeaving = false; chips.style.transition = ''; chips.style.opacity = '';
    if (token === crossfadeToken) { renderer.rmix = 1; buildRegions(); render(); }
  })();
}
// Diagnostic only (changes nothing): after a quiet rebuild (NodeGardener), the dots that moved furthest, in neighbourhood
// widths, with where their 15 feature-space neighbours sat before the rebuild. `start` still holds the old positions here,
// and regions are the pre-rebuild ones, so old and new positions are judged on the same map. About 0.3 s of CPU at 182,000
// rows: it works on copies taken at the start and gives the page its turn every 8 ms (the same report as in one go).
async function rebuildMovers(preAnch, preReg, R, top = 20) {
  // one column per figure (typed arrays) instead of an object and a Map per row: at 182,000 rows those allocations cost the
  // page three garbage-collection pauses of about 60 ms
  const n = engine.n, pos = engine.pos.slice(0, n * 2), old = engine.start.slice(0, n * 2), idx = engine.knnIdx.slice(0, n * K), alive = engine.alive.slice(0, n), w = engine.widthUnit();
  const P = new Int32Array(n), M = new Float64Array(n), ARR = new Uint8Array(n), BASE = new Uint8Array(n), FROM = new Int32Array(n), TO = new Int32Array(n), NBR = new Int32Array(n), MAJ = new Int32Array(n), MS = new Float64Array(n), GAP = new Float64Array(n), SPR = new Float64Array(n);
  const rs = new Int32Array(K), dr = new Int32Array(K), dc = new Int32Array(K); let cnt = 0;
  let t = performance.now(); const turn = async () => { if (performance.now() - t > 8) { await new Promise((r) => setTimeout(r, 0)); t = performance.now(); } };
  for (let p = 0; p < n; p++) {
    if ((p & 1023) === 1023) await turn();
    if (!alive[p]) continue;
    const m = Math.hypot(pos[2 * p] - old[2 * p], pos[2 * p + 1] - old[2 * p + 1]) / w;
    let gap = Infinity, cx = 0, cy = 0, c = 0;
    for (let k = 0; k < K; k++) { const j = idx[p * K + k]; if (j < 0 || j >= n || !alive[j]) continue; rs[c] = preReg ? preReg[j] : -1; gap = Math.min(gap, Math.hypot(old[2 * j] - old[2 * p], old[2 * j + 1] - old[2 * p + 1]) / w); cx += old[2 * j]; cy += old[2 * j + 1]; c++; }
    let spread = 0; if (c) { cx /= c; cy /= c; for (let k = 0; k < K; k++) { const j = idx[p * K + k]; if (j < 0 || j >= n || !alive[j]) continue; spread += Math.hypot(old[2 * j] - cx, old[2 * j + 1] - cy) / w; } spread /= c; }
    // neighbours' regions, distinct in order of first appearance, counted; the majority is the first with the most
    let nd = 0; for (let a = 0; a < c; a++) { let f = -1; for (let b = 0; b < nd; b++) if (dr[b] === rs[a]) { f = b; break; } if (f < 0) { dr[nd] = rs[a]; dc[nd] = 1; nd++; } else dc[f]++; }
    let maj = -1, majC = 0; for (let b = 0; b < nd; b++) if (dc[b] > majC) { majC = dc[b]; maj = dr[b]; }
    P[cnt] = p; M[cnt] = m; ARR[cnt] = preAnch[p] ? 0 : 1; BASE[cnt] = p < data.nBase ? 1 : 0; FROM[cnt] = preReg ? preReg[p] : -1; TO[cnt] = R ? R.regionOf(pos[2 * p], pos[2 * p + 1]) : -1; NBR[cnt] = nd; MAJ[cnt] = maj; MS[cnt] = c ? majC / c : 0; GAP[cnt] = gap; SPR[cnt] = spread; cnt++;
  }
  // only the top movers need ordering: the cut-off from a numeric sort of the moves, then the rows at or above it, in the
  // order a stable sort of every row by move would give them (ties stay in row order)
  const headN = Math.min(cnt, Math.max(top, Math.ceil(cnt * 0.01))); await turn();
  const sorted = M.slice(0, cnt).sort(), cut = headN ? sorted[cnt - headN] : Infinity; await turn();
  const above = [], at = []; for (let r = 0; r < cnt; r++) { if (M[r] > cut) above.push(r); else if (M[r] === cut) at.push(r); } above.sort((a, b) => M[b] - M[a]);
  const ranked = above.concat(at.slice(0, headN - above.length)); await turn();
  const name = (r) => (r >= 0 && R && R.regions[r] ? (R.regions[r].name || '#' + r) : '(gap)');
  const text = (p) => { const t = data.posts[p] && (data.posts[p].text || ''); return t.replace(/\s+/g, ' ').slice(0, 70); };
  const table = ranked.slice(0, top).map(r => ({ row: P[r], moved: +M[r].toFixed(1), kind: BASE[r] ? 'base' : ARR[r] ? 'arrival' : 'older', from: name(FROM[r]), to: name(TO[r]), 'nbr regions': NBR[r], 'nbr majority': name(MAJ[r]) + ' ' + Math.round(MS[r] * 100) + '%', 'gap to nearest nbr': +GAP[r].toFixed(1), 'nbr spread': +SPR[r].toFixed(1), text: text(P[r]) }));
  // a group is a list of entries (null = every entry); medians from a numeric sort, shares as before
  const len = (g) => (g ? g.length : cnt), val = (g, col) => (g ? Float64Array.from(g, (r) => col[r]) : Float64Array.from(col.subarray(0, cnt)));
  const q = (g, col) => { if (!len(g)) return 0; const v = val(g, col).sort(); return v[Math.floor(v.length / 2)]; };
  const share = (g, f) => { const L = len(g); if (!L) return 0; let c = 0; if (g) { for (const r of g) if (f(r)) c++; } else for (let r = 0; r < cnt; r++) if (f(r)) c++; return Math.round(100 * c / L); };
  const groups = { 'top 1% movers': ranked, 'all dots': null };
  const summary = {}; for (const [k, g] of Object.entries(groups)) {   // one figure per turn: each is a pass over every row
    const figs = [['median move', () => +q(g, M).toFixed(2)], ['% arrivals', () => share(g, r => ARR[r])], ['% base', () => share(g, r => BASE[r])], ['% split nbrs (majority < 70%)', () => share(g, r => MS[r] < 0.7)], ['% started in a gap', () => share(g, r => FROM[r] < 0)], ['% moved to nbr majority', () => share(g, r => TO[r] === MAJ[r] && TO[r] !== FROM[r])], ['median gap to nearest nbr', () => +q(g, GAP).toFixed(2)], ['median nbr spread', () => +q(g, SPR).toFixed(2)]];
    summary[k] = { dots: len(g) }; for (const [f, fn] of figs) { await turn(); summary[k][f] = fn(); }
  }
  console.info('[jet] rebuild movers, in neighbourhood widths'); console.table(summary); console.table(table);
  return { summary, table, width: w };
}
// The regions are computed in slices of about 8 ms (regions.js runSliced) from the base positions as they are when the
// call starts, with the old regions on screen meanwhile; then they are swapped in at once. A newer call supersedes one
// still computing. Returns a promise for callers that animate the result (a quiet rebuild waits for it before its glide).
let regionBuild = 0;
async function buildRegions({ crossfade = true } = {}) {
  const token = ++regionBuild;
  if (!rstate.on || state.tuneMode) { /* no regions while the base map is being tuned: they come back, recomputed, with Done tuning */ regions = null; renderer.setRegions(null); rstate.focus = -1; renderer.focusRegion = -1; state.rates = null; renderer.setHeat(null); showRegion(); layoutChips(); $('r-regions').textContent = 'off'; return; }
  if (!data.docs) { const docs = await runSliced(docsSteps(data.posts.slice(0, data.nBase))); if (!data.docs) data.docs = docs; }   // the base posts' words and word pairs, tokenised once
  const basePos = engine.pos.slice(0, data.nBase * 2);
  const next = await runSliced(computeRegionsSteps(basePos, data.nBase, { sigma: rstate.sigma, tintSigma: tintPx(), softSigma: rstate.soft, tint: rstate.tint, mergeTau: rstate.tau, posts: data.posts, docs: data.docs, feat: data.featS, dim: data.dim, labels: data.labels, labelS: data.labelS, labelColors: data.labelColors }));
  if (token !== regionBuild || !rstate.on || state.tuneMode) return;   // superseded, or regions were switched off or tuning began meanwhile
  rstate.focus = -1; renderer.focusRegion = -1;
  prevRegions = regions; const oldS = regionS ? regionS.slice() : null; if (regionS && prevRegions) { regionPrevS = regionS.slice(); } else { regionPrevS = new Int16Array(data.N).fill(-1); }   // no regions on screen (off, or tuning): nothing to fade from; stale ids would index the new palette (black blots)
  crossfadeToken++; renderer.rmix = 0;   // hold the old membership until the blend starts; assignRegions must not overwrite the previous-region slots
  renderer.rotateHeat();   // the current heat becomes the previous regions' heat for the morph
  regions = next;
  renderer.setRegions(regions, tintPx(), rstate.soft, prevRegions); if (!regionS) regionS = new Int16Array(data.N); regionS.fill(-1);
  assignRegions(0, engine.n); matchPreviousRegions(); regionRates(); recordHistory(prevRegions ? 'rebuild' : 'base'); showRegion(); if (crossfade) crossfadeRegions(); else scheduleChips();
  // what the re-clustering did to the posts: same region (by identity), moved to another, now outside, newly inside
  let moved = 0, left = 0, joined = 0, kept = 0;
  if (prevRegions && oldS) { const PR = prevRegions.regions, NR = regions.regions; for (let p = 0; p < engine.n; p++) { if (p >= data.nBase && !engine.alive[p]) continue; const o = oldS[p] >= 0 && oldS[p] < PR.length ? PR[oldS[p]].uid : 0, n = regionS[p] >= 0 && regionS[p] < NR.length ? NR[regionS[p]].uid : 0; if (o === n) { if (n) kept++; } else if (!n) left++; else if (!o) joined++; else moved++; } }
  const same = regions.regions.filter(r => r.prev).length;
  $('r-regions').textContent = `${regions.regions.length} regions from the base map (grain ${rstate.sigma} px on a ${regions.g} grid, merge ${rstate.tau}, regions with fewer than ${regions.minPoints} base posts dropped) · ${regions.ms.toFixed(0)} ms` + (prevRegions ? ` · ${same} regions kept their identity, ${regions.regions.length - same} are new; posts: ${kept.toLocaleString()} stayed, ${moved.toLocaleString()} moved to another region, ${left.toLocaleString()} now outside every region, ${joined.toLocaleString()} newly inside one` : '');
}
function assignRegions(start, count) {
  if (!regions || state.regionsLeaving) return; const R = regions.regions, cf = state.countFrom;
  for (let p = start; p < start + count; p++) {
    const old = regionS[p], isArr = p >= data.nBase, isNew = isArr && data.tS[p] > cf;
    if (old >= 0 && isArr) { R[old].count--; if (isNew) R[old].arr--; }
    if (isArr && !engine.alive[p]) { regionS[p] = -1; if (regionPrevS) regionPrevS[p] = -1; continue; }   // retired rows belong to no region
    const r = regions.regionOf(engine.pos[2 * p], engine.pos[2 * p + 1]); regionS[p] = r; if (regionPrevS && (isArr || renderer.rmix >= 1)) regionPrevS[p] = r;
    if (r >= 0 && isArr) { R[r].count++; if (isNew) R[r].arr++; }
  }
  renderer.updateRegionIds(regionS, start, count, regionPrevS);
}
function setFocus(k) {
  rstate.focus = (k === rstate.focus) ? -1 : k; renderer.focusRegion = rstate.focus;
  showRegion(); render();
}
// ---- per-region arrival rates, read from the rows themselves (stream time and current membership), so they follow a
// rebuild's re-assignment and never double count: arrivals in the last RATE_WIN s per region, their share of all arrivals
// against the region's share of the base map, and the heat each region shows (volume: rate over the busiest region;
// growth: log2 of the share ratio, full at 4x the base share).
const RATE_WIN = 60, SPARK_WIN = 600, SPARK_BIN = 10;
// Automatic settings, one pill per control (Look group): the emphasis window and the heat window follow the arrival rate,
// the clustering grain is chosen once per base map by a sweep, and the tint's edge softness and merge saddle follow the
// grain. A control with its pill on is disabled and shows the chosen value; switching the pill off frees it where it is.
// Rules, judgment calls fitted to the investigator's own picks: emphasis = 15 s at Bluesky's 8 arrivals a second, growing
// as rate^-0.75 (200 s at Hacker News's 0.24 a second), 5 to 600 s; heat window = about 150 arrivals (150 / rate, 60 s to
// 30 min); grain = the width (2 to 12 px) that maximises chips x coverage, chips counted up to 50 (about what a screen can
// show), which reproduces 4 px on a 15,000-post Bluesky base and gives 10 px on a 1,150-item Hacker News base. The tint
// is painted no wider than 5 px whatever the clustering grain, so sparse maps keep their membership without blobs that
// balloon around lone points; softness = 6 px at 15,000 base posts, shrinking with the cube root of the count (2 to 8 px), so
// a sparse map is not blurred into one cloud. The rate is arrivals per stream second over the last ten
// minutes placed, or the base map's own rate before that; a value changes only when it moves by more than 20 %.
const look = { auto: { recent: true, sigma: true, soft: true, tau: true, heatWin: true }, grainPick: null };
const TINT_MAX_PX = 5, CHIP_CAP = 50;
const tintPx = () => Math.min(rstate.sigma, TINT_MAX_PX);
function arrivalRate() {
  const span = Math.min(600, Math.max(1, state.clock - data.baseSeconds)); let c = 0;
  if (span >= 60) for (let p = firstRowAfter(state.clock - span); p < engine.n; p++) { if (data.tS[p] > state.clock) break; c++; }
  return c >= 20 ? c / span : data.nBase / Math.max(1, data.baseSeconds);
}
const clampN = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
// The grain sweep: regions of the base map at each width (no naming), scored by min(chips, CHIP_CAP) x coverage of base rows.
function chooseGrain() {
  const t0 = performance.now(), table = [];
  for (const sigma of [2, 3, 4, 5, 6, 8, 10, 12]) { const R = computeRegions(engine.pos, data.nBase, { sigma, softSigma: 6, mergeTau: 0.1 }); const chips = R.regions.length, cov = R.regions.reduce((a, r) => a + r.countBase, 0) / Math.max(1, data.nBase); table.push({ sigma, chips, cov: +cov.toFixed(3), score: +(Math.min(chips, CHIP_CAP) * cov).toFixed(2) }); }
  let best = table[0]; for (const t of table) if (t.score > best.score) best = t;
  look.grainPick = { sigma: best.sigma, table, ms: Math.round(performance.now() - t0) }; logEvent('grain sweep', { pick: best.sigma, ms: look.grainPick.ms, table: table.map(t => `${t.sigma}:${t.chips}/${t.cov}`).join(' ') });
  return best.sigma;
}
function autoLook({ grain = false } = {}) {
  if (!data) return; const A = look.auto, rate = arrivalRate();
  if (A.recent) { const recent = clampN(5 * Math.round(15 * Math.pow(8 / Math.max(rate, 1e-3), 0.75) / 5), 5, 600); if (Math.abs(recent - renderer.recentWindow) / Math.max(recent, renderer.recentWindow) > 0.2) { renderer.recentWindow = recent; $('recent').value = recent; $('recentv').textContent = recent; } }
  if (A.heatWin) { const win = clampN(30 * Math.round(150 / rate / 30), 60, 1800); if (Math.abs(win - state.rateWin) / Math.max(win, state.rateWin) > 0.2) { state.rateWin = win; $('heatWin').value = win; $('heatWinv').textContent = win; } }
  let rebuild = false;
  if (A.sigma && (grain || !look.grainPick)) { const sigma = look.grainPick && !grain ? look.grainPick.sigma : chooseGrain(); if (sigma !== rstate.sigma) { rstate.sigma = sigma; $('regionSigma').value = sigma; $('regionSigmav').textContent = sigma; rebuild = true; } }
  if (A.soft) { const soft = clampN(Math.round(6 * Math.cbrt(Math.max(1, data.nBase) / 15000)), 2, 8); if (soft !== rstate.soft) { rstate.soft = soft; $('regionSoft').value = soft; $('regionSoftv').textContent = soft; rebuild = true; } }
  if (A.tau && rstate.tau !== 0.1) { rstate.tau = 0.1; $('regionTau').value = 0.1; $('regionTauv').textContent = '0.1'; rebuild = true; }
  if (rebuild && regions && !state.rebuilding) buildRegions();
}
const PILL_CTL = { recent: ['recent'], sigma: ['regionSigma'], soft: ['regionSoft'], tau: ['regionTau'], heatWin: ['heatWin'] };
function setAuto(key, on) {
  look.auto[key] = on; for (const id of PILL_CTL[key]) { $(id).disabled = on; $(id).closest('.ctl').classList.toggle('auto-set', on); } document.querySelector(`.pill[data-auto="${key}"]`).classList.toggle('on', on);
  try { localStorage.setItem('jet.auto', JSON.stringify(look.auto)); } catch {} if (on) autoLook();
}
// The heat ramp as CSS, from the renderer's own stops.
function heatCss(h) { return `rgb(${heatRgb(h).map(v => Math.round(255 * v)).join(',')})`; }
function firstRowAfter(t) { let lo = data.nBase, hi = engine.n; while (lo < hi) { const mid = (lo + hi) >> 1; if (data.tS[mid] <= t) lo = mid + 1; else hi = mid; } return lo; }
function regionRates() {
  if (!regions) { state.rates = null; renderer.setHeat(null); return null; }
  const R = regions.regions, clock = state.clock, cnt = new Float32Array(R.length), h = new Float32Array(R.length), win = state.rateWin || RATE_WIN; let total = 0;
  for (let p = firstRowAfter(clock - win); p < engine.n; p++) { if (data.tS[p] > clock) break; const r = regionS[p]; if (r >= 0) { cnt[r]++; total++; } }
  const ratio = new Float32Array(R.length); let maxC = 0; for (let k = 0; k < R.length; k++) { if (cnt[k] > maxC) maxC = cnt[k]; ratio[k] = total >= 20 && R[k].countBase ? (cnt[k] / total) / (R[k].countBase / data.nBase) : 0; }
  if (state.heat === 'volume') { for (let k = 0; k < R.length; k++) h[k] = maxC ? cnt[k] / maxC : 0; }
  else if (state.heat === 'growth') { let mx = 0; for (let k = 0; k < R.length; k++) { h[k] = ratio[k] > 1 ? Math.log2(ratio[k]) : 0; if (h[k] > mx) mx = h[k]; } if (mx > 0) for (let k = 0; k < R.length; k++) h[k] /= mx; }   // the fastest-growing region is 1, the rest in proportion (log scale)
  // smoothing: each region identity's shown heat follows its target with a time constant of a quarter of the window (stream time)
  const ema = state.heatEma || (state.heatEma = new Map()), dt = state.rates ? Math.max(0, clock - state.rates.clock) : 0, a = 1 - Math.exp(-dt / Math.max(1, win / 4));
  if (state.heat !== 'off') for (let k = 0; k < R.length; k++) { const uid = R[k].uid, prev = ema.get(uid); const v = prev === undefined ? h[k] : prev + (h[k] - prev) * a; ema.set(uid, v); h[k] = v; }
  if (ema.size > 4 * R.length + 64) { const live = new Set(R.map(r => r.uid)); for (const uid of [...ema.keys()]) if (!live.has(uid)) ema.delete(uid); }
  state.rates = { cnt, total, ratio, h, clock }; renderer.setHeat(state.heat === 'off' ? null : h); return state.rates;
}
// Arrivals per SPARK_BIN s of stream time over the last SPARK_WIN s for one region, newest bin last.
function regionSpark(k) {
  const clock = state.clock, nb = SPARK_WIN / SPARK_BIN, bins = new Float32Array(nb), t0 = clock - SPARK_WIN;
  for (let p = firstRowAfter(t0); p < engine.n; p++) { const t = data.tS[p]; if (t > clock) break; if (regionS[p] === k) bins[Math.min(nb - 1, Math.floor((t - t0) / SPARK_BIN))]++; }
  return bins;
}
// ---- region biography: per region identity (uid), one entry per minute of stream time and per rebuild (count, arrivals,
// rate, name, centre), plus where it came from (born) and where its posts went (ended), from the mutual-overlap matching.
const regionHist = new Map();
const histOf = (uid) => { let e = regionHist.get(uid); if (!e) { e = { uid, log: [], born: null, ended: null }; regionHist.set(uid, e); } return e; };
function recordHistory(kind) {
  if (!regions) return; const rates = state.rates || regionRates();
  for (const r of regions.regions) { const e = histOf(r.uid), last = e.log[e.log.length - 1]; e.log.push({ clock: state.clock, kind, count: r.count, arr: r.arr, rate: rates ? rates.cnt[r.k] : 0, name: r.name, color: r.color, cx: r.cx, cy: r.cy, renamed: !!last && last.name !== r.name }); }
}
function showRegion() {
  const el = $('region'), k = rstate.focus; if (k < 0 || !regions || k >= regions.regions.length) { el.hidden = true; return; }
  const r = regions.regions[k], rates = state.rates || regionRates(), since = state.live ? 'going live' : 'Play'; el.hidden = false;
  $('region-title').textContent = r.name;
  $('region-count').textContent = `${r.count.toLocaleString()} living: ${r.countBase.toLocaleString()} from the base map, ${(r.count - r.countBase).toLocaleString()} arrivals` + (r.arr ? ` · +${r.arr.toLocaleString()} since ${since}` : '');
  const c = rates ? rates.cnt[k] : 0, ratio = rates ? rates.ratio[k] : 0;
  $('region-rate').textContent = rates && rates.total ? `${c} in the last ${mmss(state.rateWin || RATE_WIN)} · ${(100 * c / rates.total).toFixed(1)} % of arrivals vs ${(100 * r.countBase / data.nBase).toFixed(1)} % of the base map` + (ratio ? ` (${ratio.toFixed(1)}×)` : '') : 'no arrivals in the last minute';
  // sparkline: arrivals per 10 s over the last 10 minutes
  const cv = $('region-spark'), ctx = cv.getContext('2d'), dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 372, H = 44; if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; } ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const bins = regionSpark(k), maxB = Math.max(1, ...bins), bw = W / bins.length; ctx.fillStyle = r.color;
  for (let i = 0; i < bins.length; i++) { const hh = (H - 14) * bins[i] / maxB; ctx.fillRect(i * bw, H - 12 - hh, Math.max(1, bw - 1), hh); }
  ctx.fillStyle = '#6b7a90'; ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.fillText(`${tLabel(Math.max(0, state.clock - SPARK_WIN))}`, 0, H - 1); ctx.textAlign = 'right'; ctx.fillText(`${tLabel(state.clock)} · peak ${maxB} per ${SPARK_BIN} s`, W, H - 1);
  // biography
  const e = regionHist.get(r.uid), lines = [];
  if (e) {
    if (e.born) lines.push(`appeared at ${mmss(e.born.clock)}` + (e.born.from.length ? ` from ${e.born.from.map(f => `‘${f.name}’ (${Math.round(100 * f.share)} %)`).join(', ')}` : ' (new area)'));
    else lines.push('on the map since the base fit');
    const first = e.log[0]; if (first && e.log.length > 1) lines.push(`${first.count.toLocaleString()} posts at ${mmss(first.clock)} → ${r.count.toLocaleString()} now`);
    const renames = e.log.filter(x => x.renamed); for (const x of renames.slice(-3)) { const i = e.log.indexOf(x); lines.push(`called ‘${e.log[i - 1].name}’ until ${mmss(x.clock)}`); }
    const rebuilds = e.log.filter(x => x.kind === 'rebuild').length; if (rebuilds) lines.push(`${rebuilds} rebuild${rebuilds > 1 ? 's' : ''} kept its identity`);
  }
  $('region-hist').innerHTML = lines.map(esc).join('<br>');
  $('region-rep').innerHTML = r.rep >= 0 && data.posts[r.rep] ? esc(data.posts[r.rep].text) + '<small>the base post nearest the region’s mean vector</small>' : '';
}
// Chips keep their DOM elements between layouts (keyed by region) and update only what changed; layouts are
// coalesced to one per animation frame so a flush or a toggle never rebuilds the layer in the middle of a frame.
const chipEls = new Map(); let chipsQueued = false;
function scheduleChips() { if (chipsQueued || state.chipsHold) return; chipsQueued = true; requestAnimationFrame(() => { chipsQueued = false; layoutChips(); }); }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const wordSpan = (w) => { const s = document.createElement('span'); s.className = 'w'; s.textContent = w; return s; };
// A chip that kept its region through a rebuild but got a new name trades words in three beats: the words that leave
// fade out, the pill resizes once (a soft bounce, bigger or smaller) to the width the new words need, and the new words
// fade in. A rename arriving mid-animation is applied after the current one finishes.
async function animateWords(d, words) {
  if (d._anim) { d._next = words; return; }
  d._anim = true; const wrap = d.querySelector('.words'), width = () => d.getBoundingClientRect().width;
  try {
    for (let target = words; target; target = d._next) {
      d._next = null; const cur = [...wrap.children], have = new Map(cur.map(s => [s.textContent, s]));
      const leaving = cur.filter(s => !target.includes(s.textContent)), adding = target.filter(w => !have.has(w));
      if (leaving.length) { leaving.forEach(s => s.classList.add('out')); await sleep(300); }
      d.style.transition = 'none'; d.style.width = ''; const w0 = width(); const wasAt = new Map(cur.map(s => [s, s.getBoundingClientRect().left]));
      wrap.replaceChildren(...target.map(w => { const s = have.get(w) || wordSpan(w); if (!have.has(w)) s.classList.add('in'); return s; }));
      const w2 = width(); let slid = false;
      for (const s of wrap.children) { const from = wasAt.get(s); if (from === undefined) continue; const dx = from - s.getBoundingClientRect().left; if (Math.abs(dx) < 0.5) continue; s.style.transition = 'none'; s.style.transform = `translateX(${dx}px)`; slid = true; }
      if (slid || Math.abs(w2 - w0) > 0.5) {
        void d.offsetWidth; for (const s of wrap.children) { s.style.transition = ''; s.style.transform = ''; }
        if (Math.abs(w2 - w0) > 0.5) { d.style.width = w0 + 'px'; void d.offsetWidth; d.style.transition = 'width 400ms cubic-bezier(.34,1.25,.64,1)'; d.style.width = w2 + 'px'; }
        await sleep(400);
      }
      if (adding.length) { wrap.querySelectorAll('.w.in').forEach(s => s.classList.remove('in')); await sleep(300); }
      d.style.transition = ''; d.style.width = '';
    }
  } finally { d._anim = false; d._next = null; }
}
function focusUid() { return rstate.focus >= 0 && regions && regions.regions[rstate.focus] ? regions.regions[rstate.focus].uid : -1; }
function focusByUid(uid) { const k = regions ? regions.regions.findIndex(r => r.uid === uid) : -1; if (k >= 0) setFocus(k); else { const e = regionHist.get(uid), last = e && e.log[e.log.length - 1]; if (e && e.ended) status(`‘${last.name}’ ended ${mmss(e.ended.clock)}` + (e.ended.into.length ? ` → ‘${e.ended.into[0].name}’` : '')); } }
function layoutChips() {
  const el = $('chips'), R = view.on ? view.chips : (regions ? regions.regions : null);
  if (!R || !rstate.chips) { if (chipEls.size) { el.replaceChildren(); chipEls.clear(); } return; }
  const c = $('map'), dpr = window.devicePixelRatio || 1, v = renderer.view, W = c.clientWidth, H = c.clientHeight, placed = [], keep = new Set();
  const sx = (x) => (x - v.cx) * v.scale / dpr + W / 2, sy = (y) => H / 2 - (y - v.cy) * v.scale / dpr;
  const mix = renderer.rmix, live = new Set(R.map(r => r.uid)), fUid = focusUid();
  const shown = (r) => { const d = chipEls.get(r.uid); return d && !d.classList.contains('bye') ? 1 : 0; };
  // order: the focused chip, then chips already on screen (a crowded spot must not flip between two chips), then the larger region
  const order = [...R].sort((a, b) => (b.uid === fUid) - (a.uid === fUid) || shown(b) - shown(a) || b.count - a.count || a.uid - b.uid);
  for (const r of order) {
    const ax = r.prev && mix < 1 ? r.prev.cx + (r.cx - r.prev.cx) * mix : r.cx, ay = r.prev && mix < 1 ? r.prev.cy + (r.cy - r.prev.cy) * mix : r.cy;
    const fxs = sx(r.cx), fys = sy(r.cy); if (fxs < -40 || fxs > W + 40 || fys < -40 || fys > H + 40) continue;   // visibility and declutter use the final position
    const cxs = sx(ax), cys = sy(ay);
    // anchor the chip just above the visible spread of the region's points (1.5 standard deviations of their vertical spread, 12 to 60 px)
    const lift = Math.min(60, Math.max(12, 1.5 * r.sdy * v.scale / dpr)) + 8, cx = cxs, cy = cys - lift, fx = fxs, fy = fys - lift;
    const w = 7 * r.name.length + 70, h = 22; if (placed.some(q => Math.abs(q.x - fx) < (q.w + w) / 2 && Math.abs(q.y - fy) < h)) continue; placed.push({ x: fx, y: fy, w });
    let d = chipEls.get(r.uid), words = r.name.split(' · ');
    if (!d) {   // a new identity: the chip appears at its own place (never travels from somewhere else)
      d = document.createElement('div'); d.className = 'chip'; d.innerHTML = '<i></i><span class="words"></span><small></small>'; d._words = words; d.querySelector('.words').replaceChildren(...words.map(wordSpan));
      d.addEventListener('click', (ev) => { ev.stopPropagation(); focusByUid(d._uid); }); chipEls.set(r.uid, d); el.appendChild(d);
    } else if (d._words.join('|') !== words.join('|')) { d._words = words; animateWords(d, words); }   // renamed or re-ranked: words fade out, slide, resize, fade in
    d._uid = r.uid; keep.add(r.uid); d.classList.remove('bye'); d.classList.toggle('sel', r.uid === fUid); d.firstChild.style.background = !view.on && state.heat !== 'off' && state.rates ? heatCss(state.rates.h[r.k]) : r.color;
    // counts: a chip that kept its identity counts from the old values to the new ones along the morph; a new chip fades in
    const cnt = r.prev && mix < 1 ? Math.round(r.prev.count + (r.count - r.prev.count) * mix) : r.count, arr = r.prev && mix < 1 ? Math.round(r.prev.arr + (r.arr - r.prev.arr) * mix) : r.arr;
    const html = `${cnt.toLocaleString()}${arr ? ` <b>+${arr.toLocaleString()}</b>` : ''}`, sm = d.lastChild; if (sm.dataset.h !== html) { sm.innerHTML = html; sm.dataset.h = html; }
    d.style.left = cx + 'px'; d.style.top = cy + 'px'; d.style.opacity = (!r.prev && mix < 1) ? String(mix) : '';
    const ht = !view.on && state.heat !== 'off' && state.rates ? state.rates.h[r.k] : 0; const bc = !view.on && state.heat !== 'off' && state.rates && r.uid !== fUid ? heatCss(ht) : ''; if (d.style.borderColor !== bc) d.style.borderColor = bc;
  }
  for (const [uid, d] of chipEls) if (!keep.has(uid)) {
    if (live.has(uid)) { d.remove(); chipEls.delete(uid); }   // hidden by the declutter
    else if (!d.classList.contains('bye')) { d.classList.add('bye'); setTimeout(() => { if (d.classList.contains('bye')) { d.remove(); if (chipEls.get(uid) === d) chipEls.delete(uid); } }, 450); }   // its region is gone
  }
}

// ---- rewind and time-lapse. An epoch is the stretch between two quiet-rebuild commits. Its layout is what every row had
// then: the rows on the map at the commit (copied from engine.pos right after the commit) plus the rows placed during the
// epoch at the place they were put (recorded flush by flush; placement never moves a row until the next rebuild). At the
// next commit the epoch closes with every row's region membership and a slim copy of its regions (names, colours, centres,
// palette weights; no grids). The newest SNAP_MAX closed epochs are kept. The view is read-only: the engine, its buffers
// and the replay hash are never touched; while the view is on the renderer draws from a second pair of position buffers.
const SNAP_MAX = 60, REWIND_GLIDE_MS = 800, SCRUB_GLIDE_MS = 350;
const snaps = [];   // closed epochs, oldest first
let live = null;    // the open epoch
const ease = (t) => t * t * (3 - 2 * t);
function slimRegions(reg) { return reg ? { g: reg.g, extent: reg.extent, sigma: reg.sigma, softSigma: reg.softSigma, regions: reg.regions.map(r => ({ k: r.k, uid: r.uid, name: r.name, color: r.color, cx: r.cx, cy: r.cy, sdy: r.sdy, peak: r.peak, peakTint: r.peakTint, countBase: r.countBase })) } : null; }
function snapOpen(kind) { live = { k: snaps.length ? snaps[snaps.length - 1].k + 1 : 0, kind, clock: state.clock, n0: engine.n, pos: engine.pos.slice(0, engine.n * 2), chunks: [] }; }
function snapTail(start, w) { if (!live || start < live.n0) return; live.chunks.push(engine.pos.slice(start * 2, (start + w) * 2)); }
function snapClose() {
  if (!live) return; const t0 = performance.now(), n1 = engine.n, full = new Float32Array(n1 * 2); full.set(live.pos.subarray(0, Math.min(live.n0, n1) * 2));
  let at = live.n0 * 2; for (const c of live.chunks) { if (at + c.length > n1 * 2) break; full.set(c, at); at += c.length; }
  if (at < n1 * 2) full.set(engine.pos.subarray(at, n1 * 2), at);   // rows with no recorded chunk (none expected): their current positions
  const e = { k: live.k, kind: live.kind, clock: live.clock, end: state.clock, n0: live.n0, n1, full, reg: regionS.slice(0, n1), regions: slimRegions(regions) };
  snaps.push(e); while (snaps.length > SNAP_MAX) snaps.shift(); live = null;
  logEvent('snapshot', { k: e.k, rows: n1, bytes: full.byteLength + e.reg.byteLength, ms: +(performance.now() - t0).toFixed(2), kept: snaps.length, totalMB: +(snapBytes() / 1048576).toFixed(1) });
}
function snapClear() { snaps.length = 0; live = null; }
function snapBytes() { let b = 0; for (const e of snaps) b += e.full.byteLength + e.reg.byteLength; if (live) { b += live.pos.byteLength; for (const c of live.chunks) b += c.byteLength; } return b; }
// the view: which epoch is shown (null = the open one, drawn from engine.pos), at which stream time, and the glide from the layout shown before
const view = { on: false, T: 0, shown: null, n: 0, posA: null, posB: null, g0: 0, gMs: 0, e: 1, chips: null, auto: null, exitWhenDone: false, retargets: 0, retargetMs: 0 };
function epochAt(T) { for (let i = snaps.length - 1; i >= 0; i--) if (T >= snaps[i].clock) return T < snaps[i].end ? snaps[i] : null; return null; }
const layoutOf = (e) => e ? e.full : engine.pos, regOf = (e) => e ? e.reg : regionS, regionsOf = (e) => e ? e.regions : regions, rowsOf = (e) => e ? e.n1 : engine.n;
function rowsAt(e, T) { return Math.min(rowsOf(e), Math.max(data.nBase, firstRowAfter(T))); }
function viewLo() { return snaps.length ? snaps[0].clock : (live ? live.clock : data.baseSeconds); }
function chipList(e, prevList) {
  const R = regionsOf(e); if (!R) return null; const byUid = new Map((prevList || []).map(r => [r.uid, r]));
  return R.regions.map(r => { const p = byUid.get(r.uid); return { k: r.k, uid: r.uid, name: r.name, color: r.color, cx: r.cx, cy: r.cy, sdy: r.sdy, countBase: r.countBase, count: r.countBase, arr: 0, prev: p ? { cx: p.cx, cy: p.cy, count: p.count, arr: p.arr } : null }; });
}
function viewCounts() {   // living posts per region at the viewed time, counted from the rows shown
  const C = view.chips; if (!C) return; const reg = regOf(view.shown), cf = state.countFrom, n = view.n, cnt = new Int32Array(C.length), arr = new Int32Array(C.length);
  for (let p = data.nBase; p < n; p++) { const r = reg[p]; if (r >= 0 && r < C.length) { cnt[r]++; if (data.tS[p] > cf) arr[r]++; } }
  for (const r of C) { r.count = r.countBase + cnt[r.k]; r.arr = arr[r.k]; }
}
function viewHint() { const reb = state.rebuildMarks.length; $('r-view').textContent = view.on ? `${view.auto ? 'replay · ' : ''}${tLabel(view.T)} · ${view.shown ? (view.shown.k ? `after rebuild ${view.shown.k} of ${reb}` : 'base layout') : 'current layout'}` : 'off'; }
function viewTo(T, glideMs = REWIND_GLIDE_MS) {
  if (state.rebuilding) { status('Rewind waits for the rebuild in progress.'); return; }
  const head = state.clock; T = Math.max(viewLo(), Math.min(head, T)); const tgt = epochAt(T), now = performance.now();
  if (!view.on) {
    if (state.playing) setPlaying(false); if (state.inspect !== null) clearInspect();
    view.on = true; view.shown = null; view.e = 1; view.n = engine.n; view.posA = view.posA && view.posA.length >= data.N * 2 ? view.posA : new Float32Array(data.N * 2); view.posB = engine.pos; view.exitWhenDone = false;
    renderer.setViewBuffers(true); renderer.uploadView(engine.pos, engine.pos, 0, engine.n); if (regions) renderer.setRegions(regions, tintPx(), rstate.soft, prevRegions);
    view.chips = chipList(null, null); logEvent('rewind on', { from: Math.round(head), to: Math.round(T) });
  }
  if (tgt !== view.shown) {
    const t0 = performance.now(), nS = view.n, A = view.posA, B = view.posB, e = view.e;
    if (e < 1) { for (let i = 0; i < nS * 2; i++) A[i] += (B[i] - A[i]) * e; } else A.set(B.subarray(0, nS * 2));   // what is on screen now is where the glide starts
    const L = layoutOf(tgt), n = rowsAt(tgt, T); if (n > nS) A.set(L.subarray(nS * 2, n * 2), nS * 2);   // rows not shown before appear in place
    renderer.uploadView(A, L, 0, n);
    const rT = regOf(tgt), rS = regOf(view.shown), prev = new Int16Array(n); prev.set(rS.subarray(0, Math.min(nS, n))); if (n > nS) prev.set(rT.subarray(nS, n), nS);
    renderer.setRegions(regionsOf(tgt), rstate.sigma, rstate.soft, regionsOf(view.shown)); renderer.updateRegionIds(rT, 0, n, prev); renderer.setHeat(null);
    view.chips = chipList(tgt, view.chips); view.posB = L; view.n = n; view.shown = tgt; view.g0 = now; view.gMs = glideMs; view.e = glideMs > 0 ? 0 : 1;
    view.retargets++; view.retargetMs += performance.now() - t0;
  } else {
    const n = rowsAt(tgt, T), L = view.posB;
    if (n > view.n) { view.posA.set(L.subarray(view.n * 2, n * 2), view.n * 2); renderer.uploadView(view.posA, L, view.n, n - view.n); renderer.updateRegionIds(regOf(tgt), view.n, n - view.n, regOf(tgt)); }
    view.n = n;
  }
  view.T = T; viewCounts(); viewHint(); kickAnim(); scheduleChips(); drawTimeline();
}
function viewDraw() {
  if (view.auto) viewAutoStep();
  const e = view.e < 1 ? ease(Math.min(1, (performance.now() - view.g0) / Math.max(1, view.gMs))) : 1; view.e = e;
  const fUid = focusUid(), R = view.shown ? view.shown.regions : regions; renderer.focusRegion = fUid >= 0 && R ? R.regions.findIndex(r => r.uid === fUid) : -1;
  renderer.settle = e; renderer.rmix = e; renderer.flushStart = 0; renderer.curTime = view.T; renderer.rate = state.speed; renderer.draw(view.n, view.n); scheduleChips();   // chips follow the same clock as the dots
  if (e >= 1 && view.exitWhenDone && !view.auto) { viewExit(); return false; }
  return e < 1 || !!view.auto;
}
// back to the live state: the engine's own buffers, the live regions, membership and heat
function viewExit() {
  if (!view.on) return; view.on = false; view.auto = null; view.exitWhenDone = false; $('replay30').classList.remove('on'); view.chips = null;
  renderer.setViewBuffers(false); if (regions) renderer.setRegions(regions, tintPx(), rstate.soft, prevRegions); renderer.updateRegionIds(regionS, 0, engine.n, regionPrevS); regionRates();
  renderer.settle = 1; renderer.rmix = 1; renderer.focusRegion = rstate.focus; viewHint(); render(); drawTimeline(); logEvent('rewind off', {});
}
// the viewed time reached the play-head: finish the glide into the live layout, then leave the view
function viewFinish() { if (!view.on) return; if (view.shown === null && view.e >= 1) viewExit(); else { view.exitWhenDone = true; viewTo(state.clock, REWIND_GLIDE_MS); } }
// time-lapse: the kept stretch of the stream played in `totalMs` of real time, layouts gliding into one another at each rebuild
// The time-lapse plays the snapshots taken at rebuilds. With fewer than four of them over the loaded stream (a page that
// was never played, or played with few rebuilds) it first makes them: reset to the base map, place the stream in six
// segments with a rebuild after each (no morph, no automatic cadence), then plays the six epochs with the glide between.
async function prepareEpochs(segs = 6) {
  if (state.playing) setPlaying(false); if (view.on) viewExit();
  const end = loadedEnd(); if (end - data.baseSeconds < 60 || data.n <= data.nBase) return false;
  const savedMorph = state.morphMs, savedAuto = state.autoRebuild; state.morphMs = 0; state.autoRebuild = false; state.catchingUp = true; updateLiveBadge();   // placed without halos: these rows are the recorded past, not arrivals
  try {
    await resetToBase();
    for (let i = 1; i <= segs; i++) {
      const T = data.baseSeconds + (end - data.baseSeconds) * i / segs;
      status(`Preparing the replay: placing to ${mmss(T)} (segment ${i} of ${segs})…`); await runTo(T);
      status(`Preparing the replay: rebuild ${i} of ${segs}…`); await doRebuild('replay');
    }
  } finally { state.morphMs = savedMorph; state.autoRebuild = savedAuto; state.catchingUp = false; updateLiveBadge(); }
  logEvent('timelapse prepared', { segments: segs, snapshots: snaps.length }); return true;
}
async function timelapse(totalMs = 30000) {
  if (view.on && view.auto) { viewExit(); return; }
  if (snaps.length < 4 && !(await prepareEpochs(6))) { status('Nothing to replay yet'); return; }
  const lo = viewLo(), head = state.clock; if (head - lo < 1) { status('Nothing to replay yet'); return; }
  if (state.playing) setPlaying(false);
  const segs = snaps.length + 1, glideMs = Math.min(1200, Math.max(250, 0.6 * totalMs / segs));
  view.auto = { t0: performance.now(), ms: totalMs, lo, head, glideMs }; $('replay30').classList.add('on');
  logEvent('timelapse start', { seconds: Math.round(head - lo), epochs: segs, glideMs: Math.round(glideMs) }); viewTo(lo, 600); view.auto.t0 = performance.now() + 600;
}
function viewAutoStep() {
  const a = view.auto, t = Math.min(1, Math.max(0, (performance.now() - a.t0) / a.ms));
  if (t >= 1) { view.auto = null; $('replay30').classList.remove('on'); logEvent('timelapse done', {}); viewFinish(); return; }
  viewTo(a.lo + (a.head - a.lo) * t, a.glideMs);
}

// Topic-cluster names follow the same naming rule and NSFW setting as the regions (colours stay with their cluster index).
function relabel() {
  if (data.catIndex) return;   // a static dataset's own labels keep their names
  if (!data.docs) data.docs = docsOf(data.posts.slice(0, data.nBase));
  const named = nameClusters(data.posts.slice(0, data.nBase), data.labelS, data.labels.length, data.docs), colors = {};
  named.names.forEach((nm, i) => { colors[nm] = data.labelColors[data.labels[i]] || named.colors[nm]; }); data.labels = named.names; data.labelColors = colors;
}
function legend() {   // one short line per entry: a swatch and a few words; the explanations live in About
  const el = $('legend'); el.innerHTML = ''; const mode = renderer.mode; const add = (html) => { const d = document.createElement('div'); d.innerHTML = html; el.appendChild(d); };
  const ramp = (stops, w = 90) => `<i style="background:linear-gradient(90deg,${stops.join(',')});width:${w}px"></i>`, nm = (t) => t.length > 34 ? t.slice(0, 33) + '…' : t;
  if (mode === 0) { const counts = new Map(); for (let p = 0; p < engine.n; p++) counts.set(data.labelS[p], (counts.get(data.labelS[p]) || 0) + 1);
    [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12).forEach(([l, c]) => add(`<i style="background:${data.labelColors[data.labels[l]]}"></i>${nm(data.labels[l])} <small>${c.toLocaleString()}</small>`)); add(`<small>${data.labels.length} topics</small>`); }
  else if (mode === 1) add(`${ramp(['#fadc40', '#229e9e', '#383380'])}new → ${mmss(Math.max(1, loadedEnd() - data.baseSeconds))} old`);
  else if (mode === 2) { add('<i style="background:#8c99ad"></i>base map'); add('<i style="background:#ffb34d"></i>streamed'); }
  else if (mode === 3) add(`${ramp(['#d95930', '#9ea3ad', '#1c9e75'])}local fit: poor → full`);
  else if (mode === 5) { add('<i style="background:#f2bf4d"></i>reply'); add('<i style="background:#59a6f2"></i>original'); }
  else if (mode === 6) { if (regions) { [...regions.regions].sort((a, b) => b.count - a.count).slice(0, 12).forEach(r => add(`<i style="background:${r.color}"></i>${nm(r.name)} <small>${r.count.toLocaleString()}</small>`)); add(`<small>${regions.regions.length} regions</small>`); } else add('<small>regions off</small>'); }
  else { add('<i style="background:#ffd966"></i>latest flush'); add('<i style="background:#666c78"></i>earlier'); }
  if (renderer.showStranded) add('<i style="border:1.5px solid #f5733f;border-radius:50%;background:none"></i>stranded');
  if (renderer.growth) add(`<i style="background:#f8c060"></i>arrivals, last ${renderer.growthWindow} s`);
  if (state.heat !== 'off') add(`${ramp([0, 0.25, 0.5, 0.75, 1].map(heatCss))}${state.heat === 'growth' ? 'growth' : 'volume'}, cold → hot`);
}

function inspectAt(clientX, clientY) {
  if (view.on) return;
  const c = $('map'), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, v = renderer.view;
  const wx = v.cx + ((clientX - r.left) * dpr - c.width / 2) / v.scale, wy = v.cy - ((clientY - r.top) * dpr - c.height / 2) / v.scale;
  const P = engine.pos; let best = -1, bd = Infinity;
  for (let p = 0; p < engine.n; p++) { if (!engine.alive[p]) continue; const dx = P[2 * p] - wx, dy = P[2 * p + 1] - wy, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = p; } }
  if (best < 0 || Math.sqrt(bd) * v.scale > 12 * dpr) { clearInspect(); return; }
  const nb = Array.from(engine.knnIdx.subarray(best * K, best * K + K)), ad = engine.adoptersOf(best);
  state.inspect = { p: best, neighbours: nb, adopters: ad }; showInspect(); render();
}
function clearInspect() { state.inspect = null; renderer.hlCount = 0; $('inspect').hidden = true; render(); }
const esc = (s) => s.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
function showInspect() {
  const I = state.inspect; if (!I) return; const p = I.p, nbSet = new Set(I.neighbours), adSet = new Set(I.adopters), ids = [p], kinds = [0];
  if (state.inspectMode !== 'adopters') for (const j of I.neighbours) { ids.push(j); kinds.push(adSet.has(j) && state.inspectMode === 'both' ? 3 : 1); }
  if (state.inspectMode !== 'neighbours') for (const i of I.adopters) { if (state.inspectMode === 'both' && nbSet.has(i)) continue; ids.push(i); kinds.push(2); }
  renderer.setHighlight(ids, kinds);
  const mutual = I.neighbours.filter(j => adSet.has(j)).length, post = data.posts[p], lab = data.labels[data.labelS[p]] || 'unlabelled';
  $('inspect').hidden = false;
  $('inspectTitle').textContent = `Inspected ${SRC.one}`; $('inspect-title').textContent = `${SRC.one[0].toUpperCase() + SRC.one.slice(1)} ${p.toLocaleString()}${SRC.static ? (SRC.timeAxis ? ` · ${tLabel(data.tS[p])}` : '') : ` · ${mmss(data.tS[p])} of stream time (${wall(data.wallS[p])} local)`} · ${p < data.nBase ? 'base map' : 'streamed'}${post.reply ? ' · reply' : ''}`;
  const reg = regions && regionS[p] >= 0 ? regions.regions[regionS[p]] : null;
  $('inspect-post').innerHTML = (post.img ? `<img class="thumb" src="${post.img}" alt="">` : '') + esc(post.text) + `<small><i class="sw" style="background:${data.labelColors[lab] || '#666'}"></i>${esc(lab)}${reg ? ` · region: ${esc(reg.name)}` : (regions ? ' · outside every region' : '')}${post.tags.length ? ' · #' + post.tags.map(esc).join(' #') : ''}${post.score != null ? ` · score ${post.score}${post.descendants != null ? `, ${post.descendants} comments` : ''}` : ''}${post.kind ? ` · ${esc(String(post.kind))}` : ''}</small>`;
  $('inspect-body').innerHTML = `<div><i class="sw" style="background:#59e6c7"></i>${K} nearest by meaning (its current list)</div><div><i class="sw" style="background:#fa80c8"></i>${I.adopters.length} posts keep it in their own ${K}-nearest list (adopters)</div><div><i class="sw" style="background:#ffd959"></i>${mutual} are both</div><small>lists are as of ${mmss(state.clock)}; neighbours placed later than this post can appear</small>`;
  $('inspect-nbrs').innerHTML = I.neighbours.slice(0, 5).map((j, k) => { const q = data.posts[j]; const t = q.text.length > 220 ? q.text.slice(0, 217) + '…' : q.text; return `<div class="nbr"><b>${k + 1}.</b> ${esc(t)}</div>`; }).join('');
}

// ---- live mode: receive embedded batches from the relay and append them behind the loaded rows. The
// connection is independent of play/pause: paused, arrivals queue up and are placed on the next Play.
function updateLiveBadge() {
  const b = $('livebadge'); b.classList.toggle('on', state.live); b.classList.toggle('paused', state.live && !state.playing);
  b.classList.toggle('on', state.live || state.catchingUp); b.textContent = state.catchingUp ? 'CATCHING UP' : state.live ? (state.playing ? (state.worker ? 'LIVE · browser' : 'LIVE · relay') : 'LIVE · paused') : '';
  $('golive').textContent = (state.ws || (state.worker && (!FRESH || state.live))) ? 'Leave live' : 'Go live';
}
const playedNow = () => state.playedMs + (state.playing ? performance.now() - state.playWall : 0);
function nextRebuildText() {
  if (state.rebuilding) return 'rebuilding…';
  if (!state.autoRebuild && !state.rebuildArrivals) return '';
  const parts = [];
  if (state.autoRebuild) parts.push(mmss(Math.max(0, state.rebuildMinutes * 60 - (playedNow() - state.lastRebuildPlayed) / 1000)) + (state.playing ? '' : ' (paused)'));
  if (state.rebuildArrivals) parts.push(`${Math.max(0, state.rebuildArrivals - (engine.placedTotal - engine.placedAtRebuild)).toLocaleString()} arrivals`);
  return 'next in ' + parts.join(' or ');
}
setInterval(() => { if (engine) $('r-next').textContent = nextRebuildText(); }, 1000);
function setPlaying(on) { if (on && view.on) viewExit(); if (on && !state.playing) state.playWall = performance.now(); else if (!on && state.playing) state.playedMs += performance.now() - state.playWall; state.playing = on; $('play').textContent = on ? 'Pause' : 'Play'; updateLiveBadge(); if (on && state.live) status(state.worker ? LIVE_MSG('WebGPU') : 'Live from the relay'); if (on) playLoop(); }
// One live frame (from the relay socket or the in-browser worker): vectors + posts, placed behind the loaded rows.
function ingestFrame(hdr, vecs, refuse) {
  if (hdr.model && data.model && hdr.model !== data.model) { refuse(`Live feed refused: it embeds with ${hdr.model} but this base was built from ${data.model}. Pick a ${hdr.encoder || hdr.model} base or restart the relay with --encoder ${data.encoder}.`); return; }
  if (hdr.dim !== data.dim || vecs.length !== hdr.n * hdr.dim) { status('live frame has the wrong vector size', true); return; }
  const walls = hdr.posts.map(p => (p.time_us - data.t0us) / 1e6), posts = hdr.posts.map(p => ({ text: p.text, terms: p.terms, icao24: p.icao24, tags: p.tags || [], reply: p.reply || 0, url: p.url, kind: p.kind, img: p.img, cat: p.cat, score: p.score, descendants: p.descendants })), replies = hdr.posts.map(p => p.reply || 0);
  // stream time is continuous: a live session starts one second after the last loaded post (the real gap is kept in wallS for the readout)
  if (state.liveShift === null) state.liveShift = walls[0] - (loadedEnd() + 1);
  const times = walls.map(w => w - state.liveShift);
  for (let i = 0; i < times.length; i++) times[i] = Math.max(times[i], (i ? times[i - 1] : loadedEnd()) + 1e-3);   // keep stream time monotone
  if (data.n + times.length > data.N) {   // no row slot left (long pause, or retirement cannot keep up): drop this frame, never crash the feed
    state.droppedFull = (state.droppedFull || 0) + times.length;
    status(state.playing ? `Row store full · ${state.droppedFull.toLocaleString()} dropped`
                         : `Row store full · ${state.droppedFull.toLocaleString()} dropped`, true);
    return;
  }
  const at = engine.appendRows(vecs, times, posts, replies, walls); renderer.updateTimes(at, times.length); countArrivals(at, at + times.length);
  if (!state.playing) { if (state.live) status(`Live, paused · ${(data.n - engine.n).toLocaleString()} waiting`); drawTimeline(); }
}
// Going live is a sequence, not a jump: the recorded stream still ahead of the play-head is played to the present at
// 180x (points fade in without halos and are not counted as arrivals: they are recorded, not live), then one rebuild
// settles everything that just landed, and only then does the live stream start and "+N" begin to count.
async function liveOn(msg) {
  $('speed').disabled = true; state.countFrom = state.captureEnd; if (regions) { regions.regions.forEach(r => { r.arr = 0; }); }
  if (state.clock < loadedEnd()) {
    state.catchingUp = true; const saved = { speed: state.speed, interval: state.interval }; state.speed = 180; state.interval = 10; updateLiveBadge();
    status(`Catching up ${mmss(loadedEnd() - state.clock)} at 180×…`); logEvent('catch-up start', { behindS: Math.round(loadedEnd() - state.clock) });
    if (!state.playing) setPlaying(true);
    while (state.playing && state.clock < loadedEnd()) await new Promise(r => setTimeout(r, 100));
    if (state.playing) setPlaying(false);
    state.speed = saved.speed; state.interval = saved.interval; state.catchingUp = false; logEvent('catch-up done', { placed: engine.n });
    status('Going live: one rebuild…'); try { await doRebuild('go live'); } catch (e) { /* logged; the stream still starts */ }
  }
  state.live = true; setClock(loadedEnd()); status(msg); if (!state.playing) setPlaying(true); else updateLiveBadge();
}
function liveOff(msg, isError) { state.live = false; state.catchingUp = false; state.ws = null; state.liveShift = null; $('speed').disabled = false; updateLiveBadge(); status(msg, isError); }
function liveStatsText() { const st = state.liveStats; if (!st) return ''; const s = st.stats; return `${s.kept.toLocaleString()} kept of ${s.post_creates.toLocaleString()} ${SRC.noun} seen · embed ${s.batches ? (s.embed_ms / s.batches).toFixed(0) : '–'} ms per batch of ${s.batches ? (s.embedded / s.batches).toFixed(1) : '–'} on ${st.device}`; }

// Go live, source chosen by the Live source control: the in-browser worker (Jetstream + transformers.js, no server)
// or the local relay (jetstream_relay.py --serve).
function goLive() {
  if (!LIVE) { const u = new URL(location.href); u.searchParams.set('live', '1'); u.searchParams.set('src', state.liveSource); location.href = u.toString(); return; }
  if (state.ws) { state.ws.close(); return; }
  if (FRESH && state.worker && !state.live) { liveOn(LIVE_MSG(state.liveStats?.device || 'webgpu')); return; }
  if (state.worker) { state.worker.postMessage({ type: 'stop' }); return; }
  state.captureEnd = loadedEnd();   // everything loaded before this moment is recorded; only later rows are live arrivals
  if (state.liveSource === 'relay') goLiveRelay(); else goLiveBrowser();
}
const LIVE_MSG = (device) => `Live · ${SRC.feed} on ${device}`;
// The source list: the hidden <select id="livesrc"> holds the value; this draws it as a list with a small icon per entry,
// an antenna with a ping for live feeds and a table for static datasets.
const ICON_LIVE = '<svg class="ic ant" viewBox="0 0 14 14" aria-label="live"><circle class="ping" cx="7" cy="5" r="2" fill="none" stroke="currentColor" stroke-width="1"/><circle cx="7" cy="5" r="1.3" fill="currentColor"/><path d="M4.4 2.4a3.6 3.6 0 0 0 0 5.2M9.6 2.4a3.6 3.6 0 0 1 0 5.2M7 6.3v6.2M5 12.5h4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" fill="none"/></svg>';
const ICON_DATA = '<svg class="ic dset" viewBox="-1 -1 14 14" aria-label="dataset"><rect x="1.5" y="2" width="9" height="8" rx="1" fill="none" stroke="currentColor" stroke-width="1"/><path d="M1.5 5h9M1.5 7.5h9M5 2v8" stroke="currentColor" stroke-width="1" fill="none"/></svg>';
function sourcePicker() {
  const sel = $('livesrc'), btn = $('srcBtn'), list = $('srcList'); if (!btn) return;
  if (!/^(localhost|127\.0\.0\.1)$/.test(location.hostname)) sel.querySelector('option[value="opensky"]')?.remove();   // aircraft come through a relay on this machine: not on the public site
  const icon = (o) => RECORDED ? '' : o.parentElement?.label === 'Live' ? ICON_LIVE : ICON_DATA;
  let focus = -1;
  const paint = () => {
    const cur = sel.selectedOptions[0]; btn.innerHTML = (cur ? icon(cur) : '') + `<span class="nm">${esc(cur?.text || '')}</span>`;
    let html = '', i = 0;
    for (const node of sel.children) {
      const opts = node.tagName === 'OPTGROUP' ? [...node.children] : [node]; if (!opts.length) continue;
      if (node.tagName === 'OPTGROUP') html += `<div class="grp">${esc(node.label)}</div>`;
      for (const o of opts) html += `<div class="opt${i++ === focus ? ' focus' : ''}" role="option" data-v="${esc(o.value)}" aria-selected="${o.selected}">${icon(o)}<span class="nm">${esc(o.text)}</span></div>`;
    }
    list.innerHTML = html;
  };
  const open = (on) => { list.hidden = !on; btn.setAttribute('aria-expanded', on); if (on) { focus = [...sel.options].indexOf(sel.selectedOptions[0]); paint(); } };
  const choose = (v) => { open(false); if (v === sel.value) return; sel.value = v; paint(); sel.dispatchEvent(new Event('change')); };
  btn.addEventListener('click', () => open(list.hidden));
  list.addEventListener('click', (e) => { const o = e.target.closest('.opt'); if (o) choose(o.dataset.v); });
  document.addEventListener('pointerdown', (e) => { if (!list.hidden && !e.target.closest('.srcpick')) open(false); });
  btn.addEventListener('keydown', (e) => {
    const n = sel.options.length;
    if (e.key === 'Escape') open(false);
    else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); if (list.hidden) open(true); else { focus = (focus + (e.key === 'ArrowDown' ? 1 : n - 1)) % n; paint(); } }
    else if (e.key === 'Enter' && !list.hidden && focus >= 0) { e.preventDefault(); choose(sel.options[focus].value); }
  });
  paint();
}
function attachWorker(w) {
  w.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'progress') status(`Loading the encoder… ${m.pct?.toFixed(0) ?? ''} %`);
    else if (m.type === 'ready') { $('r-live').textContent = `in-browser: ${SRC.feed} direct, ${m.hf} on ${m.device} (loaded in ${(m.loadMs / 1000).toFixed(1)} s)`; if (!FRESH) liveOn(LIVE_MSG(m.device)); }
    else if (m.type === 'frame') { if (FRESH && state.cacheDb) writeFrame(state.cacheDb, m.hdr, m.vecs); ingestFrame(m.hdr, m.vecs, (why) => { state.liveRefused = why; w.postMessage({ type: 'stop' }); });
      if (FRESH && !state.freshReady && !m.hdr.backfilling && !m.hdr.pendingAfter) { state.freshReady = true; logEvent('backfill embedded', { rows: data.n }); if (!state.live) status('Ready'); } }
    else if (m.type === 'stats') { state.liveStats = { stats: m.stats, device: state.liveStats?.device || 'webgpu' }; $('r-livestats').textContent = liveStatsText(); if (FRESH) $('r-filter').textContent = filterText(m.stats); }
    else if (m.type === 'log') console.log('[live worker]', m.msg);
    else if (m.type === 'error') { const why = 'In-browser live source failed: ' + m.msg; w.terminate(); state.worker = null; liveOff(why, true); }
    else if (m.type === 'stopped') { w.terminate(); state.worker = null; const refused = state.liveRefused; state.liveRefused = null; liveOff(refused || 'Left the live feed. Press Go live to reconnect.', !!refused); }
  };
  w.onerror = (e) => { w.terminate(); state.worker = null; liveOff('In-browser live source failed: ' + (e.message || 'worker error'), true); };
}
function goLiveBrowser() {
  const w = new Worker('./live_worker.js', { type: 'module' }); state.worker = w; updateLiveBadge();
  status('Loading the encoder…');
  attachWorker(w);
  w.postMessage({ type: 'start', batchSeconds: Math.min(state.interval, 5), device: params.get('device') || 'webgpu' });
}
const filterText = (s) => SRC.numeric ? (s && s.post_creates ? `${s.kept.toLocaleString()} samples of ${s.post_creates.toLocaleString()} states` : '—') : s && s.post_creates ? `${s.kept.toLocaleString()} kept of ${s.post_creates.toLocaleString()} ${SRC.noun} seen: ${s.not_english.toLocaleString()} not English, ${s.too_short.toLocaleString()} under ${SRC.minWords} words, ${s.link_only.toLocaleString()} link-only, ${s.duplicate.toLocaleString()} duplicates${s.not_english_text ? `, ${s.not_english_text.toLocaleString()} tagged English but not` : ''}` : '—';

// ---- Loading preview: while the base window is being embedded, every post appears as a dot on the first two principal
// axes of the vectors (axes from the first 1,024 posts, then fixed). The fit starts from these same positions, so when
// it is ready the cloud glides into the fitted map with no jump. Drawn on a plain 2-D canvas over the (not yet started)
// WebGPU canvas; the renderer's view is set to the same mapping at hand-off.
const paint = () => new Promise(r => { let d = false; const go = () => { if (!d) { d = true; r(); } }; requestAnimationFrame(go); setTimeout(go, 50); });   // next frame, or 50 ms if frames are not delivered
const PREVIEW_EXTENT = 22;   // map units across the shorter side (positions have standard deviation 3 per axis)
function previewMapping() { const c = $('preview'), dpr = window.devicePixelRatio || 1; let W = c.clientWidth, H = c.clientHeight; if (!W || !H) { const r = $('map').parentElement.getBoundingClientRect(); W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height)); } return { W, H, scale: 0.9 * Math.min(W, H) * dpr / PREVIEW_EXTENT }; }
function drawPreview(col) {
  if (col.tw) { drawTween(col); return; }
  const c = $('preview'), dpr = window.devicePixelRatio || 1, { W, H, scale } = previewMapping(); if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const ctx = c.getContext('2d'); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height); ctx.fillStyle = 'rgba(159,176,198,0.55)';
  const xy = col.previewXY, n = col.previewN, s = scale, cx = c.width / 2, cy = c.height / 2, r = 1.2 * dpr;
  for (let i = 0; i < n; i++) { const x = cx + xy[2 * i] * s, y = cy - xy[2 * i + 1] * s; ctx.fillRect(x - r, y - r, 2 * r, 2 * r); }
}
// ---- Fresh start: fetch the last FRESH minutes from Jetstream through the worker, keep the first half for the base map.
// Resolves once the base window is complete (the worker keeps running; frames after the window are queued in `later`
// until attachWorker takes over). Rows are time-ordered, so the window is complete when a post past its end arrives.
// A static dataset is not played: it is shown finished. While its cloud is still up, every row is placed out of sight
// (one flush per month or year of a dated table, as the ASRS replay flushed month by month, or 1,024 rows at a time), and
// only then does the map appear, every row gliding from its place in the cloud to its place on the map with nothing else
// running, so the glide gets every frame. A dated table can then be scrubbed back on the timeline; a table without dates
// has no timeline and nothing to play.
function staticUi() {
  document.body.classList.toggle('notime', noTime());
  if (!UNTIL) for (const id of ['rebuildNow', 'r-next', 'lineage-btn']) { const el = $(id); if (el) el.hidden = true; }   // nothing streams: no re-layouts, no lineage
  $('replay30').hidden = noTime() || !UNTIL; $('speed').closest('.ctl')?.setAttribute('hidden', '');
  $('replay30').textContent = 'Replay in 30 s'; $('lineage-btn').hidden = noTime() || !UNTIL;
  if (data.catIndex) {   // coloured by the dataset's own label, named as the dataset names it
    const o = [...$('mode').options].find((x) => x.value === '0'); if (o) o.textContent = SRC.categoryName || 'label';
    $('mode').value = '0'; $('mode').dispatchEvent(new Event('change'));
  }
}
// the flush interval Play uses on a rewound map: one unit of dates (month, year), or 1,024 rows for an undated table
async function staticInterval() {
  for (let i = 0; i < 600 && data.n < (SRC.rowsLoaded || 0); i++) await new Promise((r) => setTimeout(r, 50));
  const span = loadedEnd() - data.baseSeconds, unitT = SRC.timeAxis ? SRC.timeSpan / ((SRC.timeAxis.d1 - SRC.timeAxis.d0) / (SRC.timeAxis.unit === 'year' ? YEAR_MS : SRC.timeAxis.unit === 'month' ? YEAR_MS / 12 : 86400000)) : 0;
  state.interval = SRC.timeAxis ? unitT : span * 1024 / Math.max(1, data.n - data.nBase);
  const sel = $('interval'), o = new Option(SRC.timeAxis ? `1 ${SRC.timeAxis.unit}` : '1,024 rows', String(state.interval), true, true); sel.add(o); sel.closest('.ctl').querySelector('.unit').textContent = SRC.timeAxis ? 'of dates' : '';
  updateReadouts(state.lastStats); drawTimeline();
}
// Every row's place in the cloud (its vector on the cloud's two principal axes, as pcaProject), computed on the GPU from the
// features already there and written straight into the start buffer the glide reads: 0.5 s of CPU on 160,000 rows of
// 384 numbers becomes one dispatch. The CPU copy of the start positions follows by an asynchronous read.
const WGSL_CLOUD = `struct U { n: u32, dim: u32, p0: u32, p1: u32, s0: f32, s1: f32, p2: f32, p3: f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> ax: array<f32>;   // mean, axis 1, axis 2 (dim each)
@group(0) @binding(3) var<storage, read_write> outXY: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= u.n) { return; }
  var a0 = 0.0; var a1 = 0.0; let b = i * u.dim;
  for (var d = 0u; d < u.dim; d = d + 1u) { let x = feat[b + d] - ax[d]; a0 = a0 + x * ax[u.dim + d]; a1 = a1 + x * ax[2u * u.dim + d]; }
  outXY[2u * i] = a0 * u.s0; outXY[2u * i + 1u] = a1 * u.s1;
}`;
function gpuCloudStart(axes, n) {
  const d = engine.device, dim = data.dim, ab = new Float32Array(3 * dim); ab.set(axes.mean, 0); ab.set(axes.comps[0], dim); ab.set(axes.comps[1], 2 * dim);
  const axBuf = d.createBuffer({ size: ab.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }), uBuf = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  d.queue.writeBuffer(axBuf, 0, ab); const ub = new ArrayBuffer(32); new Uint32Array(ub, 0, 2).set([n, dim]); new Float32Array(ub, 16, 2).set([3.0 / axes.sds[0], 3.0 / axes.sds[1]]); d.queue.writeBuffer(uBuf, 0, ub);
  const pipe = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: WGSL_CLOUD }), entryPoint: 'main' } });
  const xyBuf = d.createBuffer({ size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bg = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [uBuf, engine.featBuf, axBuf, xyBuf].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const enc = d.createCommandEncoder(), pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(n / 64)); pass.end();
  const rb = d.createBuffer({ size: n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); enc.copyBufferToBuffer(xyBuf, 0, engine.startBuf, 0, n * 8); enc.copyBufferToBuffer(xyBuf, 0, rb, 0, n * 8); d.queue.submit([enc.finish()]);
  rb.mapAsync(GPUMapMode.READ).then(() => { engine.start.set(new Float32Array(rb.getMappedRange()), 0); rb.destroy(); xyBuf.destroy(); axBuf.destroy(); uBuf.destroy(); });
}
function staticDone() {
  const fitS = (data.fit.ms / 1000).toFixed(1);
  if (UNTIL) status(`Map of the ${data.nBase.toLocaleString()} ${SRC.noun} up to ${tLabelDate(UNTIL)}, fitted in ${fitS} s · Play places the other ${(data.n - data.nBase).toLocaleString()} (Batched RevAdopt) · Full map returns to all ${data.n.toLocaleString()}`);
  else status(`${data.n.toLocaleString()} ${SRC.noun}, one map fitted on this GPU in ${fitS} s${SRC.timeAxis ? ' · drag back along the timeline to fit the map at an earlier date and play forward from there' : ''}`);
}
// the date a rewind would fit up to: the end of the time unit under T (the whole month or year)
function rewindDate(T) { const ms = dateAt(Math.max(0, T)), d = new Date(ms), u = SRC.timeAxis.unit; return (u === 'year' ? Date.UTC(d.getUTCFullYear() + 1, 0) : u === 'month' ? Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1) : Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1)) - 1; }
function tLabelDate(ms) { return SRC.timeAxis ? tLabel(tOfDate(ms)) : new Date(ms).toISOString().slice(0, 10); }
function offerRewind(T) {
  if (!SRC.static || !SRC.timeAxis) return false; const until = rewindDate(T);
  if (until >= SRC.timeAxis.d1 - 1) return false;
  $('fitbar-text').textContent = `Fit the map of the ${SRC.noun} up to ${tLabelDate(until)}, then play the rest onto it?`; $('fitbar').hidden = false; $('fitbar').dataset.until = String(until);
  state.rewindT = tOfDate(until); drawTimeline(); return true;
}
function goUntil(until) { const u = new URL(location.href); if (until) u.searchParams.set('until', String(until)); else u.searchParams.delete('until'); location.href = u.toString(); }
async function freshCollect(minutes, device) {   // minutes may shrink for a source that holds less (the OpenSky relay)
  let dim = SRC.dim || 384, nowUs = Date.now() * 1000, cursorUs = nowUs - minutes * 60e6;
  if (SRC.numeric || SRC.recorded) {   // the relay holds only what it has seen since it started: the window shrinks to that, the base is its first half
    let info; try { info = await (await fetch(SRC.recorded ? SRC.recorded + 'info.json' : `${RELAY}/info`, { cache: 'no-store' })).json(); } catch { throw new Error(`no OpenSky relay at ${RELAY}: run python3 browser_jetstream/relay/opensky_relay.py --out data/jetstream_demo/opensky_NAME`); }
    if (!info.n || info.latest_us - info.earliest_us < 90e6) throw new Error(`the OpenSky relay has ${info.n} samples over ${((info.latest_us - info.earliest_us) / 60e6).toFixed(1)} min; wait for at least three polls`);
    SRC.dim = dim = info.dim; SRC.model = info.model;   // the feature model is the relay's (opensky-state-v2: 48 numbers)
    if (SRC.recorded || info.latest_us < nowUs - 180e6) { nowUs = info.latest_us + 1; cursorUs = nowUs - minutes * 60e6; }   // a recording served offline: its window ends at its last sample
    if (info.earliest_us > cursorUs) { cursorUs = info.earliest_us; minutes = (nowUs - cursorUs) / 60e6; }
  }
  if (HANDOFF) { cursorUs = HANDOFF.t0us; minutes = (nowUs - cursorUs) / 60e6; }   // the handed-over base fixes the window's start and the base's end
  const col = { worker: null, minutes, cursorUs, liveStartUs: nowUs, baseEndUs: HANDOFF ? HANDOFF.baseEndUs : cursorUs + minutes * 30e6, basePosts: [], baseVecs: [], later: [], embedded: 0, fromCache: 0, done: false, axes: null, previewXY: new Float32Array(4096 * 2), previewN: 0, db: null, device, layoutN: 0, refitting: false, refits: 0, refitMs: 0 };
  // 1. the cache: posts of this hour embedded on an earlier visit; the worker then fetches only what came after them
  window.__col = col;   // debugging aid: the collection state while loading
  status('Reading the cache of the last visit…'); const tc = performance.now(); col.db = SRC.numeric || SRC.recorded || SRC.static ? null : await openCache(SOURCE); let cached = [];
  if (col.db) { cached = await readRange(col.db, cursorUs, nowUs); const t1 = performance.now(); prune(col.db, cursorUs - 30 * 60e6).then(() => logEvent('cache pruned', { ms: Math.round(performance.now() - t1) })); }   // expiry runs behind the load, never before it
  // a cache that does not reach back to the cursor (an earlier visit whose backfill was cut short) would leave a hole in the
  // hour that the worker never fills: ignore it and fetch the whole window again
  if (cached.length && cached[0].time_us > cursorUs + 180e6) { logEvent('cache gap', { firstMinAfterCursor: Math.round((cached[0].time_us - cursorUs) / 60e6), posts: cached.length }); cached = []; }
  const resumeUs = cached.length ? cached[cached.length - 1].time_us + 1 : 0; col.fromCache = cached.length; logEvent('cache read', { posts: cached.length, ms: Math.round(performance.now() - tc) });
  $('preview').hidden = false; if (SRC.static) $('map').style.visibility = $('chips').style.visibility = 'hidden';   // a static dataset's map is shown only when finished
  const onFrame = (m) => {
    if (col.done) { col.later.push(m); return; }
    const P = m.hdr.posts; let cut = P.length; for (let i = 0; i < P.length; i++) if (P[i].time_us >= col.baseEndUs) { cut = i; break; }
    // base window: originals only (replies are never used); preview positions as they arrive
    const keep = []; for (let i = 0; i < cut; i++) if (!P[i].reply) keep.push(i);
    if (keep.length) { const v = new Float32Array(keep.length * dim); keep.forEach((i, r) => v.set(m.vecs.subarray(i * dim, (i + 1) * dim), r * dim)); for (const i of keep) col.basePosts.push(P[i]); col.baseVecs.push(v); if (!SRC.static) previewAdd(col, v, keep.length); }
    col.embedded += P.length;
    if (cut < P.length) { col.later.push({ hdr: { ...m.hdr, n: P.length - cut, posts: P.slice(cut) }, vecs: m.vecs.slice(cut * dim) }); col.done = true; return; }
    const t = (P[P.length - 1].time_us - cursorUs) / 1e6;
    status(SRC.static ? `Loading ${SRC.feed}: ${col.basePosts.length.toLocaleString()} ${SRC.noun} in` : `Embedding the last ${minutes} min: ${col.basePosts.length.toLocaleString()} base ${SRC.noun} in, ${mmss(t)} of ${mmss(minutes * 60)}${m.hdr.pendingAfter ? ` (${m.hdr.pendingAfter.toLocaleString()} waiting)` : ''}`);
  };
  // cached records first (they are earlier than anything the worker will send), as frames of 512
  for (let lo = 0; lo < cached.length; lo += 512) {   // frames past the base window go to `later` (the playable half)
    const rec = cached.slice(lo, lo + 512), vecs = new Float32Array(rec.length * dim); rec.forEach((r, i) => vecs.set(new Float32Array(r.vec), i * dim));
    onFrame({ hdr: { n: rec.length, dim, model: 'sentence-transformers/all-MiniLM-L6-v2', encoder: 'minilm', source: 'cache', stream: SOURCE, posts: rec.map(r => ({ uri: r.uri, time_us: r.time_us, did: r.did, rkey: r.rkey, created_at: r.created_at, text: r.text, tags: r.tags, reply: r.reply, embed: r.embed, url: r.url, kind: r.kind, score: r.score, descendants: r.descendants })), cursorUs, liveStartUs: nowUs, pendingAfter: 0, backfilling: true }, vecs });
    await paint();   // let the page paint between batches: the preview is drawn, a preview fit can start
  }
  if (cached.length && col.axes && col.layoutN === 0) { previewRefit(col, true); while (col.refitting) await new Promise(r => setTimeout(r, 50)); }   // at least one preview fit before the final one
  const w = new Worker(SRC.worker || './live_worker.js', { type: 'module' }); state.worker = w; col.worker = w; updateLiveBadge();
  return new Promise((resolve, reject) => {
    if (col.done) { resolve(col); }
    w.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'progress') status(`Loading the encoder… ${m.pct?.toFixed(0) ?? ''} %`);
      else if (m.type === 'note') status(m.msg);
      else if (m.type === 'peek') staticPeek(col, m);
      else if (m.type === 'ready') { if (m.dim) { SRC.dim = dim = m.dim; SRC.model = m.model; SRC.rowsLoaded = m.rows; SRC.rowsTotal = m.total; } if (m.baseEndUs) { tickerStop(); col.baseEndUs = m.baseEndUs; SRC.timeAxis = m.timeAxis; SRC.timeSpan = minutes * 60; SRC.colors = m.colors; SRC.categoryName = m.categoryName; } col.liveStartUs = m.liveStartUs; $('r-live').textContent = `in-browser: ${SRC.feed} direct, ${m.hf} on ${m.device} (loaded in ${(m.loadMs / 1000).toFixed(1)} s)`; status(col.fromCache ? `Fetching ${((nowUs - resumeUs) / 60e6).toFixed(0)} min since the last visit…` : `Fetching the last ${minutes} min…`); }
      else if (m.type === 'frame') { if (col.db) writeFrame(col.db, m.hdr, m.vecs); const was = col.done; onFrame(m); if (!was && col.done) resolve(col); }
      else if (m.type === 'backfilled' && SRC.static && !col.done) { col.done = true; resolve(col); }   // every row is in the base map
      else if (m.type === 'stats') { state.liveStats = { stats: m.stats, device: 'webgpu' }; }
      else if (m.type === 'log') console.log('[live worker]', m.msg);
      else if (m.type === 'error') { w.terminate(); state.worker = null; reject(new Error('In-browser source failed: ' + m.msg)); }
    };
    w.onerror = (e) => { w.terminate(); state.worker = null; reject(new Error('In-browser source failed: ' + (e.message || 'worker error'))); };
    w.postMessage({ type: 'start', source: SOURCE, relay: SRC.static ? SOURCE : SRC.recorded || RELAY, rows: SRC.rows, untilMs: UNTIL, export: params.get('export') === '1' && /^(localhost|127\.0\.0\.1)$/.test(location.hostname), gpu: params.get('gpu') !== '0', backfillMinutes: minutes, cursorUs, liveStartUs: nowUs, resumeUs, baseMinutes: (col.baseEndUs - cursorUs) / 60e6, batchSeconds: 0.25, embedChunk: +(params.get('chunk') || 32), device: params.get('device') || 'webgpu' });
  });
}
// ---- While a static dataset embeds: the vectors coming out (a thinned sample) appear as dots on their first two principal
// axes, the same projection the base fit then starts from, so at the end every dot glides from the cloud into the map. Over
// it, the ticker shows one row at a time being embedded: its tokens (or picture) appear, turn into its vector (384 numbers
// as grey squares) and the strip flies to that row's dot.
// ---- A static dataset's preview fits, the way a live source's base window is refitted as it fills: a UMAP of a sample of
// the rows that have arrived (up to 8,192; 4,096 for long vectors), refitted from its last layout as the sample grows and
// a few more times after, on the GPU, while the whole table's exact neighbour lists are computed. Each refit is tweened
// in (no jumps). The final fit then starts from the preview: every sampled row at its preview place, every other row at
// the mean of its placed neighbours (Jacobi rounds over the exact lists), so the map expands out of what was on screen.
const PV_A = 1.3667583810909143, PV_B = 0.8160247038816811, PV_TWEEN = 450;
async function staticPreviewLoop(col) {
  const P = col.peek, dim = P.dim, S = dim > 100 ? 4096 : 8192; let xy = null, have = 0, still = 0;
  while (!col.previewStop) {
    const n = Math.min(S, P.n);
    if (n < 1000 || (n === have && still >= 6)) { await sleep(150); continue; }   // nothing new and settled: leave the GPU to the final fit
    still = n === have ? still + 1 : 0;
    const init = new Float32Array(n * 2); if (xy) init.set(xy.subarray(0, have * 2)); init.set(col.previewXY.subarray(have * 2, n * 2), have * 2);
    let fit; try { fit = await StreamingEngine.fitBase(col.device, P.vecs.slice(0, n * dim), n, dim, { a: PV_A, b: PV_B, steps: 100, lr: 0.05, negScale: 6, init }); } catch (e) { console.warn('[preview] sample fit failed', e); break; }
    if (col.previewStop) break;
    const from = new Float32Array(n * 2), cur = col.tw ? tweenAt(col) : null; from.set(col.previewXY.subarray(0, n * 2)); if (cur) from.set(cur.subarray(0, Math.min(cur.length, n * 2)));
    col.tw = { from, to: fit.xy, n, t0: performance.now(), ms: PV_TWEEN, fade: !xy }; xy = fit.xy; have = n; col.refits++;
    col.sample = { xy, rows: P.rows.slice(0, n), n }; tweenLoop(col);
    await sleep(PV_TWEEN);
  }
}
function tweenAt(col) { const T = col.tw, e = ease(Math.min(1, (performance.now() - T.t0) / T.ms)), out = new Float32Array(T.n * 2); for (let i = 0; i < T.n * 2; i++) out[i] = T.from[i] + (T.to[i] - T.from[i]) * e; return out; }
function tweenLoop(col) { if (col.twAnim) return; col.twAnim = true; const step = () => { drawTween(col); if (col.tw && !$('preview').hidden && performance.now() - col.tw.t0 < col.tw.ms) requestAnimationFrame(step); else col.twAnim = false; }; requestAnimationFrame(step); }
function drawTween(col) {
  const c = $('preview'), dpr = window.devicePixelRatio || 1, { W, H, scale } = previewMapping(); if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const ctx = c.getContext('2d'), T = col.tw, e = ease(Math.min(1, (performance.now() - T.t0) / T.ms)), s = scale, cx = c.width / 2, cy = c.height / 2, r = 1.2 * dpr;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height);
  if (T.fade && e < 1) { ctx.fillStyle = `rgba(159,176,198,${(0.55 * (1 - e)).toFixed(3)})`; const xy = col.previewXY; for (let i = T.n; i < col.previewN; i++) { const x = cx + xy[2 * i] * s, y = cy - xy[2 * i + 1] * s; ctx.fillRect(x - r, y - r, 2 * r, 2 * r); } }   // the rest of the first cloud fades out
  ctx.fillStyle = 'rgba(159,176,198,0.7)';
  for (let i = 0; i < T.n; i++) { const x = cx + (T.from[2 * i] + (T.to[2 * i] - T.from[2 * i]) * e) * s, y = cy - (T.from[2 * i + 1] + (T.to[2 * i + 1] - T.from[2 * i + 1]) * e) * s; ctx.fillRect(x - r, y - r, 2 * r, 2 * r); }
}
// the final fit's start: sampled rows where the preview shows them, the rest by Jacobi rounds of the mean of their placed
// neighbours on the exact lists, any row still unplaced on the first cloud; a tiny fixed jitter keeps rows from coinciding
async function previewSeed(col, idx, nB, feat, dim) {
  col.previewStop = true; await col.pvLoopP; const smp = col.sample; if (!smp) return null;
  const pos = new Float32Array(nB * 2), has = new Uint8Array(nB), K = 15, jit = (i, k) => { let h = Math.imul((i + 1) ^ (k * 0x9e3779b9), 0x85ebca6b) >>> 0; h ^= h >>> 13; return ((h % 2001) / 1000 - 1) * 0.03; };
  let placed = 0; for (let j = 0; j < smp.n; j++) { const r = smp.rows[j]; if (r < nB && !has[r]) { pos[2 * r] = smp.xy[2 * j]; pos[2 * r + 1] = smp.xy[2 * j + 1]; has[r] = 1; placed++; } }
  for (let round = 0; round < 60; round++) {
    const add = []; for (let i = 0; i < nB; i++) { if (has[i]) continue; let x = 0, y = 0, c = 0; for (let k = 0; k < K; k++) { const j = idx[i * K + k]; if (j >= 0 && has[j]) { x += pos[2 * j]; y += pos[2 * j + 1]; c++; } } if (c) add.push(i, x / c, y / c); }
    if (!add.length) break; for (let a = 0; a < add.length; a += 3) { const i = add[a]; pos[2 * i] = add[a + 1] + jit(i, 1); pos[2 * i + 1] = add[a + 2] + jit(i, 2); has[i] = 1; placed++; }
    if (round % 4 === 3) await sleep(0);
  }
  for (let i = 0; i < nB; i++) if (!has[i]) { const p = pcaProject(feat.subarray(i * dim, (i + 1) * dim), 1, dim, col.axes, 3.0); pos[2 * i] = p[0]; pos[2 * i + 1] = p[1]; }
  col.seeded = { sampled: smp.n, placed, ofRows: nB }; return pos;
}
function staticPeek(col, m) {
  const dim = m.dim, add = m.vecs.length / dim; if (!add) return;
  if (!col.peek) col.peek = { vecs: new Float32Array(Math.max(4096, add) * dim), rows: new Uint32Array(Math.max(4096, add)), n: 0, dim };
  const P = col.peek; if ((P.n + add) * dim > P.vecs.length) { const g = new Float32Array(Math.max(P.vecs.length * 2, (P.n + add) * dim)); g.set(P.vecs); P.vecs = g; }
  if (P.n + add > P.rows.length) { const g = new Uint32Array(Math.max(P.rows.length * 2, P.n + add)); g.set(P.rows); P.rows = g; }
  P.vecs.set(m.vecs, P.n * dim); if (m.rows) P.rows.set(m.rows, P.n); P.n += add;
  if (col.previewXY.length < P.n * 2) { const g = new Float32Array(Math.max(col.previewXY.length * 2, P.n * 2)); g.set(col.previewXY); col.previewXY = g; }
  if (!col.axes) { if (P.n < 512) return; col.axes = pcaAxes(P.vecs, Math.min(P.n, 1024), dim); pcaProject(P.vecs, P.n, dim, col.axes, 3.0, col.previewXY); col.previewN = P.n; }
  else { col.previewXY.set(pcaProject(m.vecs, add, dim, col.axes, 3.0), col.previewN * 2); col.previewN += add; }
  drawPreview(col); if (!col.pvLoopP && P.n >= 1000 && m.rows) col.pvLoopP = staticPreviewLoop(col);
  for (const sm of m.samples || []) ticker.queue.push({ ...sm, xy: pcaProject(sm.vec, 1, dim, col.axes, 3.0) });
  if (ticker.queue.length > 6) ticker.queue.splice(0, ticker.queue.length - 6);   // keep up: only the newest few wait
  if (!ticker.busy) tickerNext();
}
const ticker = { queue: [], busy: false, on: true };
window.__ticker = { ticker, next: () => tickerNext(), strip: stripCanvas };   // debugging aid: replay a row through the ticker
function tickerStop() { ticker.on = false; ticker.queue.length = 0; const el = $('ticker'); if (el) el.replaceChildren(); }
function stripCanvas(vec) {   // 384 numbers as 4 rows of 96 grey squares (scaled by sqrt(dim): a unit vector's numbers are about 1/sqrt(dim))
  const dim = vec.length, cols = Math.ceil(dim / 4), q = 4, c = document.createElement('canvas'); c.width = cols * q; c.height = 4 * q; c.className = 'strip';
  const ctx = c.getContext('2d'), k = Math.sqrt(dim) / 2.5;
  for (let i = 0; i < dim; i++) { const g = Math.round(255 * Math.max(0, Math.min(1, 0.5 + vec[i] * k * 0.5))); ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect((i % cols) * q, Math.floor(i / cols) * q, q - 0.5, q - 0.5); }
  return c;
}
function tickerNext() {
  const el = $('ticker'), sm = ticker.queue.shift(); if (!el || !sm || !ticker.on) { ticker.busy = false; return; }
  ticker.busy = true; el.replaceChildren();
  const card = document.createElement('div'); card.className = 'tick'; el.appendChild(card);
  const src = document.createElement('div'); src.className = 'src'; card.appendChild(src);
  if (sm.img) { const im = new Image(); im.src = sm.img; im.className = 'pic'; src.appendChild(im); }
  else sm.tokens.forEach((t, i) => { const sp = document.createElement('span'); sp.textContent = t; sp.className = t.startsWith('##') ? 'tok sub' : 'tok'; sp.animate([{ opacity: 0, transform: 'translateY(-5px)' }, { opacity: 1, transform: 'none' }], { duration: 160, delay: i * 18, fill: 'backwards' }); src.appendChild(sp); });
  const strip = stripCanvas(sm.vec); card.appendChild(strip);
  const T1 = 380, T2 = 300, T3 = 420;   // tokens in, tokens become the vector, the vector flies to its dot
  src.animate([{ opacity: 1 }, { opacity: 1, offset: 0.55 }, { opacity: 0, transform: 'scaleY(0.4)' }], { duration: T1 + T2, fill: 'forwards' });
  strip.animate([{ opacity: 0, transform: 'scaleX(0.3)' }, { opacity: 0, transform: 'scaleX(0.3)', offset: T1 / (T1 + T2) }, { opacity: 1, transform: 'none' }], { duration: T1 + T2, fill: 'forwards' });
  setTimeout(() => {
    if (!ticker.on) { ticker.busy = false; return; }
    const pv = $('preview'), pr = pv.getBoundingClientRect(), sr = strip.getBoundingClientRect(), { scale } = previewMapping(), dpr = window.devicePixelRatio || 1;
    const tx = pr.left + pr.width / 2 + sm.xy[0] * scale / dpr, ty = pr.top + pr.height / 2 - sm.xy[1] * scale / dpr, dx = tx - (sr.left + sr.width / 2), dy = ty - (sr.top + sr.height / 2);
    strip.animate([{ transform: 'none', opacity: 1 }, { transform: `translate(${dx}px, ${dy}px) scale(0.03)`, opacity: 0.9 }], { duration: T3, easing: 'cubic-bezier(.5,0,.8,.4)', fill: 'forwards' }).onfinish = () => {
      const dot = document.createElement('div'); dot.className = 'ping'; dot.style.left = `${tx - pr.left}px`; dot.style.top = `${ty - pr.top}px`; el.appendChild(dot);
      dot.animate([{ transform: 'translate(-50%,-50%) scale(0.3)', opacity: 1 }, { transform: 'translate(-50%,-50%) scale(2.2)', opacity: 0 }], { duration: 500 }).onfinish = () => dot.remove();
      card.remove(); tickerNext();
    };
  }, T1 + T2);
}
// Preview layout: new rows get a principal-axes position; from 1,000 posts on, a short warm-started fit of everything
// so far (150 steps, old rows from their last positions) runs whenever at least 256 new rows have arrived and the
// previous one has finished, so the cloud takes its final shape early and the final fit only refines it.
function previewRefit(col, force = false) {
  if (HANDOFF || SRC.static) return;   // the base comes ready-made; a static dataset arrives all at once
  const have = col.basePosts.length; if (col.refitting || (col.done && !force) || !col.device || have < 1000 || have - col.layoutN < 256) return;
  col.refitting = true; const dim = SRC.dim || 384, n = have, t0 = performance.now();
  const all = new Float32Array(n * dim); { let off = 0; for (const b of col.baseVecs) { if (off + b.length > all.length) break; all.set(b, off); off += b.length; } }
  const init = col.previewXY.slice(0, n * 2);
  StreamingEngine.fitBase(col.device, all, n, dim, { a: 1.3667583810909143, b: 0.8160247038816811, steps: 150, lr: 0.05, negScale: 6, init }).then((fit) => {
    if (col.done && !force) return; col.previewXY.set(fit.xy, 0); col.layoutN = n; col.refits++; col.refitMs += performance.now() - t0; drawPreview(col);
  }).catch((e) => console.warn('[preview] refit failed', e)).finally(() => { col.refitting = false; previewRefit(col); });
}
function previewAdd(col, v, n) {
  const dim = SRC.dim || 384; if (col.previewN + n > col.previewXY.length / 2) { const g = new Float32Array(Math.max(col.previewXY.length * 2, (col.previewN + n) * 2)); g.set(col.previewXY); col.previewXY = g; }
  if (!col.axes) { const have = col.basePosts.length; if (have < 1024 && have < 200) return; const all = new Float32Array(have * dim); let off = 0; for (const b of col.baseVecs) { all.set(b, off); off += b.length; } col.axes = pcaAxes(all, Math.min(have, 1024), dim); pcaProject(all, have, dim, col.axes, 3.0, col.previewXY); col.previewN = have; }
  else { const y = pcaProject(v, n, dim, col.axes, 3.0); col.previewXY.set(y, col.previewN * 2); col.previewN += n; }
  drawPreview(col); previewRefit(col);
}
// Build the data object loadData would have produced, from the collected base window: originals only, exact lists and
// a UMAP layout fitted on this GPU, k-means topic clusters named by their distinctive words.
async function freshBuild(device, col, cap) {
  if (HANDOFF) return handoffBuild(device, col, cap);
  const dim = SRC.dim || 384, A = 1.3667583810909143, B = 0.8160247038816811;   // spread 1.2, min_dist 0.03 (the builder's constants)
  const all = col.basePosts; let orig = []; for (let i = 0; i < all.length; i++) if (!all[i].reply) orig.push(i);
  if (SRC.numeric) { const last = new Map(); for (const i of orig) last.set(all[i].icao24 || i, i); orig = [...last.values()].sort((a, b) => a - b); }   // one sample per aircraft: its latest in the base window
  const nBase = orig.length; if (nBase < 200) throw new Error(`only ${nBase} original posts in the base window; try a longer ?fresh=`);
  const feat = new Float32Array(nBase * dim); { let off = 0; const flat = new Float32Array(all.length * dim); for (const v of col.baseVecs) { flat.set(v, off); off += v.length; } orig.forEach((i, r) => feat.set(flat.subarray(i * dim, (i + 1) * dim), r * dim)); }
  status(SRC.static ? `Fitting the map: ${nBase.toLocaleString()} ${SRC.noun}${UNTIL ? ` up to ${tLabelDate(UNTIL)}` : ''}…` : `Fitting the base map: ${nBase.toLocaleString()} ${SRC.noun}…`); await new Promise(r => setTimeout(r, 30));
  while (col.refitting) await new Promise(r => setTimeout(r, 50));
  let init = null; if (SRC.static && col.pvLoopP) init = (idx, nB) => previewSeed(col, idx, nB, feat, dim); else if (col.axes && col.previewN === all.length) { init = new Float32Array(nBase * 2); orig.forEach((i, r) => { init[2 * r] = col.previewXY[2 * i]; init[2 * r + 1] = col.previewXY[2 * i + 1]; }); } else if (col.axes) { init = pcaProject(feat, nBase, dim, col.axes, 3.0); }
  let fit;
  if (TUNED_ERR) throw TUNED_ERR;
  if (TUNED) {   // the live map's lists for placement (fitBase with no layout steps), then the tuner's layout
    const lists = await StreamingEngine.fitBase(device, feat, nBase, dim, { a: A, b: B, steps: 0, init: new Float32Array(nBase * 2), onProgress: (m) => status(`Fitting the base map (${m})…`) });
    const t = await tunedLayout(device, feat, nBase, dim, TUNED, (m) => status(`Fitting the tuned base map (${m})…`));
    fit = { ...lists, xy: t.xy, steps: TUNED.epochs, edges: t.edges, ms: lists.ms + t.ms, tuned: tunedToParam(TUNED) };
  } else fit = await StreamingEngine.fitBase(device, feat, nBase, dim, { a: A, b: B, steps: 1500, lr: init ? 0.05 : 0.1, negScale: 6, init, onProgress: (m) => status(SRC.static ? `Fitting the map of ${nBase.toLocaleString()} ${SRC.noun} (${m.replace('base fit: ', '')})…` : `Fitting the base map (${m})…`) });
  const basePosts = orig.map(i => ({ text: all[i].text, terms: all[i].terms, icao24: all[i].icao24, tags: all[i].tags || [], reply: 0, url: all[i].url, kind: all[i].kind, img: all[i].img, cat: all[i].cat, score: all[i].score, descendants: all[i].descendants }));
  let lab, named;
  if (SRC.static && all.some((p) => p.cat !== undefined)) {
    // the dataset's own label (country, anomaly label, subject...) is each row's topic, as Embedding Atlas colours its examples:
    // every label when the dataset brings its colours (ASRS: the aviation dashboard's), otherwise the 24 commonest and "other"
    const counts = new Map(), bump = (c) => counts.set(c, (counts.get(c) || 0) + 1);
    for (const p of all) bump(p.cat); for (const m of col.later) for (const p of m.hdr.posts) bump(p.cat);
    const ranked = [...counts].sort((a, b) => b[1] - a[1]).map(([c]) => c), keep = SRC.colors ? ranked : ranked.slice(0, 24), rest = ranked.length - keep.length;
    const names = keep.slice(); if (rest) names.push(`(other ${rest})`);
    const colors = {}; names.forEach((nm, i) => { colors[nm] = SRC.colors?.[nm] || (i < keep.length ? PALETTE[i % PALETTE.length] : '#8c99ad'); });
    col.catIndex = new Map(keep.map((c, i) => [c, i])); col.catOther = rest ? keep.length : 0;
    lab = new Uint16Array(nBase); orig.forEach((i, r) => { lab[r] = col.catIndex.get(all[i].cat) ?? col.catOther; });
    named = { names, colors, terms: names.map(() => []) };
  } else {
    status('Topic clusters of the base map…'); await new Promise(r => setTimeout(r, 30));
    const kc = Math.min(20, Math.max(2, Math.floor(nBase / 20))); lab = await kmeans(feat, nBase, dim, kc, { iters: 15, seed: 1, onProgress: (it, total) => status(`Topic clusters of the base map… pass ${it} of ${total}`) });
    named = nameClusters(basePosts, lab, kc);
  }
  const N = nBase + cap, featS = new Float32Array(N * dim); featS.set(feat);
  const tS = new Float32Array(N).fill(Infinity), wallS = new Float32Array(N), replyS = new Uint8Array(N), labelS = new Uint16Array(N).fill(65535); labelS.set(lab);
  orig.forEach((i, r) => { const t = (all[i].time_us - col.cursorUs) / 1e6; tS[r] = t; wallS[r] = t; });
  const baseSeconds = SRC.static ? Math.min(col.minutes * 60, (col.baseEndUs - col.cursorUs) / 1e6) : col.minutes * 30, fetchedAt = new Date(col.liveStartUs / 1000);
  const baseMeta = { n_base: nBase, base_seconds: baseSeconds, originals_only: true, dropped_base_window_replies: all.length - nBase, k: 15, a: A, b: B, dim, model: SRC.model || 'sentence-transformers/all-MiniLM-L6-v2', pooling: SRC.numeric ? 'none' : 'mean', encoder: SRC.model || 'minilm', fit: fit.tuned ? 'tuned' : 'browser', ...(fit.tuned ? { tuned: fit.tuned } : {}), fit_steps: fit.steps, fit_ms: fit.ms, edges: fit.edges, labels: named.names, label_colors: named.colors, label_terms: named.terms };
  const rowsMeta = { n: nBase, duration_seconds: col.minutes * 60, t0_us: col.cursorUs, encoder: baseMeta.encoder, bases: ['fresh'], fetched_at: fetchedAt.toISOString() };
  return { N, n: nBase, nCapture: nBase, dim, nBase, encoder: baseMeta.encoder, model: baseMeta.model, k: 15, a: A, b: B, baseName: 'fresh', baseMeta, rowsMeta, t0us: col.cursorUs,
           labels: named.names, labelColors: named.colors, labelTerms: named.terms, baseSeconds, tS, wallS, labelS, replyS, posts: basePosts, featS,
           baseXY: fit.xy, bIdx: fit.idx, bDist: fit.dist, bRho: fit.rho, bSigma: fit.sigma, fit };
}
// The base handed over by the tuner: its rows, positions and topic clusters as they were there; this page only recomputes
// the 15-neighbour lists, rho and sigma its placement and rebuilds read (fitBase with no layout steps, same vectors).
async function handoffBuild(device, col, cap) {
  const H = HANDOFF, dim = H.dim, nBase = H.nBase, A = 1.3667583810909143, B = 0.8160247038816811;
  status(`Base map from the tuner: ${nBase.toLocaleString()} ${SRC.noun}, neighbour lists…`); await new Promise(r => setTimeout(r, 30));
  const lists = await StreamingEngine.fitBase(device, H.feat, nBase, dim, { a: A, b: B, steps: 0, init: new Float32Array(nBase * 2), onProgress: (m) => status(`Base map from the tuner (${m})…`) });
  const fit = { ...lists, xy: Float32Array.from(H.xy), steps: 0, ms: lists.ms, handoff: H.fit || { kind: 'handoff' } };
  const N = nBase + cap, featS = new Float32Array(N * dim); featS.set(H.feat);
  const tS = new Float32Array(N).fill(Infinity), wallS = new Float32Array(N), replyS = new Uint8Array(N), labelS = new Uint16Array(N).fill(65535); labelS.set(H.labelS);
  for (let r = 0; r < nBase; r++) { const t = (H.timesUs[r] - col.cursorUs) / 1e6; tS[r] = t; wallS[r] = t; }
  const baseSeconds = (H.baseEndUs - H.t0us) / 1e6, fetchedAt = new Date(col.liveStartUs / 1000);
  const baseMeta = { n_base: nBase, base_seconds: baseSeconds, originals_only: true, dropped_base_window_replies: 0, k: 15, a: A, b: B, dim, model: H.model, pooling: SRC.numeric ? 'none' : 'mean', encoder: H.model, fit: 'handoff', handoff: fit.handoff, fit_steps: 0, fit_ms: fit.ms, edges: fit.edges, labels: H.labels, label_colors: H.labelColors, label_terms: H.labelTerms };
  const rowsMeta = { n: nBase, duration_seconds: col.minutes * 60, t0_us: col.cursorUs, encoder: H.model, bases: ['handoff'], fetched_at: fetchedAt.toISOString() };
  return { N, n: nBase, nCapture: nBase, dim, nBase, encoder: H.model, model: H.model, k: 15, a: A, b: B, baseName: 'handoff', baseMeta, rowsMeta, t0us: col.cursorUs,
           labels: H.labels, labelColors: H.labelColors, labelTerms: H.labelTerms, baseSeconds, tS, wallS, labelS, replyS, posts: H.posts.slice(), featS,
           baseXY: fit.xy, bIdx: fit.idx, bDist: fit.dist, bRho: fit.rho, bSigma: fit.sigma, fit };
}
// The live map as the tuner reads it (handoff.js): the base rows at the positions on screen now, every later row loaded
// so far with its vector (so coming back needs no download or embedding), and the view (camera, canvas place, dot size)
// so the tuner opens on exactly what is shown.
function handoffOf() {
  const n = data.nBase, dim = data.dim, m = data.n, timesUs = new Float64Array(n);   // every loaded row, placed or still waiting for (let r = 0; r < n; r++) timesUs[r] = data.t0us + data.tS[r] * 1e6;
  const later = []; for (let r = n; r < m; r++) { const p = data.posts[r]; if (p) later.push({ ...p, time_us: Math.round(data.t0us + data.wallS[r] * 1e6) }); }
  const rect = renderer.canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  return { source: SOURCE, recKey: RECORDED ? REC_KEY : null, t0us: data.t0us, baseEndUs: data.t0us + data.baseSeconds * 1e6, nBase: n, dim, model: data.model, posts: data.posts.slice(0, n), timesUs,
           feat: data.featS.slice(0, n * dim), xy: engine.pos.slice(0, n * 2), labelS: data.labelS.slice(0, n), labels: data.labels, labelColors: data.labelColors, labelTerms: data.labelTerms,
           later: { posts: later, feat: data.featS.slice(n * dim, (n + later.length) * dim), endUs: later.length ? later[later.length - 1].time_us : data.t0us + data.baseSeconds * 1e6, liveStartUs: state.windowEndUs || null },
           view: { cx: renderer.view.cx, cy: renderer.view.cy, scale: renderer.view.scale, rect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height }, dpr, pointPx: renderer.pointPx * dpr },
           fit: data.baseMeta.handoff || (data.fit?.tuned ? { kind: 'tuned settings', settings: data.fit.tuned } : null) };
}
// a view handed over (camera centre and scale for a canvas at another place on the page): the same world point lands on the same page pixel
function viewFrom(v, rect) {
  const dpr = window.devicePixelRatio || 1, s = v.scale * dpr / v.dpr;
  const dx = (rect.left + rect.width / 2) - (v.rect.left + v.rect.width / 2), dy = (rect.top + rect.height / 2) - (v.rect.top + v.rect.height / 2);
  return { cx: v.cx + dx * dpr / s, cy: v.cy - dy * dpr / s, scale: s };
}
// The window handed back by the tuner: no download, no embedding. A recording is complete; a live source resumes after
// the last post it held, so only newer posts are fetched (and embedded) when it catches up.
async function handoffCollect(device) {
  const H = HANDOFF, L = H.later, dim = H.dim, nowUs = RECORDED ? (L.liveStartUs || L.endUs + 1) : Date.now() * 1000;
  const col = { worker: null, minutes: (nowUs - H.t0us) / 60e6, cursorUs: H.t0us, liveStartUs: nowUs, baseEndUs: H.baseEndUs, basePosts: [], baseVecs: [], later: [], embedded: 0, fromCache: 0, done: true, axes: null, previewXY: new Float32Array(2), previewN: 0, db: null, device, layoutN: 0, refitting: false, refits: 0, refitMs: 0, handedBack: true };
  for (let lo = 0; lo < L.posts.length; lo += 2048) { const P = L.posts.slice(lo, lo + 2048); col.later.push({ hdr: { n: P.length, dim, model: H.model, encoder: H.model, source: 'handoff', stream: SOURCE, posts: P }, vecs: L.feat.slice(lo * dim, (lo + P.length) * dim) }); }
  if (!RECORDED && !SRC.numeric) {   // fetch only what is newer than the window held
    col.db = await openCache(SOURCE);
    const w = new Worker(SRC.worker || './live_worker.js', { type: 'module' }); col.worker = w; state.worker = w;
    col.startMsg = ({ type: 'start', source: SOURCE, relay: RELAY, backfillMinutes: col.minutes, cursorUs: H.t0us, liveStartUs: nowUs, resumeUs: L.endUs + 1, baseMinutes: (H.baseEndUs - H.t0us) / 60e6, batchSeconds: 0.25, embedChunk: +(params.get('chunk') || 32), device: params.get('device') || 'webgpu' });   // posted once the page listens (attachWorker)
  }
  return col;
}
function goLiveRelay() {
  const ws = new WebSocket(`ws://127.0.0.1:${LIVE_PORT}`); ws.binaryType = 'arraybuffer'; state.ws = ws; updateLiveBadge();
  ws.onopen = () => { $('r-live').textContent = `local relay on port ${LIVE_PORT}`; liveOn('Live from the relay'); };
  ws.onmessage = (ev) => {
    const buf = ev.data, dv = new DataView(buf), hl = dv.getUint32(0, true), hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    ingestFrame(hdr, new Float32Array(buf.slice(4 + hl)), (why) => { state.liveRefused = why; ws.close(); });
  };
  ws.onclose = () => { const was = state.live, refused = state.liveRefused; state.liveRefused = null; liveOff(refused || (was ? 'Left the live feed. Press Go live to reconnect.' : `No relay on ws://127.0.0.1:${LIVE_PORT}. Start jetstream_relay.py with --serve ${LIVE_PORT}, then press Go live again.`), !!refused || !was); };
  ws.onerror = () => {};
}
// Parity check: embed n recorded posts in the browser and compare with the relay's torch vectors already loaded.
function parity(n = 200) {
  return new Promise((resolve, reject) => {
    const w = new Worker('./live_worker.js', { type: 'module' }); const idx = []; const step = Math.max(1, Math.floor(data.n / n)); for (let i = 0; i < data.n && idx.length < n; i += step) idx.push(i);
    w.onmessage = (ev) => { const m = ev.data; if (m.type === 'error') { w.terminate(); reject(new Error(m.msg)); } if (m.type !== 'parity') return;
      const cos = idx.map((r, j) => { let d = 0; for (let k = 0; k < data.dim; k++) d += data.featS[r * data.dim + k] * m.vecs[j * data.dim + k]; return d; }).sort((a, b) => a - b);
      w.terminate(); resolve({ n: idx.length, device: m.device, model: data.model, cosMin: cos[0], cosP5: cos[Math.floor(0.05 * cos.length)], cosMedian: cos[cos.length >> 1], cosMean: cos.reduce((a, b) => a + b, 0) / cos.length }); };
    w.postMessage({ type: 'parity', texts: idx.map(i => data.posts[i].text), device: params.get('device') || 'webgpu' });
  });
}

// The dataset menu works from the first moment, not only once the map is built: picking another source mid-build reloads
// the page on it, which abandons the build (and a feed that never answers).
function wireSource() {
  // 3D: cast the map into three dimensions and back at any time (cast3d.js); the 2D live map keeps running underneath
  $('view3d')?.addEventListener('click', async (e) => {
    e.preventDefault(); if (!data?.baseXY || !engine || state.castToggling) return;
    const a = e.currentTarget; if (tuner?.active) { status('Finish tuning first (Done tuning), then cast into 3D.'); return; }
    state.castToggling = true; a.setAttribute('aria-busy', 'true');
    try {
      if (!cast) cast = new Cast3D({ device: engine.device, engine, data, renderer, state, canvas: $('map3d'), gizmoEl: $('gizmo3d'), status, staticMap: !!SRC.static });
      if (!cast.on) { $('chips').hidden = true; a.textContent = '2D'; await cast.enter(); }
      else {
        const { changed } = await cast.exit(); $('chips').hidden = false; a.textContent = '3D';
        if (changed) {   // the 2D map is now the 3D map as it was seen: regions, map lists and readouts follow (one quiet rebuild if anything has arrived)
          data.baseMeta.handoff = { kind: 'flattened from the 3D view', settings: tuner?.P3 ? tunedToParam(tuner.P3) : null }; if (!/tuned base/.test($('h1sub').textContent)) $('h1sub').textContent += ' · tuned base';
          render(); if (engine.n > data.nBase) await doRebuild('3D view'); else { buildRegions(); render(); }
          status('Back in 2D: the map as it was seen in 3D');
        } else { render(); status('Back in 2D'); }
      }
    } catch (err) { console.error(err); status(`3D failed: ${err.message || err}`, true); $('map3d').hidden = true; $('chips').hidden = false; a.textContent = '3D'; if (cast) cast.on = false; }
    finally { state.castToggling = false; a.removeAttribute('aria-busy'); }
  });
  // Base map tuner: a mode of this page, on this map (tunemode.js). The button toggles it; the map never leaves the screen.
  $('tunerLink')?.addEventListener('click', async (e) => {
    e.preventDefault(); if (!data?.baseXY || !engine || state.tuneToggling || state.castToggling) return;
    const a = e.currentTarget;   // before any await: the event's currentTarget is cleared once dispatch ends
    state.tuneToggling = true; a.setAttribute('aria-busy', 'true');
    try {
      if (!tuner) tuner = new TuneMode({ device: engine.device, engine, data, renderer, state, $, drawFrame, status, staticMap: !!SRC.static, get cast() { return cast; }, rebuild: (why) => doRebuild(why),
        rebuildRegions: () => { buildRegions(); render(); }, shrinkRegions: () => shrinkRegions() });
      if (!tuner.active) { if (renderer.mode === 6 && !cast?.on) { state.modeBeforeTune = 6; renderer.mode = 0; } await tuner.enter(); a.textContent = 'Done tuning'; }   // dots coloured by region would turn grey with no regions: topic colours while tuning
      else { const was3 = !!tuner.d3; if (state.modeBeforeTune != null) { renderer.mode = state.modeBeforeTune; state.modeBeforeTune = null; } await tuner.exit(); tuner.hide(); a.textContent = 'Base map tuner'; if (was3) { status('3D tuning done · 2D keeps it'); return; } if (tuner.everMoved) { data.baseMeta.handoff = { kind: 'tuned in the live map', settings: tunedToParam(tuner.P) + `_rs${tuner.P.restart}` }; if (!/tuned base/.test($('h1sub').textContent)) $('h1sub').textContent += ' · tuned base'; } status('Base map tuned'); }
    } catch (err) { console.error(err); status(`Tuning failed: ${err.message || err}`, true); }
    finally { state.tuneToggling = false; a.removeAttribute('aria-busy'); }
  });
  if (!RECORDED) for (const [k, D] of Object.entries(DATASETS)) $('srcDatasets').appendChild(new Option(D.label, k));
  if (RECORDED) { const sel = $('livesrc'); sel.innerHTML = ''; for (const [k, r] of Object.entries(RECORDED.sources)) sel.add(new Option(r.label, k, false, k === REC_KEY)); sel.addEventListener('change', () => { location.hash = sel.value; }); sel.closest('.ctl').querySelector('.lbl').textContent = 'Recording'; $('golive').hidden = true; }
  else { $('livesrc').value = state.liveSource; } if (SRC.static) { for (const id of ['golive', 'play', 'replay30']) $(id).hidden = true; $('speed').closest('.ctl')?.setAttribute('hidden', ''); if (UNTIL) $('play').hidden = $('fullmap').hidden = false; }   /* nothing to play while it loads; staticReady shows Replay for a dated table */ if (!RECORDED) $('livesrc').addEventListener('change', (e) => { const v = e.target.value; if (FRESH && v !== 'relay') { const u = new URL(location.href); if (v === 'browser') u.searchParams.delete('source'); else u.searchParams.set('source', v); for (const q of ['fresh', 'base', 'tuned', 'rows', 'cap', 'until']) u.searchParams.delete(q); location.href = u.toString(); return; } state.liveSource = v === 'relay' ? 'relay' : 'browser'; });
  sourcePicker();
}
function wire() {
  const c = $('map'); let drag = null, moved = false;
  c.addEventListener('click', (e) => { if (!moved) inspectAt(e.clientX, e.clientY); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { clearInspect(); if (rstate.focus >= 0) setFocus(rstate.focus); } });
  $('inspect-clear').addEventListener('click', clearInspect);
  $('region-clear').addEventListener('click', () => { if (rstate.focus >= 0) setFocus(rstate.focus); });
  { const box = $('heatPals'), pick = (name, quiet) => { if (!HEAT_PALETTES[name]) name = 'ember'; renderer.setHeatPalette(name); $('heatPalv').textContent = name; for (const b of box.children) b.classList.toggle('on', b.dataset.pal === name); try { localStorage.setItem('jet.heatPal', name); } catch {} if (!quiet) { layoutChips(); legend(); render(); } };
    for (const [name, st] of Object.entries(HEAT_PALETTES)) { const b = document.createElement('button'); b.dataset.pal = name; b.setAttribute('aria-label', name); b.style.background = `linear-gradient(90deg,${st.map(c => `rgb(${c.map(v => Math.round(255 * v)).join(',')})`).join(',')})`; b.addEventListener('click', () => pick(name)); box.appendChild(b); }
    let saved = 'ember'; try { saved = localStorage.getItem('jet.heatPal') || 'ember'; } catch {} pick(saved, true); }
  $('heat').addEventListener('change', (e) => { state.heat = e.target.value; state.heatEma = null; regionRates(); legend(); render(); });   // a new mode starts from its own values, not smoothed from the old
  $('heatWin').addEventListener('input', (e) => { state.rateWin = +e.target.value; $('heatWinv').textContent = e.target.value; regionRates(); legend(); render(); if (rstate.focus >= 0) showRegion(); });
  try { const saved = JSON.parse(localStorage.getItem('jet.auto') || 'null'); if (saved) for (const k of Object.keys(look.auto)) if (typeof saved[k] === 'boolean') look.auto[k] = saved[k]; } catch {}
  for (const k of Object.keys(look.auto)) setAuto(k, look.auto[k]);
  for (const b of document.querySelectorAll('.pill[data-auto]')) b.addEventListener('click', () => { setAuto(b.dataset.auto, !look.auto[b.dataset.auto]); legend(); render(); });
  try { setNsfwFilter(localStorage.getItem('jet.nsfw') !== '0'); } catch {} $('nsfwOn').checked = nsfwFilterOn(); $('nsfwOn').addEventListener('change', (e) => { setNsfwFilter(e.target.checked); try { localStorage.setItem('jet.nsfw', e.target.checked ? '1' : '0'); } catch {} data.docs = null; relabel(); buildRegions(); legend(); render(); if (state.inspect) showInspect(); });
  // collapsed groups remember whether they were open (per browser; nothing leaves the page)
  for (const d of document.querySelectorAll('details[data-remember]')) { try { const v = localStorage.getItem('jet.panel.' + d.id); if (v !== null) d.open = v === '1'; } catch {} d.addEventListener('toggle', () => { try { localStorage.setItem('jet.panel.' + d.id, d.open ? '1' : '0'); } catch {} }); }
  for (const id of ['im-both', 'im-neighbours', 'im-adopters']) $(id).addEventListener('change', (e) => { if (e.target.checked) { state.inspectMode = e.target.value; showInspect(); render(); } });
  const bs = $('base'); (data.rowsMeta.bases || [BASE]).forEach(b => { const o = document.createElement('option'); o.value = b; o.textContent = b.replace('base_', '').replace(/^(\d+)min/, 'first $1 min').replace(/^(\d+)posts/, 'first $1 posts').replace(/_originals\b/, ' · originals only').replace(/_(bge|gte)-small/, ' · $1-small').replace(/_minilm\b/, ' · minilm').replace(/_/g, ' · '); bs.appendChild(o); }); bs.value = BASE;
  bs.addEventListener('change', (e) => { const u = new URL(location.href); u.searchParams.set('base', e.target.value); location.href = u.toString(); });
  c.addEventListener('wheel', (e) => { e.preventDefault(); state.viewTouched = true; const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; const mx = (e.clientX - r.left) * dpr - c.width / 2, my = -((e.clientY - r.top) * dpr - c.height / 2);
    const f = Math.exp(-e.deltaY * 0.0015), v = renderer.view; const wx = v.cx + mx / v.scale, wy = v.cy + my / v.scale; v.scale *= f; v.cx = wx - mx / v.scale; v.cy = wy - my / v.scale; render(); }, { passive: false });
  c.addEventListener('pointerdown', (e) => { state.viewTouched = true; drag = { x: e.clientX, y: e.clientY }; moved = false; c.setPointerCapture(e.pointerId); });
  c.addEventListener('pointermove', (e) => { if (!drag) return; if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) moved = true; const dpr = window.devicePixelRatio || 1, v = renderer.view; v.cx -= (e.clientX - drag.x) * dpr / v.scale; v.cy += (e.clientY - drag.y) * dpr / v.scale; drag = { x: e.clientX, y: e.clientY }; render(); });
  c.addEventListener('pointerup', () => { drag = null; }); c.addEventListener('dblclick', () => { renderer.fitTo(data.baseXY); render(); });
  // footer canvases: a click ahead of the play-head runs the stream to that bin; a click or drag behind it rewinds the view
  for (const el of [$('timeline'), $('lineage')]) {
    let scrub = null; const scale = () => (el.id === 'lineage' ? lineageScale : timeScale)(el.clientWidth), tAt = (ev) => { const r = el.getBoundingClientRect(); return scale().tOf((ev.clientX - r.left) / r.width * el.clientWidth); };
    if (el.id === 'lineage') { el.addEventListener('pointermove', (ev) => { if (!scrub) showLoupe(el, ev); else $('loupe').hidden = true; }); el.addEventListener('pointerleave', () => { $('loupe').hidden = true; }); }
    el.addEventListener('pointerdown', (ev) => { scrub = { x0: ev.clientX, moved: false }; el.setPointerCapture(ev.pointerId); });
    // a static map: back before what it holds offers a fit at that date (the map before then never existed as a layout)
    const rewinds = (T) => SRC.static && SRC.timeAxis && el.id === 'timeline' && T < data.baseSeconds - 1e-3;
    el.addEventListener('pointermove', (ev) => { if (!scrub) return; if (Math.abs(ev.clientX - scrub.x0) > 2) scrub.moved = true; if (!scrub.moved) return; const T = tAt(ev); if (rewinds(T)) { if (view.on) viewExit(); offerRewind(T); return; } if (T < state.clock) viewTo(T, SCRUB_GLIDE_MS); else if (view.on) viewFinish(); });
    el.addEventListener('pointerup', (ev) => { const was = scrub; scrub = null; if (!was) return; const T = tAt(ev);
      if (rewinds(T)) { offerRewind(T); return; }
      if (was.moved) { if (T >= state.clock) viewFinish(); return; }
      if (el.id === 'lineage') { const r = el.getBoundingClientRect(), hit = lineageHit((ev.clientX - r.left) / r.width * el.clientWidth, (ev.clientY - r.top) / r.height * el.clientHeight, 4); if (hit) { focusByUid(hit.uid); drawTimeline(); return; } }
      if (T < state.clock - 0.5) viewTo(T, REWIND_GLIDE_MS); else { const ts = timeScale(el.clientWidth); runTo(Math.floor(T / ts.binSec) * ts.binSec); } });
    el.addEventListener('pointercancel', () => { scrub = null; });
  }
  $('lineage-btn').addEventListener('click', () => setFooter(footerMode === 'lineage' ? 'timeline' : 'lineage'));
  $('replay30').addEventListener('click', () => timelapse(30000));
  $('fitbar-go').addEventListener('click', () => goUntil(+$('fitbar').dataset.until));
  $('fitbar-no').addEventListener('click', () => { $('fitbar').hidden = true; state.rewindT = null; drawTimeline(); });
  $('fullmap').addEventListener('click', () => goUntil(0));
  $('play').addEventListener('click', () => setPlaying(!state.playing));
  $('step').addEventListener('click', () => { if (state.playing) setPlaying(false); stepOnce(); });
  $('golive').addEventListener('click', goLive);
  $('reset').addEventListener('click', resetToBase); wireTail();
}
async function resetToBase() { if (state.playing) setPlaying(false); engine.reset(); renderer.updateLabels(0, data.N); await engine.baseRanksReady; scores.fill(0); state.placeBins.clear(); state.ledgerLog = []; state.lastWave = null; state.rebuildMarks = []; state.lastRebuildClock = data.baseSeconds; state.lastRebuildPlayed = playedNow(); recomputeScores(0); updateBaseP5(); applyThreshold(); state.cumulativeMs = 0; state.log = []; state.lastStats = null; setClock(data.baseSeconds); state.achieved = 0; renderer.resetAppear(); state.countFrom = data.baseSeconds; if (view.on) viewExit(); snapClear(); regionHist.clear(); state.histMinute = -1; if (regions) { for (const r of regions.regions) { r.count = r.countBase; r.arr = 0; } regionS.fill(-1); assignRegions(0, data.nBase); regionRates(); recordHistory('base'); showRegion(); } snapOpen('base'); updateReadouts(null); render(); drawTimeline(); legend(); status('Reset: base map only.'); }
function wireTail() {
  $('speed').addEventListener('input', (e) => { state.speed = +e.target.value; $('speedv').textContent = state.speed; });
  $('interval').addEventListener('change', (e) => { state.interval = +e.target.value; state.worker?.postMessage({ type: 'batch', batchSeconds: Math.min(state.interval, 5) }); });
  $('rebuildNow').addEventListener('click', () => { requestRebuild(); });
  $('autoRebuild').addEventListener('change', (e) => { state.autoRebuild = e.target.checked; });
  $('rebuildMin').addEventListener('change', (e) => { state.rebuildMinutes = Math.max(0.5, +e.target.value || 0.5); });
  $('rebuildArrivals').addEventListener('change', (e) => { state.rebuildArrivals = Math.max(0, 1000 * (+e.target.value || 0)); });
  $('leash').addEventListener('input', (e) => { state.leash = +e.target.value; $('leashv').textContent = e.target.value; });
  $('morphMs').addEventListener('input', (e) => { state.morphMs = 1000 * +e.target.value; $('morphv').textContent = e.target.value; });
  $('budget').value = state.budget; $('budget').addEventListener('change', (e) => { state.budget = Math.max(0, +e.target.value || 0); });
  $('mode').addEventListener('change', (e) => { renderer.mode = +e.target.value; legend(); render(); });
  $('settleMs').addEventListener('input', (e) => { state.settleMs = +e.target.value; $('settlev').textContent = e.target.value; });
  $('boost').addEventListener('input', (e) => { renderer.arrivalBoost = +e.target.value; $('boostv').textContent = e.target.value; render(); });
  $('recent').addEventListener('input', (e) => { renderer.recentWindow = +e.target.value; $('recentv').textContent = e.target.value; render(); });
  $('window').addEventListener('change', (e) => { state.window = +e.target.value; recomputeScores(0); updateBaseP5(); applyThreshold(); updateReadouts(state.lastStats); legend(); render(); drawTimeline(); });
  $('thrAuto').addEventListener('change', (e) => { state.autoThreshold = e.target.checked; $('thrKept').disabled = e.target.checked; applyThreshold(); updateReadouts(state.lastStats); render(); drawTimeline(); });
  $('thrKept').addEventListener('input', (e) => { state.manualKept = +e.target.value; applyThreshold(); updateReadouts(state.lastStats); render(); drawTimeline(); });
  $('stranded').addEventListener('change', (e) => { renderer.showStranded = e.target.checked; legend(); render(); });
  $('growth').addEventListener('change', (e) => { renderer.growth = e.target.checked; legend(); render(); });
  $('gwin').addEventListener('input', (e) => { renderer.growthWindow = +e.target.value; $('gwinv').textContent = e.target.value; legend(); render(); });
  $('gden').addEventListener('input', (e) => { renderer.densityScale = +e.target.value; render(); });
  $('regionsOn').addEventListener('change', (e) => { rstate.on = e.target.checked; buildRegions(); render(); });
  $('regionAlpha').addEventListener('input', (e) => { renderer.regionAlpha = +e.target.value; render(); });
  $('regionSigma').addEventListener('change', (e) => { rstate.sigma = +e.target.value; $('regionSigmav').textContent = e.target.value; buildRegions(); render(); });
  $('regionSigma').addEventListener('input', (e) => { $('regionSigmav').textContent = e.target.value; });
  $('regionSoft').addEventListener('change', (e) => { rstate.soft = +e.target.value; $('regionSoftv').textContent = e.target.value; buildRegions(); render(); });
  $('regionSoft').addEventListener('input', (e) => { $('regionSoftv').textContent = e.target.value; });
  $('regionTau').addEventListener('change', (e) => { rstate.tau = +e.target.value; $('regionTauv').textContent = e.target.value; buildRegions(); render(); });
  $('regionTau').addEventListener('input', (e) => { $('regionTauv').textContent = e.target.value; });
  $('regionTint').addEventListener('change', (e) => { rstate.tint = e.target.value; buildRegions(); render(); });
  $('ghost').addEventListener('input', (e) => { renderer.ghostAlpha = +e.target.value; render(); });
  $('chipsOn').addEventListener('change', (e) => { rstate.chips = e.target.checked; layoutChips(); });
  { const box = $('chipWords'), saved = (() => { try { return localStorage.getItem('jet.chipWords'); } catch { return null; } })();
    const pick = (n, quiet) => { setNamingSlots(n); for (const b of box.children) b.classList.toggle('on', +b.dataset.n === n); try { localStorage.setItem('jet.chipWords', String(n)); } catch {} if (!quiet && engine && rstate.on) buildRegions({ crossfade: false }); };
    for (const n of [1, 2, 3]) { const b = document.createElement('button'); b.className = 'pill'; b.textContent = n; b.dataset.n = n; b.addEventListener('click', () => pick(n)); box.appendChild(b); }
    pick(+(params.get('words') || saved || 3), !regions); }
  $('psize').addEventListener('input', (e) => { renderer.pointPx = +e.target.value; render(); });
  $('dim').addEventListener('input', (e) => { renderer.dimBase = +e.target.value; render(); });
  $('scoreBins').addEventListener('click', () => {
    const thr = threshold(), bins = new Map(); for (let p = 0; p < engine.n; p++) { const b = p < data.nBase ? 'base map' : mmss(Math.floor(data.tS[p] / 600) * 600) + '+'; const e = bins.get(b) || [0, 0, []]; e[1]++; e[2].push(scores[p] * K); if (scores[p] < thr) e[0]++; bins.set(b, e); }
    $('r-bins').innerHTML = `window ${state.window}, stranded = ${state.autoThreshold ? state.baseQ5 + ' or fewer' : 'fewer than ' + state.manualKept} of ${K} kept, scored on the current map<br>` + [...bins].map(([b, e]) => `${b}: ${e[1].toLocaleString()} posts, median ${fmt(median(e[2]))} of ${K} kept, ${fmt(100 * e[0] / e[1])} % stranded`).join('<br>');
  });
  // a resize refits the map to the window, unless the view was zoomed or panned by hand
  window.addEventListener('resize', () => { if (!state.viewTouched && data?.baseXY && !view.on && !cast?.on) requestAnimationFrame(() => { renderer.fitTo(data.baseXY); render(); }); render(); drawTimeline(); });
}

async function main() {
  if (BAD_SOURCE) { const known = Object.keys(DATASETS).join(', '); status(`There is no dataset called "${BAD_SOURCE}". Use ?source= with one of ${known}, or open the page without ?source= for live Bluesky posts.`, true); return; }
  try {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser. Use Chrome, Edge or Safari 26 with WebGPU enabled.');
    $('capname').textContent = SRC.static ? SRC.feed : SRC.what[0].toUpperCase() + SRC.what.slice(1);   // the map caption: what is shown, then how
    $('h1sub').textContent = SRC.h1.replace(new RegExp('^' + (SRC.feed || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ': '), '') + (TUNED ? ' · tuned base' : ''); $('r-filter').textContent = SRC.filter;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); if (!adapter) throw new Error('No WebGPU adapter.');
    wireSource();
    let col = null, N;
    if (FRESH) N = SRC.static ? CAP + 4096 : 40000 + CAP;   // room for up to 40,000 base posts (a static dataset: up to 256,000 rows in all); the preview fits need the device before the base is known
    else { data = await loadData(CAPTURE_URL, BASE, status, CAP); N = data.N; }
    const dim = FRESH ? (SRC.dim || 384) : data.dim;
    const needBind = Math.max(64 * N * 4, (N + FLUSH_CAP) * dim * 4);   // features carry FLUSH_CAP scratch rows for list refreshes
    if (adapter.limits.maxStorageBufferBindingSize < needBind) throw new Error('This GPU cannot bind a ' + (needBind / 1048576).toFixed(0) + ' MB storage buffer; lower ?cap=');
    // the points shader binds 9 storage buffers in the vertex stage; the WebGPU default limit is 8 (the investigator's Chrome refuses the pipeline without this), most adapters allow 10
    device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [], requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10), maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, Math.max(needBind, 134217728)), maxBufferSize: Math.min(adapter.limits.maxBufferSize, Math.max(needBind, 268435456)) } });
    device.lost.then((info) => status('GPU device lost: ' + info.message, true));
    if (FRESH && params.get('base') === 'handoff') {   // a base from the tuner: only if it is this source's (and this recording's)
      HANDOFF = await loadHandoff().catch(() => null);
      if (!HANDOFF || HANDOFF.source !== SOURCE || (RECORDED && HANDOFF.recKey !== REC_KEY) || (SRC.dim && HANDOFF.dim !== SRC.dim)) { console.warn('[jet] no handed-over base for this source; fitting as usual'); HANDOFF = null; const u = new URL(location.href); u.searchParams.delete('base'); history.replaceState(null, '', u); }
      if (HANDOFF) $('h1sub').textContent += ' · tuned base';
    }
    if (FRESH) { col = HANDOFF?.later ? await handoffCollect(device) : await freshCollect(FRESH, device); data = await freshBuild(device, col, SRC.static ? Math.max(0, (SRC.rowsLoaded || 0) - col.basePosts.length) + 4096 : CAP); if (col.catIndex) { data.catIndex = col.catIndex; data.catOther = col.catOther; } state.cacheDb = col.db; } state.countFrom = data.baseSeconds;
    const info = adapter.info || {}; $('r-gpu').textContent = [info.vendor, info.architecture, info.description].filter(Boolean).join(' · ') || 'adapter details hidden by the browser';
    engine = new StreamingEngine(device, data, { lazyLedger: !!SRC.static }); await engine.checkShaders();
    renderer = new PointRenderer(device, $('map'), engine, data); renderer.onRedraw = () => render(); scores = new Float32Array(data.N);   // onRedraw: a draw skipped by the queue guard is made up once the GPU catches up
    engine.onLabels = (s, w) => renderer.updateLabels(s, w);
    status('scoring the base map'); await engine.baseRanksReady; recomputeScores(0); updateBaseP5(); applyThreshold();
    countArrivals(0, data.n); setClock(data.baseSeconds); state.lastRebuildClock = data.baseSeconds; wire(); renderer.mode = +$('mode').value; status('computing regions of the base map'); try { const r = await fetch('./nsfw_words.txt', { cache: 'no-store' }); if (r.ok) { const n = setNsfwWords(await r.text()); console.info('[jet] nsfw list', n); } } catch (e) { console.warn('[jet] nsfw list not loaded', e); } relabel(); status('choosing the region grain'); await new Promise(r => setTimeout(r, 20)); autoLook({ grain: true }); await buildRegions(); snapOpen('base'); legend(); render(); drawTimeline(); updateReadouts(null);
    const bm = data.baseMeta, dropped = bm.dropped_base_window_replies || 0;
    if (FRESH) {
      document.title = SRC.static ? `UMAP Realtime · ${SRC.feed}` : `UMAP Realtime · ${SRC.feed}, last ${FRESH} min`; $('base').parentElement.hidden = true;
      $('r-capture').textContent = SRC.static ? `${SRC.feed} (${SRC.what}): ${(SRC.rowsLoaded || 0).toLocaleString()} rows, ${SRC.filter}; ${UNTIL ? `the map holds the rows up to ${tLabelDate(UNTIL)}, Play places the rest` : 'one map of every row'}` : `fresh: the last ${FRESH} min of ${SRC.feed} (${SRC.what}), fetched at ${new Date(col.liveStartUs / 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} local and embedded here · room for ${CAP.toLocaleString()} more rows`;
      $('r-base').textContent = `${data.nBase.toLocaleString()} ${SOURCE === 'jetstream' ? 'original (non-reply) posts' : SRC.noun} from ${Math.round(col.minutes)} to ${Math.round(col.minutes / 2)} min before the fetch${SOURCE === 'jetstream' ? ' (replies are not embedded or used)' : ''}, fitted on this GPU in ${(data.fit.ms / 1000).toFixed(1)} s (${data.fit.steps} steps, ${data.fit.edges.toLocaleString()} edges, after ${col.refits} preview fits while loading) · ${data.labels.length} topic clusters` + (data.fit.tuned ? ` · tuned base layout (${data.fit.tuned}); placement and rebuilds keep the 15-neighbour lists and a, b of spread 1.2, min distance 0.03` : '') + (data.fit.handoff ? ` · base positions from the base map tuner${data.fit.handoff.settings ? ` (${data.fit.handoff.settings})` : ''}; placement and rebuilds keep the 15-neighbour lists and a, b of spread 1.2, min distance 0.03` : '');
      $('r-encoder').textContent = SRC.static ? `${data.model} (${data.dim} numbers per ${SRC.one}) · ${(SRC.rowsLoaded || 0).toLocaleString()} of ${(SRC.rowsTotal || 0).toLocaleString()} rows (?rows= to change)` : SRC.numeric ? `${data.model}: ${data.dim} state features from the relay` : `${data.model} (${data.dim}-dim, mean pooling); embedded in the browser on WebGPU`;
      // the stream half of the hour and everything after it: recorded rows up to captureEnd, live rows beyond
      state.liveShift = 0; state.captureEnd = (col.liveStartUs - col.cursorUs) / 1e6; state.windowEndUs = col.liveStartUs; if (col.worker) attachWorker(col.worker); if (col.startMsg) col.worker.postMessage(col.startMsg);
      for (const m of col.later) ingestFrame(m.hdr, m.vecs, () => {}); col.later = [];
      if (SRC.static) { staticUi(); await staticInterval(); }
      // hand-off from the preview: the renderer takes the preview's mapping, rows start where the preview showed them
      if (col.axes) {
        const pm = previewMapping(), target = { ...renderer.view };   // the view fitTo gave the fitted map; the glide eases from the preview's view to it
        renderer.view.cx = 0; renderer.view.cy = 0; renderer.view.scale = pm.scale; const from = { ...renderer.view };
        // rows start where the cloud showed them: the base rows from the fit's start, a static dataset's placed rows from the
        // same projection of their vectors
        if (SRC.static && !col.seeded) gpuCloudStart(col.axes, engine.n); else { engine.start.set(data.fit.y0.subarray(0, data.nBase * 2)); engine.device.queue.writeBuffer(engine.startBuf, 0, engine.start.subarray(0, data.nBase * 2)); }
        // the chips wait out the glide (laid out every frame they would cost layout time and pop in and out as the zoom
        // changes) and fade in once the map has settled
        const ch = $('chips'); state.chipsHold = true; ch.style.opacity = '0';
        crossfadeToken++; state.morphing = true; renderer.settle = 0; renderer.rmix = 0; drawFrame(); $('map').style.visibility = ch.style.visibility = ''; $('preview').hidden = true;
        glide(2500, (e) => { renderer.view.cx = from.cx + (target.cx - from.cx) * e; renderer.view.cy = from.cy + (target.cy - from.cy) * e; renderer.view.scale = from.scale + (target.scale - from.scale) * e; }).then(() => {
          state.chipsHold = false; render(); ch.style.transition = 'opacity .45s ease'; requestAnimationFrame(() => { ch.style.opacity = ''; }); setTimeout(() => { ch.style.transition = ''; }, 600); if (SRC.static) staticDone(); if (UNTIL) engine.ledgerReady; });   // a rewound map: Play's placement needs the ledger; started once the glide is over
      } else { $('map').style.visibility = $('chips').style.visibility = ''; $('preview').hidden = true; if (SRC.static) staticDone(); }
      if (HANDOFF?.view) { Object.assign(renderer.view, viewFrom(HANDOFF.view, renderer.canvas.getBoundingClientRect())); render(); }   // the tuner's view, unchanged
      dropSnapshot();   // this page's own map is drawn: the previous page's picture fades out
      if (!SRC.static) status(HANDOFF ? `Base map from the tuner · ${(data.n - data.nBase).toLocaleString()} arriving` : `Base map fitted · ${(data.n - data.nBase).toLocaleString()} arriving`);
    } else {
      document.title = `UMAP Realtime · ${CAPTURE}`;
      const cs = data.rowsMeta.capture_stats || {};
      $('r-capture').textContent = `${CAPTURE}: ${data.n.toLocaleString()} posts over ${mmss(data.rowsMeta.duration_seconds)} starting ${wall(0)} local` + (CAP ? ` · room for ${CAP.toLocaleString()} live arrivals` : '');
      $('r-base').textContent = `${data.nBase.toLocaleString()} ${bm.originals_only ? 'original (non-reply) ' : ''}posts from the first ${mmss(data.baseSeconds)}${dropped ? ` (${dropped.toLocaleString()} replies in that window left out)` : ''}; stream = ${(data.n - data.nBase).toLocaleString()} posts · ${data.labels.length} topic clusters`;
      $('r-encoder').textContent = `${data.model} (${data.dim}-dim${bm.pooling ? ', ' + bm.pooling + ' pooling' : ''}); embedded in the relay, never in the browser`;
      $('r-filter').textContent = filterText(cs);
      status('Ready');
    }
    window.jet = { get tuner() { return tuner; }, get cast() { return cast; }, get regionPrevS() { return regionPrevS; }, engine, renderer, data, state, events, stepOnce, runTo, get scores() { return scores; }, flushPending, drawFrame, displayClock, kickAnim, parity, strandedCurve, enforceBudget, compactAll, doRebuild, requestRebuild, get regions() { return regions; }, buildRegions, regionRates, regionSpark, regionHist, showRegion, setFocus, render, layoutChips, snaps, get live() { return live; }, view, viewTo, viewExit, viewFinish, timelapse, snapBytes, setFooter, drawLineage, lineage };
  } catch (e) { console.error(e); status('Error: ' + e.message, true); }
}
main();
