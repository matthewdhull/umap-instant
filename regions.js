// Soft regions over the frozen base map. Clustering is a port of Ren, Hohman & Moritz, "A Scalable
// Approach to Clustering Embedding Projections" (arXiv 2504.07285), taken from the ASRS dashboard:
// KDE on a grid, hill-climb every pixel to its local density maximum, union clusters whose peak barely
// rises above a shared saddle, truncate each cluster at 0.1x its peak. Regions are computed ONCE from
// the base map and never move; arrivals acquire membership by landing inside them. Instead of hard
// outlines, each region is drawn as a blurred, density-weighted tint that blends with its neighbours.
export const REGION_GRID = 512, REGION_MERGE_TAU = 0.2, REGION_TRUNCATE = 0.1;

// Every step below is written as a generator that yields between blocks of work: runSync runs it straight through (the
// same result as before, used where a caller needs it at once); runSliced runs it in slices of about 8 ms and gives the
// page its turn between slices, so a rebuild of the regions never stops the animation (0.47 s of CPU at 91,000 rows).
export function runSync(it) { for (;;) { const r = it.next(); if (r.done) return r.value; } }
export async function runSliced(it, budgetMs = 8) {
  let t = performance.now();
  for (;;) { const r = it.next(); if (r.done) return r.value; if (performance.now() - t > budgetMs) { await new Promise((res) => setTimeout(res, 0)); t = performance.now(); } }
}

export function gaussianBlurGrid(src, g, sigma) { return runSync(blurSteps(src, g, sigma)); }
function* blurSteps(src, g, sigma) {
  const radius = Math.max(1, Math.min(48, Math.round(sigma * 3)));
  const kernel = new Float32Array(radius * 2 + 1); let ksum = 0;
  for (let i = -radius; i <= radius; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); kernel[i + radius] = v; ksum += v; }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;
  const tmp = new Float32Array(g * g), out = new Float32Array(g * g);
  for (let y = 0; y < g; y++) { const row = y * g; for (let x = 0; x < g; x++) { let acc = 0; for (let k = -radius; k <= radius; k++) { const xx = x + k; if (xx >= 0 && xx < g) acc += src[row + xx] * kernel[k + radius]; } tmp[row + x] = acc; } if ((y & 31) === 31) yield; }
  for (let x = 0; x < g; x++) { for (let y = 0; y < g; y++) { let acc = 0; for (let k = -radius; k <= radius; k++) { const yy = y + k; if (yy >= 0 && yy < g) acc += tmp[yy * g + x] * kernel[k + radius]; } out[y * g + x] = acc; } if ((x & 31) === 31) yield; }
  return out;
}

export function regionClusterDensity(dmap, g, mergeTau = REGION_MERGE_TAU) {
  const n = g * g, parent = new Int32Array(n);
  for (let y = 0; y < g; y++) for (let x = 0; x < g; x++) {
    const i = y * g + x; let best = dmap[i], bestIdx = i;
    for (let dy = -1; dy <= 1; dy++) { const yy = y + dy; if (yy < 0 || yy >= g) continue;
      for (let dx = -1; dx <= 1; dx++) { if (!dx && !dy) continue; const xx = x + dx; if (xx < 0 || xx >= g) continue; const j = yy * g + xx, v = dmap[j]; if (v > best || (v === best && j < bestIdx)) { best = v; bestIdx = j; } } }
    parent[i] = bestIdx;
  }
  for (let pass = 0; pass < 64; pass++) { let changed = false; for (let i = 0; i < n; i++) { const p = parent[parent[i]]; if (p !== parent[i]) { parent[i] = p; changed = true; } } if (!changed) break; }
  const rootIds = new Map(), cmap = new Int32Array(n);
  for (let i = 0; i < n; i++) { const r = parent[i]; let id = rootIds.get(r); if (id === undefined) { id = rootIds.size; rootIds.set(r, id); } cmap[i] = id; }
  const nInit = rootIds.size, peak = new Float64Array(nInit);
  for (let i = 0; i < n; i++) { const c = cmap[i]; if (dmap[i] > peak[c]) peak[c] = dmap[i]; }
  const edgeMax = new Map();
  const consider = (a, b, v) => { if (a === b) return; const key = a < b ? a * nInit + b : b * nInit + a; const cur = edgeMax.get(key); if (cur === undefined || v > cur) edgeMax.set(key, v); };
  for (let y = 0; y < g; y++) for (let x = 0; x < g; x++) {
    const i = y * g + x, c = cmap[i], v = dmap[i];
    if (x + 1 < g) consider(c, cmap[i + 1], Math.min(v, dmap[i + 1]));
    if (y + 1 < g) { consider(c, cmap[i + g], Math.min(v, dmap[i + g])); if (x + 1 < g) consider(c, cmap[i + g + 1], Math.min(v, dmap[i + g + 1])); if (x > 0) consider(c, cmap[i + g - 1], Math.min(v, dmap[i + g - 1])); }
  }
  const uf = new Int32Array(nInit); for (let i = 0; i < nInit; i++) uf[i] = i;
  const find = i => { while (uf[i] !== i) { uf[i] = uf[uf[i]]; i = uf[i]; } return i; };
  const cand = [];
  edgeMax.forEach((v, key) => { const a = Math.floor(key / nInit), b = key - a * nInit; const small = peak[a] <= peak[b] ? a : b; cand.push([(peak[small] - v) / Math.max(peak[small], 1e-12), a, b]); });
  cand.sort((p, q) => p[0] - q[0]);
  for (const e of cand) { if (e[0] >= mergeTau) break; const ra = find(e[1]), rb = find(e[2]); if (ra === rb) continue; if (peak[ra] >= peak[rb]) uf[rb] = ra; else uf[ra] = rb; }
  const mergedIds = new Map(), rootToFinal = new Int32Array(nInit), peakM = [];
  for (let c = 0; c < nInit; c++) { const r = find(c); let id = mergedIds.get(r); if (id === undefined) { id = mergedIds.size; mergedIds.set(r, id); peakM.push(0); } rootToFinal[c] = id; if (peak[c] > peakM[id]) peakM[id] = peak[c]; }
  let globalPeak = 0; for (const p of peakM) if (p > globalPeak) globalPeak = p;
  const floor = globalPeak * 1e-4, finalMap = new Int32Array(n);
  for (let i = 0; i < n; i++) { const id = rootToFinal[cmap[i]]; finalMap[i] = (dmap[i] > floor && dmap[i] >= peakM[id] * REGION_TRUNCATE) ? id : -1; }
  return { cl: finalMap, peak: peakM };
}

const STOP = new Set(("a about above after again against all am an and any are as at be because been before being below between both but by can did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with you your yours yourself yourselves " +
  "like just im dont thats lol lmao yeah yes oh ok okay really people think know going got want need way say said make good time thing things day today one get go see also even still much many would could gonna wanna gotta ive youre theyre dont cant wont isnt new old big little back next last first also every never ever always something someone anything everyone anyone nothing everything else maybe actually literally right well lot lots bit pretty sure much more seen used use using call called great true best better look looking come came went take took give gave put let keep made makes find found tell told ask asked try trying start started show shows thanks thank please check").split(/\s+/));
const PROFANE = new Set(['fuck', 'fucking', 'shit', 'cum', 'porn', 'nsfw', 'sex', 'dick', 'ass', 'bitch']);
// NSFW words hidden from region and topic names while the filter is on (the default): whole words and phrases plus
// substring rules, loaded by the page from nsfw_words.txt (the public LDNOOBW list and this lane's additions). Posts are
// never filtered, only the words that may name a chip.
const NSFW = { on: true, words: new Set(), subs: [] };
export function setNsfwWords(text) { const words = new Set(), subs = []; for (let line of text.split('\n')) { line = line.trim().toLowerCase(); if (!line || line.startsWith('#')) continue; if (line.startsWith('*')) subs.push(line.slice(1).trim()); else words.add(line); } NSFW.words = words; NSFW.subs = subs; return { words: words.size, rules: subs.length }; }
export function setNsfwFilter(on) { NSFW.on = !!on; }
export function nsfwFilterOn() { return NSFW.on; }
export const isNsfw = (w) => NSFW.on && (PROFANE.has(w) || NSFW.words.has(w) || NSFW.subs.some(s => w.includes(s)));
const APOS = /[’‘ʼ`´]/g;   // curly and other apostrophes, normalised to ' before the stop test
const DOMAIN = /\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|tv|io|gg|co|uk|me|app|xyz|social|info|news|live|us|ca|de|fr|jp|be|it|es|nl|ai)(?:\/\S*)?/g;   // bare domains without a scheme (onlyfans.com/..., bsky.app)
// Token stream of a post: lowercase, links and mentions removed, curly apostrophes normalised, a possessive 's stripped
// (trump's is trump), any remaining contraction (i'll, we've, don't) and stop or profane words dropped. A dropped token
// leaves a gap (null) so bigrams only ever join two content words that stood side by side in the post.
function tokens(text) {
  const t = text.toLowerCase().replace(APOS, "'").replace(/https?:\/\/\S+|\bwww\.\S+/g, ' ').replace(DOMAIN, ' ').replace(/(?<!\w)@[\w.\-]+/g, ' ');
  const out = [];
  for (const raw of t.match(/[a-z][a-z']{2,}/g) || []) {
    const w = raw.replace(/^'+|'+$/g, '').replace(/'s$/, '');
    out.push(w.length >= 3 && !w.includes("'") && !STOP.has(w) && !isNsfw(w) ? w : null);
  }
  return out;
}
// a plural and its singular (movies/movie, games/game, peoples/people) never both name one region
const stem = (t) => t.length >= 4 ? t.replace(/ies$/, 'y').replace(/([^su])s$/, '$1') : t;
const stemOf = (t) => t.split(' ').map(stem).join(' ');
// One document per post: the set of its words and the set of its adjacent word pairs (bigrams).
export function docsOf(posts) { return runSync(docsSteps(posts)); }
export function* docsSteps(posts) {
  const n = posts.length, docs = new Array(n), gdf = new Map(), gbf = new Map();
  for (let i = 0; i < n; i++) {
    if ((i & 255) === 255) yield;
    const seq = posts[i].terms || tokens(posts[i].text), w = new Set(), b = new Set();   // a numeric source (aircraft states) brings its own naming terms
    for (let k = 0; k < seq.length; k++) { if (!seq[k]) continue; w.add(seq[k]); if (k + 1 < seq.length && seq[k + 1] && seq[k + 1] !== seq[k]) { const bg = seq[k] + ' ' + seq[k + 1]; if (!isNsfw(bg)) b.add(bg); } }
    docs[i] = { w, b }; for (const t of w) gdf.set(t, (gdf.get(t) || 0) + 1); for (const t of b) gbf.set(t, (gbf.get(t) || 0) + 1);
  }
  return { docs, gdf, gbf, n };
}
// Names for groups of posts (regions of the map, or topic clusters), the class-based TF-IDF of BERTopic with two guards:
//   score(word, group) = coverage x log(1 + mean group size / base-wide count of the word) x sqrt(coverage / its best coverage in any group)
// where coverage = share of the group's posts containing the word. The last factor lets a word lead only in the group where
// it is most concentrated, so "trump" names one chip rather than four. A word must appear in at least 4 % of the group's
// posts (and at least 3) to count: a rare word with a huge ratio names nothing a reader would recognise. A two-word phrase
// replaces its leading word when at least half of that word's posts in the group carry the phrase ("shane gillis" for
// "gillis"). A name is three word slots; a phrase takes two.
// Naming defaults per source: text keeps a word from 4 % of a group's posts; aircraft samples carry few terms each, and a
// rare route or type in a twentieth of a region otherwise names the whole region, so they need a quarter.
const NAMING = { minShare: 0.04, slots: 3 };
export function setNamingMinShare(v) { NAMING.minShare = v; }
// Words per region name (1-3); a two-word phrase only fits when two slots are left.
export function setNamingSlots(v) { NAMING.slots = Math.max(1, Math.min(3, v | 0)); }
export function nameGroups(groups, D, opts = {}) { return runSync(nameGroupsSteps(groups, D, opts)); }
export function* nameGroupsSteps(groups, D, { slots = NAMING.slots, minShare = NAMING.minShare, minCount = 3, phraseShare = 0.5 } = {}) {
  const { docs, gdf, gbf, n } = D, G = groups.length, meanSize = groups.reduce((a, g) => a + g.length, 0) / Math.max(1, G);
  const cw = groups.map(() => new Map()), cb = groups.map(() => new Map()), leadW = new Map(), leadB = new Map();
  for (let g = 0; g < G; g++) { const rows = groups[g], W = cw[g], B = cb[g]; let c0 = 0; for (const i of rows) { if ((++c0 & 1023) === 0) yield; const d = docs[i]; if (!d) continue; for (const t of d.w) W.set(t, (W.get(t) || 0) + 1); for (const t of d.b) B.set(t, (B.get(t) || 0) + 1); }
    const size = Math.max(1, rows.length); W.forEach((c, t) => { const cov = c / size; if (cov > (leadW.get(t) || 0)) leadW.set(t, cov); }); B.forEach((c, t) => { const cov = c / size; if (cov > (leadB.get(t) || 0)) leadB.set(t, cov); }); yield; }
  const out = [];
  for (let g = 0; g < G; g++) out.push(nameOne(groups[g], g)), yield;
  return out;
  function nameOne(rows, g) {
    const size = Math.max(1, rows.length), minC = Math.max(minCount, Math.ceil(minShare * size)), W = cw[g], B = cb[g];
    const score = (c, gf, lead) => { const cov = c / size; return cov * Math.log(1 + meanSize / Math.max(1, gf)) * Math.sqrt(cov / Math.max(cov, lead)); };
    const words = []; W.forEach((c, t) => { if (c >= minC) words.push({ t, c, s: score(c, gdf.get(t) || c, leadW.get(t) || 0), f: c / size, gf: (gdf.get(t) || c) / n }); }); words.sort((a, b) => b.s - a.s || a.t.localeCompare(b.t));
    const chosen = [], used = new Set(), termInfo = []; let left = slots;
    for (const w of words) {
      const st = stem(w.t); if (used.has(st)) continue;
      let phrase = null;
      if (left >= 2) { B.forEach((c, t) => { if (c < minC || c < phraseShare * w.c) return; const [a, b] = t.split(' '); if (a !== w.t && b !== w.t) return; const other = a === w.t ? b : a; if (used.has(stem(other))) return; if (!phrase || c > phrase.c) phrase = { t, c, other }; }); }
      if (phrase) { chosen.push(phrase.t); used.add(st); used.add(stem(phrase.other)); termInfo.push({ t: phrase.t, f: phrase.c / size, gf: (gbf.get(phrase.t) || phrase.c) / n }); left -= 2; }
      else { chosen.push(w.t); used.add(st); termInfo.push({ t: w.t, f: w.f, gf: w.gf }); left -= 1; }
      if (left <= 0) break;
    }
    if (!chosen.length && minShare > 0) {   // a mixed group where no word reaches the coverage floor: name it by the same score without the floor
      const alt = nameGroups([rows], { docs, gdf, gbf, n }, { slots, minShare: 0, minCount, phraseShare })[0]; return alt;
    }
    const terms = chosen.slice(); for (const w of words) { if (terms.length >= 5) break; if (used.has(stem(w.t))) continue; used.add(stem(w.t)); terms.push(w.t); termInfo.push({ t: w.t, f: w.f, gf: w.gf }); }
    return { name: chosen.length ? chosen.join(' · ') : '', terms, termInfo };
  }
}

// Compute regions from the first n rows of pos (the base map). Returns everything the renderer and the
// chips need, plus regionOf(x, y) for membership lookups of later arrivals.
export const PALETTE = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
  '#1f77b4', '#aec7e8', '#ffbb78', '#98df8a', '#d62728', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2',
  '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#3182bd'];
// Names for k topic clusters of the base posts (nameGroups rule).
export function nameClusters(posts, labelS, k, D = null) {
  D = D || docsOf(posts); const groups = Array.from({ length: k }, () => []); for (let i = 0; i < posts.length; i++) if (labelS[i] < k) groups[labelS[i]].push(i);
  const named = nameGroups(groups, D), names = [], terms = [], colors = {};
  named.forEach((e, c) => { const nm = String(c + 1).padStart(2, '0') + ' ' + (e.name || 'cluster'); names.push(nm); terms.push(e.terms); colors[nm] = PALETTE[c % PALETTE.length]; });
  return { names, terms, colors };
}
export function computeRegions(pos, n, opts = {}) { return runSync(computeRegionsSteps(pos, n, opts)); }
export function* computeRegionsSteps(pos, n, { sigma = 5, tintSigma = 0, posts = null, docs = null, feat = null, dim = 0, labels = null, labelS = null, labelColors = null, palette = null, softSigma = 6, tint = 'topic', mergeTau = REGION_MERGE_TAU } = {}) {
  const t0 = performance.now(), g = REGION_GRID;
  let ex0 = Infinity, ex1 = -Infinity, ey0 = Infinity, ey1 = -Infinity;
  for (let i = 0; i < n; i++) { const x = pos[2 * i], y = pos[2 * i + 1]; if (x < ex0) ex0 = x; if (x > ex1) ex1 = x; if (y < ey0) ey0 = y; if (y > ey1) ey1 = y; }
  // Pad the grid well past the blur reach (density blur + soft blur, in cells) so no tint is cut off at the texture edge.
  const padFrac = Math.max(0.1, (3 * sigma + 3 * softSigma + 8) / g);
  const padX = (ex1 - ex0) * padFrac, padY = (ey1 - ey0) * padFrac; ex0 -= padX; ex1 += padX; ey0 -= padY; ey1 += padY;
  const spanX = ex1 - ex0, spanY = ey1 - ey0;
  const gx = (x) => Math.min(g - 1, Math.max(0, Math.floor((x - ex0) / spanX * g))), gy = (y) => Math.min(g - 1, Math.max(0, Math.floor((y - ey0) / spanY * g)));
  const counts = new Float32Array(g * g); for (let i = 0; i < n; i++) counts[gy(pos[2 * i + 1]) * g + gx(pos[2 * i])] += 1;
  yield; const dmap = yield* blurSteps(counts, g, sigma); yield; const { cl, peak } = regionClusterDensity(dmap, g, mergeTau); yield;
  // the tint may be painted narrower than the clustering grain: each region's peak at that width normalises its tint, so cores reach full strength and edges truncate where the points end
  const dmapT = tintSigma && tintSigma !== sigma ? yield* blurSteps(counts, g, tintSigma) : dmap, peakT = new Float64Array(peak.length);
  for (let i = 0; i < g * g; i++) { const id = cl[i]; if (id >= 0 && dmapT[i] > peakT[id]) peakT[id] = dmapT[i]; }
  // per-region bookkeeping from the base rows
  const per = new Map(); yield;
  for (let i = 0; i < n; i++) { const id = cl[gy(pos[2 * i + 1]) * g + gx(pos[2 * i])]; if (id < 0) continue; let e = per.get(id); if (!e) { e = { id, count: 0, sx: 0, sy: 0, syy: 0, rows: [], labels: new Map() }; per.set(id, e); } e.count++; e.sx += pos[2 * i]; e.sy += pos[2 * i + 1]; e.syy += pos[2 * i + 1] * pos[2 * i + 1]; e.rows.push(i); if (labelS) e.labels.set(labelS[i], (e.labels.get(labelS[i]) || 0) + 1); }
  const minPoints = Math.max(25, Math.round(n * 0.0008));
  const kept = [...per.values()].filter(e => e.count >= minPoints).sort((a, b) => b.count - a.count);
  const idToRegion = new Int32Array(peak.length).fill(-1); kept.forEach((e, k) => { idToRegion[e.id] = k; });
  // names: the nameGroups rule over the base posts inside each region; a representative post = the base row whose vector is
  // nearest the region's mean vector (by cosine; the vectors are unit length), shown on chip click so a reader can recognise the region
  const D = posts ? (docs || (yield* docsSteps(posts))) : null, named = D ? yield* nameGroupsSteps(kept.map(e => e.rows), D) : null;
  const repOf = (rows) => { if (!feat || !dim || !rows.length) return -1; const m = new Float64Array(dim); for (const i of rows) { const o = i * dim; for (let k = 0; k < dim; k++) m[k] += feat[o + k]; } let best = -1, bd = -Infinity; for (const i of rows) { const o = i * dim; let d = 0; for (let k = 0; k < dim; k++) d += feat[o + k] * m[k]; if (d > bd) { bd = d; best = i; } } return best; };
  const PASTEL = ['#9ecae1', '#fdd0a2', '#c7e9c0', '#dadaeb', '#fcbba1', '#d9d9d9', '#fdd835', '#bcbddc', '#a1d99b', '#fdae6b', '#9e9ac8', '#74c476', '#e7969c', '#7fcdbb', '#c994c7', '#fee391'];
  const regions = [];
  for (let k = 0; k < kept.length; k++) { yield; const e = kept[k]; regions.push((() => {
    const nm = named ? named[k] : null, name = nm && nm.name ? nm.name : `region ${k + 1}`, terms = nm ? nm.terms : [], termInfo = nm ? nm.termInfo : [], rep = repOf(e.rows);
    let majority = -1, mc = 0; e.labels.forEach((c, l) => { if (c > mc) { mc = c; majority = l; } });
    const color = tint === 'topic' && labels && majority >= 0 ? (labelColors[labels[majority]] || '#999999') : PASTEL[k % PASTEL.length];
    return { k, id: e.id, countBase: e.count, count: e.count, arr: 0, uid: 0, prev: null, termInfo, rep, cx: e.sx / e.count, cy: e.sy / e.count, sdy: Math.sqrt(Math.max(0, e.syy / e.count - (e.sy / e.count) ** 2)), name, terms, majority, color, peak: peak[e.id], peakTint: peakT[e.id] || peak[e.id] };
  })()); }
  yield;
  // soft field: per region, blur its indicator inside an expanded bbox, weight by density relative to the region's peak, and blend by normalised sum
  const rgbSum = new Float32Array(g * g * 3), aSum = new Float32Array(g * g), pad = Math.ceil(3 * softSigma);
  const bbox = new Map(); for (let i = 0; i < g * g; i++) { const r = cl[i] >= 0 ? idToRegion[cl[i]] : -1; if (r < 0) continue; const x = i % g, y = (i / g) | 0; const b = bbox.get(r); if (!b) bbox.set(r, [x, y, x, y]); else { if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y; if (x > b[2]) b[2] = x; if (y > b[3]) b[3] = y; } }
  const radius = Math.max(1, Math.round(softSigma * 3)), kern = new Float32Array(radius * 2 + 1); { let s = 0; for (let i = -radius; i <= radius; i++) { const v = Math.exp(-(i * i) / (2 * softSigma * softSigma)); kern[i + radius] = v; s += v; } for (let i = 0; i < kern.length; i++) kern[i] /= s; }
  for (const r of regions) {
    const b = bbox.get(r.k); if (!b) continue; const x0 = Math.max(0, b[0] - pad), y0 = Math.max(0, b[1] - pad), x1 = Math.min(g - 1, b[2] + pad), y1 = Math.min(g - 1, b[3] + pad), w = x1 - x0 + 1, h = y1 - y0 + 1;
    const m = new Float32Array(w * h), tmp = new Float32Array(w * h);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) { const i = y * g + x; if (cl[i] >= 0 && idToRegion[cl[i]] === r.k) m[(y - y0) * w + (x - x0)] = 1; }
    yield;
    for (let y = 0; y < h; y++) { for (let x = 0; x < w; x++) { let acc = 0; for (let k = -radius; k <= radius; k++) { const xx = x + k; if (xx >= 0 && xx < w) acc += m[y * w + xx] * kern[k + radius]; } tmp[y * w + x] = acc; } if ((y & 31) === 31) yield; }
    const c = hex(r.color);
    for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) { if (y === 0 && (x & 31) === 31) yield; let acc = 0; for (let k = -radius; k <= radius; k++) { const yy = y + k; if (yy >= 0 && yy < h) acc += tmp[yy * w + x] * kern[k + radius]; }
      if (acc < 0.01) continue; const i = (y + y0) * g + (x + x0); const a = acc * (0.45 + 0.55 * Math.min(1, dmap[i] / r.peak)); rgbSum[3 * i] += c[0] * a; rgbSum[3 * i + 1] += c[1] * a; rgbSum[3 * i + 2] += c[2] * a; aSum[i] += a; }
  }
  const rgba = new Uint8Array(g * g * 4), edgeFade = radius;   // safety net: force the outermost texels to fade to nothing
  yield;
  for (let i = 0; i < g * g; i++) { if ((i & 32767) === 32767) yield; let a = Math.min(1, aSum[i]); if (a <= 0) continue;
    const x = i % g, y = (i / g) | 0, dEdge = Math.min(x, y, g - 1 - x, g - 1 - y); if (dEdge < edgeFade) { a *= dEdge / edgeFade; if (a <= 0) continue; } const inv = a / Math.max(aSum[i], 1e-6); rgba[4 * i] = Math.round(255 * rgbSum[3 * i] * inv); rgba[4 * i + 1] = Math.round(255 * rgbSum[3 * i + 1] * inv); rgba[4 * i + 2] = Math.round(255 * rgbSum[3 * i + 2] * inv); rgba[4 * i + 3] = Math.round(255 * a); }
  for (const r of regions) { const b = bbox.get(r.k); if (b) r.world = { x0: ex0 + b[0] / g * spanX, y0: ey0 + b[1] / g * spanY, x1: ex0 + (b[2] + 1) / g * spanX, y1: ey0 + (b[3] + 1) / g * spanY }; }
  const regionOf = (x, y) => { if (x < ex0 || x > ex1 || y < ey0 || y > ey1) return -1; const id = cl[gy(y) * g + gx(x)]; return id < 0 ? -1 : idToRegion[id]; };
  return { g, extent: { ex0, ey0, spanX, spanY }, cl, regions, rgba, regionOf, ms: performance.now() - t0, sigma, softSigma, mergeTau, minPoints };
}
function hex(c) { return [parseInt(c.slice(1, 3), 16) / 255, parseInt(c.slice(3, 5), 16) / 255, parseInt(c.slice(5, 7), 16) / 255]; }
