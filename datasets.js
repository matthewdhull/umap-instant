// Static datasets: a finished table instead of a live feed. Every row gets a vector (text through the same MiniLM encoder
// as the live feeds, images through DINOv2-small, table rows as standardised numbers, ASRS through its own 50-number
// TF-IDF features). The whole table is loaded (`?rows=` samples fewer) and every row is fitted into one UMAP map on the
// GPU, with no stream. Dragging the timeline back fits the map of the rows dated up to then (?until=); Play then places
// the rest in date order with Batched RevAdopt.
//   time      a table with dates is placed in date order on a real time axis ('month', 'year' or 'day': the unit the
//             timeline labels in); without one the rows go in a fixed shuffle and there is no timeline
//   category  the table's own label for a row (country, anomaly label, subject...): the topic colours and legend, as
//             Embedding Atlas colours its examples, instead of k-means clusters
//
// The hosted tables are the ones the Embedding Atlas examples load (https://apple.github.io/embedding-atlas/examples/),
// fetched from the same pinned URLs and filtered the same way (wine reviews: one row per distinct review).
// ASRS is not hosted anywhere public: browser_jetstream/pack_asrs.py packs it, with the aviation dashboard's display
// labels and their colours, from the aviation_safety_analytics checkout into data/jetstream_demo/static/asrs/ (never committed).
const HF = 'https://huggingface.co/datasets/';
const WOOF = ['Shih-Tzu', 'Rhodesian ridgeback', 'Beagle', 'English foxhound', 'Border terrier', 'Australian terrier', 'Golden retriever', 'Old English sheepdog', 'Samoyed', 'Dingo'];
// the movies table writes old films' two-digit years a century late (The Birth of a Nation, 1915, as 2015; Duel in the Sun,
// 1946, as 2046); its newest film is from 2011, so a later year is moved back a century
const movieDate = (r) => { const d = new Date(Date.parse(r['Release Date'] + ' UTC')); if (isNaN(d)) return null; if (d.getUTCFullYear() > 2011) d.setUTCFullYear(d.getUTCFullYear() - 100); return d; };
const join = (...xs) => xs.filter((x) => x !== null && x !== undefined && x !== '').join(' · ');

export const DATASETS = {
  // about 3 million arXiv papers (Cornell's metadata snapshot, CC0), prepared ahead of time on a GPU by arxiv/build_arxiv.py:
  // MiniLM vectors reduced to 64 principal components, and each paper's 30 nearest papers by the full vectors, so the browser
  // neither embeds nor searches; the files sit on a separate host (static-data/arxiv/info.json says where), never in Git
  arxiv: {
    label: 'arXiv papers', what: 'arXiv papers (titles and abstracts)', noun: 'papers', one: 'paper', kind: 'lists', dim: 64, capRows: 3220000,   // capRows: room for every paper (3,215,335) plus the 4,096 the page adds; at 3,200,000 the feature buffer outgrew the device limit asked for
    time: 'year', date: (r) => Date.UTC(r[2], 0), categoryName: 'field',
  },
  asrs: {
    label: 'ASRS safety reports', what: 'ASRS aviation safety reports', noun: 'reports', one: 'report', kind: 'vectors',
    local: '../data/jetstream_demo/static/asrs/', time: 'month', date: (r) => Date.UTC(+r.month.slice(0, 4), +r.month.slice(5, 7) - 1),
    category: (r) => r.cat, categoryName: 'anomaly label',
    // how label words are shown (names are computed lower case): the aviation acronyms in their usual casing
    caps: ['TCAS II', 'TCAS', 'TRACON', 'IMC', 'VMC', 'VFR', 'IFR', 'TFR', 'ATC', 'ATIS', 'ARTCC', 'CTAF', 'NOTAM', 'RNAV', 'ILS', 'VOR', 'DME', 'GPS', 'FMS', 'FMC', 'MCP', 'APU', 'MEL', 'QRH', 'ASRS', 'FAA', 'TA', 'RA', 'UAS', 'ADS-B', 'LOC', 'GS', 'PAPI', 'CFI', 'ATP', 'FBO', 'SID', 'STAR', 'NMAC', 'UAV', 'PDC'],   // a phrase also matches written as one word (tcasii)
  },
  'wine-reviews': {
    label: 'Wine reviews', what: 'wine reviews (Wine Enthusiast)', noun: 'reviews', one: 'review', kind: 'text',
    url: HF + 'spawn99/wine-reviews/resolve/e6b10f4db3091a6fed8c5b294c0cc885e7f6e99d/data/train-00000-of-00001.parquet', format: 'parquet',
    columns: ['title', 'country', 'province', 'description', 'points', 'price', 'variety', 'designation'],
    text: (r) => r.description, dedupe: (r) => [r.title, r.country, r.province, r.description, r.points, r.price, r.variety, r.designation].join('\u0001'),
    show: (r) => join(r.title, r.country, r.variety, r.points != null ? `${r.points} points` : '', r.price != null ? `$${r.price}` : ''),
    category: (r) => r.country, categoryName: 'country',
  },
  vispubdata: {
    label: 'Visualization publications', what: 'IEEE VIS papers (vispubdata)', noun: 'papers', one: 'paper', kind: 'text',
    url: 'https://raw.githubusercontent.com/pisenberg/vispubdata/46b3a2b6f3cffa3a60dd76153c4b58704b286fda/vispubdata-update/results/vispubdata-update.csv-aminer.csv',
    sha256: '41ff9f2461ee9f9e24da6acc7e5aca1c0e5f11623c9cc0ba3b095930d23b563b', format: 'csv', time: 'year', date: (r) => (+r.Year ? Date.UTC(+r.Year, 0) : null),
    text: (r) => r.Abstract ? `${r.Title}. ${r.Abstract}` : r.Title, show: (r) => join(r.Year, r.Conference, r.Title), link: (r) => r.Link,
    category: (r) => r.Conference, categoryName: 'conference',
  },
  medmcqa: {
    label: 'MedMCQA', what: 'medical exam questions (MedMCQA)', noun: 'questions', one: 'question', kind: 'text',
    url: HF + 'openlifescienceai/medmcqa/resolve/91c6572c454088bf71b679ad90aa8dffcd0d5868/data/train-00000-of-00001.parquet', format: 'parquet',
    columns: ['question', 'subject_name', 'topic_name'], text: (r) => r.question, show: (r) => join(r.subject_name, r.topic_name),
    category: (r) => r.subject_name, categoryName: 'subject',
  },
  SuperGPQA: {
    label: 'SuperGPQA', what: 'graduate-level questions (SuperGPQA)', noun: 'questions', one: 'question', kind: 'text',
    url: HF + 'm-a-p/SuperGPQA/resolve/4430d4458112c7d4497fdcf94d7cc223313d6acf/SuperGPQA-all.jsonl',
    sha256: '28b998e70205ee95e540317b5adc06a06552a3961fb50b153df126b833f7a910', format: 'jsonl',
    text: (r) => r.question, show: (r) => join(r.discipline, r.field, r.subfield), category: (r) => r.discipline, categoryName: 'discipline',
  },
  ScienceQA: {
    label: 'ScienceQA', what: 'school science questions (ScienceQA)', noun: 'questions', one: 'question', kind: 'text',
    url: HF + 'derek-thomas/ScienceQA/resolve/f18b0a70359ebfb41f658fd564208d0355b013f4/data/train-00000-of-00001-1028f23e353fbe3e.parquet', format: 'parquet',
    columns: ['question', 'hint', 'subject', 'topic', 'grade'], text: (r) => r.hint ? `${r.question} ${r.hint}` : r.question, show: (r) => join(r.subject, r.topic, r.grade),
    category: (r) => r.topic, categoryName: 'topic',
  },
  imagewoof: {
    label: 'ImageWoof', what: 'dog photos (ImageWoof)', noun: 'images', one: 'image', kind: 'image',
    url: HF + 'frgfm/imagewoof/resolve/refs%2Fconvert%2Fparquet/160px/train/0000.parquet', format: 'parquet',
    columns: ['image', 'label'], show: (r) => WOOF[Number(r.label)] || `label ${r.label}`, category: (r) => WOOF[Number(r.label)], categoryName: 'breed',
  },
  movies: {
    label: 'Movies', what: 'films (vega-datasets movies)', noun: 'films', one: 'film', kind: 'table',
    url: 'https://cdn.jsdelivr.net/npm/vega-datasets@3.2.1/data/movies.json', sha256: 'e63c499759e3b07b49563e036f55290f87feb56def8703ec049ca305ab1523d3', format: 'json',
    time: 'year', date: (r) => movieDate(r)?.getTime() ?? null,
    numeric: ['US Gross', 'Worldwide Gross', 'Production Budget', 'Running Time min', 'IMDB Rating', 'IMDB Votes', 'Rotten Tomatoes Rating'],
    categorical: ['MPAA Rating', 'Major Genre', 'Creative Type', 'Source'],
    levels: { 'US Gross': ['small gross', 'big gross'], 'Production Budget': ['low budget', 'big budget'], 'IMDB Rating': ['poorly rated', 'well rated'], 'Running Time min': ['short films', 'long films'] },
    show: (r) => join(r.Title, movieDate(r)?.getUTCFullYear(), r['Major Genre']), category: (r) => r['Major Genre'], categoryName: 'genre',
  },
  'census-income': {
    label: 'Census income', what: 'adults in the 1994 US census (adult income)', noun: 'people', one: 'person', kind: 'table',
    url: HF + 'scikit-learn/adult-census-income/resolve/fbeef6ec0e6fd88a5028b94683144000a6b380d5/adult.csv', sha256: '250e154ed75714ae57a564926d66c6319cd6aac1bcd32774cc76841a88d74e53', format: 'csv',
    numeric: ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week'],
    categorical: ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'income'],
    levels: { age: ['younger', 'older'], 'hours.per.week': ['part time', 'long hours'], 'education.num': ['less schooling', 'more schooling'] },
    show: (r) => join(r.age && `age ${r.age}`, r.sex, r.occupation, r.income), category: (r) => r.occupation, categoryName: 'occupation',
  },
  'california-housing': {
    label: 'California housing', what: 'California census blocks (housing)', noun: 'blocks', one: 'block', kind: 'table',
    url: HF + 'gvlassis/california_housing/resolve/17110e600427385a20e8358b00adc9b032d185ff/data/train-00000-of-00001.parquet', format: 'parquet',
    numeric: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal'], categorical: [],
    levels: { MedInc: ['low income', 'high income'], HouseAge: ['newer houses', 'older houses'], AveRooms: ['fewer rooms', 'more rooms'], AveBedrms: ['fewer bedrooms', 'more bedrooms'], Population: ['sparse blocks', 'crowded blocks'], AveOccup: ['small households', 'large households'], Latitude: ['south', 'north'], Longitude: ['west', 'east'], MedHouseVal: ['cheap homes', 'expensive homes'] },
    show: (r) => join(`median income ${(+r.MedInc).toFixed(1)}`, `house value ${(+r.MedHouseVal).toFixed(2)}`, `${(+r.Latitude).toFixed(2)}, ${(+r.Longitude).toFixed(2)}`),
  },
};

// text models to compare (the page's model picker, ?model=): all give 384 numbers per text, so the GPU buffers stay the same
// size; `id` is the transformers.js (ONNX) copy, `name` the original, `pool` how token vectors become one text vector
export const TEXT_MODELS = {
  minilm: { label: 'MiniLM-L6 (sentence-transformers)', id: 'Xenova/all-MiniLM-L6-v2', name: 'sentence-transformers/all-MiniLM-L6-v2', pool: 'mean' },
  bge: { label: 'BGE-small (BAAI)', id: 'Xenova/bge-small-en-v1.5', name: 'BAAI/bge-small-en-v1.5', pool: 'cls' },
  gte: { label: 'GTE-small (Alibaba)', id: 'Xenova/gte-small', name: 'thenlper/gte-small', pool: 'mean' },
};
// image models to compare (the same picker for an image dataset): DINOv2 learned from pictures alone, CLIP from pictures
// with their captions; `dim` differs (384, 512), which a static dataset's buffers allow (they are sized when it loads)
export const IMAGE_MODELS = {
  dinov2: { label: 'DINOv2-small (Meta)', id: 'Xenova/dinov2-small', name: 'facebook/dinov2-small', dim: 384 },
  clip: { label: 'CLIP ViT-B/32 (OpenAI)', id: 'Xenova/clip-vit-base-patch32', name: 'openai/clip-vit-base-patch32', dim: 512 },
};
// the picker's models for a dataset kind, and the one used when ?model= names none of them
export const modelsFor = (kind) => (kind === 'text' ? TEXT_MODELS : kind === 'image' ? IMAGE_MODELS : null);
export const defaultModel = (kind) => (kind === 'text' ? 'minilm' : kind === 'image' ? 'dinov2' : null);

// the data menu's order (quick to load and worth a look first) and each dataset's first-visit load time in seconds, measured
// on the public site from an empty browser cache (Apple M-series Mac, 2026-10-24); a repeat visit reads this browser's cache
export const MENU_ORDER = ['vispubdata', 'arxiv', 'imagewoof', 'movies', 'SuperGPQA', 'ScienceQA', 'asrs', 'california-housing', 'census-income', 'wine-reviews', 'medmcqa'];
export const LOAD_S = { vispubdata: 8, imagewoof: 11, movies: 7, SuperGPQA: 12, ScienceQA: 15, asrs: 24, 'california-housing': 7, 'census-income': 17, 'wine-reviews': 37, medmcqa: 37 };
