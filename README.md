# UMAP Instant

A UMAP map that grows as data arrives, in the browser on WebGPU. New items (Bluesky posts, Wikipedia edits, Nostr notes,
Hacker News items) are embedded in the browser and placed beside their nearest neighbours on the existing map as they
arrive; static datasets (aviation safety reports, wine reviews, exam questions, papers, dog photos, tables) are fitted as
one map, can be rewound on their timeline and played forward.

Open `index.html` (GitHub Pages). Needs a browser with WebGPU (Chrome or Edge on desktop).

- `?source=` picks a dataset: asrs, wine-reviews, vispubdata, medmcqa, SuperGPQA, ScienceQA, imagewoof, movies,
  census-income, california-housing, or a live feed (wikipedia, nostr, hackernews); none = live Bluesky.
- `static-data/` holds the ASRS reports (public NASA ASRS synopses, labelled as in the aviation safety dashboard) and
  precomputed vectors (MiniLM, BGE and GTE text, DINOv2 and CLIP images, float16) and each model's finished map for the
  hosted datasets; the tables themselves are read
  from Hugging Face and the other hosts the Embedding Atlas examples use.

This repository is a build output of the research code (streaming-umap, browser_jetstream/build_site.py); edit there.
