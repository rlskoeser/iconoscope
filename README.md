# iconoscope

`iconoscope` is a python package for exploring image collections using visual similarity based on image embeddings extracted from models like CLIP and DINOv2/3. Embeddings can be used to generate a mosaic where visually similar images appear near each other, or for interactive exploration in a preliminary HTML viewer.

The name **iconoscope** comes from the Greek words for _image_ (εἰκών) and _to see_ (σκοπεῖν). The [iconoscope](https://en.wikipedia.org/wiki/Iconoscope) was the first practical video camera tube used in early television cameras.

This package was inspired by [Andrej Karpathy's CNN embedding visualizer](https://cs.stanford.edu/people/karpathy/cnnembed/). Version 0.1 started as a python port of Karpathy's cnnembed Matlab code created with Claude Code.

## Install

```bash
pip install -e .
pip install -e ".[umap]"   # UMAP reducer
pip install -e ".[clip]"   # CLIP embedding backend
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv sync --extra umap
```

## Usage

Two-stage pipeline: embed first (slow, GPU-accelerated), then mosaic (fast, iterate freely).

### 1. Embed

Extract features from a directory of images and save to an HDF5 file:

```bash
iconoscope embed ./images/ mycollection.h5
```

### 2. Mosaic

Reduce embeddings to 2D and render a mosaic image:

```bash
iconoscope mosaic mycollection.h5
```

Output defaults to `mycollection.jpg`. Coordinates are cached in the HDF5 file so subsequent runs skip the slow dimensionality reduction step.

### 3. Viewer

Generate an interactive HTML viewer:

```bash
iconoscope viewer mycollection.h5
```

Writes `mycollection.html`. Requires a pre-built Svelte app (`cd viewer && npm install && npm run build`).

### Other commands

```bash
iconoscope info mycollection.h5          # inspect what's stored in an embeddings file
iconoscope cluster mycollection.h5 -k 12 # k-means clustering on embeddings
```

## Embeddings file

The `.h5` file is the hand-off between pipeline stages. It holds:

- `features` — L2-normalized float32 embeddings `[N, D]`
- `paths` — image paths `[N]`
- `coords` — cached 2D coordinates `[N, 2]` (written after the first mosaic run)
- cluster assignments (written by `cluster`)

Run `iconoscope info` to inspect the contents of any embeddings file.
