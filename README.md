# iconoscope

Extract CNN/ViT features from images and render a 2D t-SNE/UMAP mosaic. Inspired by [Andrej Karpathy's CNN embedding visualizer](https://cs.stanford.edu/people/karpathy/cnnembed/).

## Install

```
pip install -e .
```

Optional backends:

```
pip install -e ".[umap]"   # UMAP reducer
pip install -e ".[clip]"   # CLIP embedding backend
```

## Usage

Two-stage pipeline that separates feature extraction from mosaic rendering.

### 1. Embed — extract CNN features

```
iconoscope embed ./my_images/ -o embeddings.npz
```

| Flag | Default | Description |
|---|---|---|
| `image_dir` | (required) | Directory of input images |
| `-o, --output` | `embeddings.npz` | Output `.npz` path |
| `--backend` | `resnet50` | `resnet50` or `clip` |
| `--batch-size` | `64` | Inference batch size |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |

Scans the directory for images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`), extracts 2048-d ResNet50 features (or 512-d CLIP), L2-normalizes, and saves to a documented `.npz` format.

### 2. Mosaic — render layout

```
iconoscope mosaic embeddings.npz -o mosaic.jpg --save-coords coords.npy
```

| Flag | Default | Description |
|---|---|---|
| `embeddings` | (required) | `.npz` file from the embed step |
| `-o, --output` | `mosaic.jpg` | Output image path |
| `--reducer` | `tsne` | `tsne` or `umap` |
| `--layout` | `full_grid` | `full_grid` (no gaps) or `first_come` (fast) |
| `--width` | `2000` | Canvas width in pixels |
| `--height` | `2000` | Canvas height in pixels |
| `--thumb-size` | `50` | Thumbnail size in pixels, or `auto` to fit all images |
| `--save-coords` | — | Save 2D coords to `.npy` |
| `--load-coords` | — | Skip reduction, load coords from `.npy` |
| `--show` | — | Display mosaic after rendering |

### 3. Iterate layout without re-reducing

```
iconoscope mosaic embeddings.npz -o mosaic_preview.jpg \
    --load-coords coords.npy --layout first_come --thumb-size 30
```

## Embeddings file format

Standard NumPy `.npz` with two arrays:

| Key | dtype | Shape | Description |
|---|---|---|---|
| `features` | `float32` | `(N, D)` | L2-normalized feature vectors |
| `paths` | `str` / `U` | `(N,)` | Image paths matching each row |

Compatible with any tool that produces this schema (e.g. a different model or language), as long as the dtype and key names match.

## Notes

- `full_grid` uses Hungarian assignment (globally optimal) for ≤5000 cells, KD-tree greedy beyond that
- `first_come` is fast but may leave gaps
- Thumbnail size auto-adjusts upward when there are fewer images than grid cells
