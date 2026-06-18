import warnings
from pathlib import Path

import numpy as np
import polars as pl
import umap
from lap import lapjv
from PIL import Image
from sklearn.decomposition import PCA


def reduce_features(features: np.ndarray) -> np.ndarray:
    """Takes an array of embedding feature vectors and returns normalized coordinates.
    Uses PCA to reduce, UMAP to transform to two dimensions, then normalize from 0 to 1 for
    both axes. Returns an array of x,y coordinates for each feature vector in the input."""
    # use PCA to reduce vectors from 768 to 50 (but handle small datasets < 50)
    n_components = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=n_components).fit_transform(features)
    # use umap to project the reduced vectors into two dimensions
    coords = umap.UMAP(n_components=2).fit_transform(reduced)

    # determine smalleest and largest coordinates, and then
    # scale all coordinates to normalize from 0 to 1.0
    min_coords, max_coords = coords.min(0), coords.max(0)
    span = np.where(max_coords - min_coords > 0, max_coords - min_coords, 1.0)
    return (coords - min_coords) / span


def assign_grid(
    coords: np.ndarray, grid_cols: int, grid_rows: int
) -> dict[tuple[int, int], int]:
    """Assign N images to grid cells via lapjv (Jonker-Volgenant).

    Returns {(row, col): img_idx}. Cells with no image are omitted.
    """

    N = len(coords)
    n_cells = grid_cols * grid_rows

    grid_cells = np.array(
        [(r, c) for r in range(grid_rows) for c in range(grid_cols)],
        dtype=np.float32,
    )
    cell_centers = (grid_cells + 0.5) / np.array(
        [[grid_rows, grid_cols]], dtype=np.float32
    )

    if N < n_cells:
        padding = np.full((n_cells - N, 2), 0.5, dtype=np.float32)
        padded = np.vstack([coords, padding])
    else:
        padded = coords[:n_cells]

    aspect = grid_cols / grid_rows
    scale = np.array([[aspect, 1.0]], dtype=np.float32)
    cost = np.linalg.norm(
        (padded * scale)[:, np.newaxis] - (cell_centers * scale)[np.newaxis],
        axis=2,
    ).astype(np.float64)

    # lap.lapjv returns (opt_cost, x, y): x[img]=cell, y[cell]=img
    _, _, col_ind = lapjv(cost)

    return {
        (int(grid_cells[cell_idx][0]), int(grid_cells[cell_idx][1])): img_idx
        for cell_idx, img_idx in enumerate(col_ind)
        if img_idx < N
    }


def run_mosaic(
    embeddings: Path,
    output: Path | None = None,
    width: int = 2000,
    height: int = 2000,
    thumb_size: int = 50,
    jpeg_quality: int = 90,
) -> None:
    if output is None:
        output = embeddings.with_suffix(".jpg")

    coords_path = embeddings.with_suffix(".coords.parquet")

    df = pl.read_parquet(embeddings)
    paths = df["image_path"].to_list()
    features = np.array(df["features"].to_list(), dtype=np.float32)

    if coords_path.exists():
        print(f"Loading cached coords from {coords_path}")
        coords = pl.read_parquet(coords_path).to_numpy().astype(np.float32)
    else:
        print(f"Running UMAP on {len(paths)} images…")
        coords = reduce_features(features)
        pl.DataFrame(coords, schema=["x", "y"]).write_parquet(coords_path)
        print(f"Saved coords to {coords_path}")

    grid_cols = width // thumb_size
    grid_rows = height // thumb_size
    print(f"Assigning {len(paths)} images to {grid_cols}×{grid_rows} grid…")
    assignments = assign_grid(coords, grid_cols, grid_rows)

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    for (row, col), img_idx in assignments.items():
        try:
            thumb = (
                Image.open(paths[img_idx])
                .convert("RGB")
                .resize((thumb_size, thumb_size), Image.LANCZOS)
            )
            canvas.paste(thumb, (col * thumb_size, row * thumb_size))
        except Exception as exc:
            warnings.warn(f"Could not load {paths[img_idx]}: {exc}")

    canvas.save(output, quality=jpeg_quality)
    print(f"Saved mosaic to {output}")
