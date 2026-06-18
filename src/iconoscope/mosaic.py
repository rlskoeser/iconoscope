import math
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import umap
from lap import lapjv
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
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


LAPJV_CELL_LIMIT = 5000


def assign_to_grid(
    coords: np.ndarray, grid_cols: int, grid_rows: int
) -> dict[tuple[int, int], int]:
    """Given an array of x,y coordinates as returned from :meth:`reduce_features` and
    a grid size based on columns and rows, assign each item in the coordinates
    array to a cell in the grid. Uses lapjv (Jonker-Volgenant) to determine the best fit
    for grids up to LAPJV_CELL_LIMIT cells; falls back to greedy KD-tree for larger grids.

    Returns a dictionary of the grid assignment for each input item, based on index
    in the original coords array.
        {(row, col): item_idx}. Cells with no image are omitted.
    """

    # determine number of items to be positioned based on number of x,y coordinates
    n_items = len(coords)
    # determine number of cells in the grid
    n_cells = grid_cols * grid_rows

    # warn if grid size results in any images being omitted
    if n_items > n_cells:
        print(f"Warning: omitting {n_items - n_cells:,} images from the grid")

    # generate an array of grid coordinates for the requested grid size
    grid_cells = np.array(
        [(r, c) for r in range(grid_rows) for c in range(grid_cols)],
        dtype=np.float32,
    )
    # generate a grid of center-cell coordinates from 0.0 to 1.0, for placement
    cell_centers = (grid_cells + 0.5) / np.array(
        [[grid_rows, grid_cols]], dtype=np.float32
    )
    # determine aspect ratio for the requested grid
    aspect = grid_cols / grid_rows
    # scale the cell center coordinates based on the aspect ratio of the requested grid
    scale = np.array([[aspect, 1.0]], dtype=np.float32)

    if n_cells <= LAPJV_CELL_LIMIT:
        # lapjv requires a square cost matrix; if there are more cells than items,
        # add padding items to fill out the grid
        if n_items < n_cells:
            padding = np.full((n_cells - n_items, 2), 0.5, dtype=np.float32)
            square_coords = np.vstack([coords, padding])
        else:
            square_coords = coords[:n_cells]

        # compute pairwise Euclidean distances to create cost matrix for lapjv
        cost = cdist(square_coords * scale, cell_centers * scale).astype(np.float64)
        # lap.lapjv returns a tuple of optional cost, reverse mapping, and column index.
        # The column index is the assigned mapping: array of item indices assigned to each cell
        _, _, col_ind = lapjv(cost)
        # use lapjv column index to map back to column/row placement in the grid
        return {
            # take lapjv assigned slot for each image and map to grid position
            # decompose the flat array of grid cells back into row,col format
            (int(grid_cells[cell_idx][0]), int(grid_cells[cell_idx][1])): item_idx
            for cell_idx, item_idx in enumerate(col_ind)
            if item_idx < n_items  # omit any padding items needed to make square
        }

    else:
        # build spatial index of the grid cells for quick lookup of nearest cells
        tree = cKDTree(cell_centers * scale)
        # scale image coordinates to the grid; omit any beyond number of available slots
        # NOTE: important to truncate coords to n_cells or loop will never finish
        scaled_coords = coords[:n_cells] * scale
        # keep track of grid cells as they are claimed
        used: set[int] = set()
        # return structure, same as for lapjv: tuple of row,col in grid and image index
        assignments: dict[tuple[int, int], int] = {}
        k = min(n_cells, 50)
        for item_idx in range(len(scaled_coords)):
            # query for the k nearest cells
            while True:
                _, cell_indices = tree.query(scaled_coords[item_idx], k=k)
                # choose the first cell that is not already used
                chosen = next((i for i in cell_indices if i not in used), None)
                # stop looping as soon as grid cell is chosen
                if chosen is not None:
                    break
                # if nearest cells are already taken, expand the search - quadruple k and try again
                k = min(k * 4, n_cells)
            used.add(chosen)
            # add the chosen grid x,y as integers to the assignment dict for item index
            assignments[(int(grid_cells[chosen][0]), int(grid_cells[chosen][1]))] = (
                item_idx
            )
        return assignments


MIN_THUMB_SIZE = 20
MAX_THUMB_SIZE = 350


def generate_mosaic(
    embeddings_path: Path,
    output: Path | None = None,
    width: int = 2000,
    height: int = 2000,
    jpeg_quality: int = 90,
) -> None:
    """Generate a mosaic of images based on previous calculated image embeddings.
    Load image feature vectors from the embeddings parquet file, reduce with PCA+UMAP,
    update parquet file with umap coordinates, then assign to grid slots and create
    a mosaic image of thumbnails.
    """

    if output is None:
        output = embeddings_path.with_suffix(".jpg")

    # load image embeddings from saved parquet file
    df = pl.read_parquet(embeddings_path)
    # if coordinates have not already been calculated, reduce and store results
    if "umap" not in df.columns:
        print(f"Running UMAP on {df.height} image embeddings…")
        df = df.with_columns(umap=reduce_features(df["features"].to_numpy()))
        print(f"Updated {embeddings_path} with UMAP coordinates")
        df.write_parquet(embeddings_path)  # save the result
    else:
        print(f"Using existing UMAP coordinates in {embeddings_path}")

    paths = df["image_path"].to_list()

    # determine ideal thumbnail size based on mosaic size and number of images
    n_images = df.height
    # round to integer since we need pixel size; don't go beyond predefined max/min sizes
    thumb_size = max(
        MIN_THUMB_SIZE,
        min(round(math.sqrt((width * height) / n_images)), MAX_THUMB_SIZE),
    )
    # in future, if adding support for non-square thumbnails, calculate based on aspect ratio
    # thumb_w * thumb_h * N ≈ W * H
    # Fix the aspect ratio r = thumb_w / thumb_h, substitute thumb_w = r * thumb_h, and solve:
    # thumb_h = sqrt((W * H) / (N * r))
    # thumb_w = r * thumb_h

    # determine number of grid slots based on the desired image size
    grid_cols = width // thumb_size
    grid_rows = height // thumb_size
    # determine actual size based on determined thumbnail size and use that for the mosaic canvas
    actual_width = grid_cols * thumb_size
    actual_height = grid_rows * thumb_size

    assignments = assign_to_grid(df["umap"].to_numpy(), grid_cols, grid_rows)

    # create a blank canvas for the calculated size
    canvas = Image.new("RGB", (actual_width, actual_height), color=(255, 255, 255))
    for (row, col), img_idx in assignments.items():
        try:
            # open the image and resize to desired thumbnail size
            thumb = (
                Image.open(paths[img_idx])
                .convert("RGB")
                .resize((thumb_size, thumb_size), Image.LANCZOS)
                # resize with Lanczos (sinc) to minimize aliasing & artifacts
            )
            # paste the thumbnail on the grid in the appropriate slot
            canvas.paste(thumb, (col * thumb_size, row * thumb_size))
        except Exception as exc:
            warnings.warn(f"Could not load {paths[img_idx]}: {exc}")

    canvas.save(output, quality=jpeg_quality)
    print(f"Saved mosaic to {output}")
