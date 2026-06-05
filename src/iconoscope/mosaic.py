from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
from PIL import Image

from iconoscope.layout import assign_grid
from iconoscope.reduce import reduce_to_2d


def run_mosaic(
    embeddings: Path,
    output: Path,
    reducer: str = "tsne",
    layout: str = "full_grid",
    width: int = 2000,
    height: int = 2000,
    thumb_size: int | str = 50,
    load_coords: Path | None = None,
    save_coords: Path | None = None,
    jpeg_quality: int = 90,
    show_progress: bool = True,
) -> None:
    data = np.load(embeddings, allow_pickle=False)
    features = data["features"]
    image_paths = [Path(str(p)) for p in data["paths"]]
    N = features.shape[0]

    if show_progress:
        print(f"Loaded {N} embeddings from {embeddings}")

    if thumb_size == "auto":
        # 1.1 ensures grid cells < N after integer division, so every cell gets filled
        thumb_size = max(1, int(sqrt(width * height / N) * 1.1))
        if show_progress:
            print(f"Auto thumb size: {thumb_size}px")

    grid_cols = width // thumb_size
    grid_rows = height // thumb_size
    n_cells = grid_cols * grid_rows
    if N < n_cells:
        print(f"Warning: only {N} images but grid has {n_cells} cells. "
              "Consider --thumb-size auto, increasing --thumb-size, or reducing --width/--height.")

    if load_coords is not None:
        coords = np.load(load_coords)
        if show_progress:
            print(f"Loaded 2D coordinates from {load_coords}")
    else:
        coords = reduce_to_2d(features, reducer=reducer, random_state=42)
        if show_progress:
            print(f"Reduced to 2D with {reducer}")

    if save_coords is not None:
        np.save(save_coords, coords)
        if show_progress:
            print(f"Saved 2D coordinates to {save_coords}")

    assignments = assign_grid(coords, grid_cols=grid_cols, grid_rows=grid_rows, method=layout)
    if show_progress:
        print(f"Assigned {len(assignments)} images to grid ({grid_cols}x{grid_rows}) using {layout}")

    render_mosaic(
        image_paths,
        assignments,
        canvas_width=width,
        canvas_height=height,
        thumb_size=thumb_size,
        output_path=output,
        jpeg_quality=jpeg_quality,
    )
    if show_progress:
        print(f"Mosaic saved to {output}")


def render_mosaic(
    image_paths: list[Path],
    assignments: dict[tuple[int, int], int],
    canvas_width: int = 2000,
    canvas_height: int = 2000,
    thumb_size: int = 50,
    output_path: Path | None = None,
    jpeg_quality: int = 90,
) -> Image.Image:
    grid_cols = canvas_width // thumb_size
    grid_rows = canvas_height // thumb_size

    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    thumb_w = canvas_width // grid_cols
    thumb_h = canvas_height // grid_rows

    for (r, c), img_idx in assignments.items():
        if img_idx >= len(image_paths):
            continue
        try:
            img = Image.open(image_paths[img_idx]).convert("RGB")
            img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
            x = c * thumb_w
            y = r * thumb_h
            canvas.paste(img, (x, y))
        except Exception:
            pass

    if output_path is not None:
        ext = output_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            canvas.save(output_path, quality=jpeg_quality, optimize=True)
        else:
            canvas.save(output_path)

    return canvas
