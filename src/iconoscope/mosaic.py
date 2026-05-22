from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
from PIL import Image


def render_mosaic(
    image_paths: list[Path],
    assignments: dict[tuple[int, int], int],
    canvas_width: int = 2000,
    canvas_height: int = 2000,
    thumb_size: int = 50,
    output_path: Path | None = None,
    show: bool = False,
    jpeg_quality: int = 90,
) -> Image.Image:
    N = len(image_paths)
    grid_cols = canvas_width // thumb_size
    grid_rows = canvas_height // thumb_size

    auto_thumb = max(thumb_size, int(sqrt(canvas_width * canvas_height / max(N, 1)) * 1.1))
    if auto_thumb > thumb_size:
        thumb_size = auto_thumb
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
        ext = Path(output_path).suffix.lower()
        if ext in (".jpg", ".jpeg"):
            canvas.save(output_path, quality=jpeg_quality, optimize=True)
        else:
            canvas.save(output_path)

    if show:
        canvas.show()

    return canvas
