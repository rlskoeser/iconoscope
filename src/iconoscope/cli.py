from __future__ import annotations

import argparse
import sys
from math import sqrt
from pathlib import Path

import numpy as np

from iconoscope.embed import extract_features
from iconoscope.layout import assign_grid
from iconoscope.mosaic import render_mosaic
from iconoscope.reduce import reduce_to_2d


def find_images(image_dir: Path, ext: str | None = None) -> list[Path]:
    if ext:
        ext = ext.lstrip('.')   # ensure extension has one and only one .
        return sorted(image_dir.rglob(f"*.{ext}"))
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in extensions]


def cmd_embed(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        sys.exit(f"error: {image_dir} is not a directory")

    image_paths = find_images(image_dir, args.ext)
    if not image_paths:
        sys.exit(f"error: no images found in {image_dir}")

    print(f"Found {len(image_paths)} images")
    features, paths = extract_features(
        image_paths,
        backend=args.backend,
        batch_size=args.batch_size,
        device=args.device,
        show_progress=True,
    )
    path_strings = np.array([str(p) for p in paths])
    np.savez(args.output, features=features, paths=path_strings)
    print(
        f"Saved embeddings ({features.shape[0]} images, {features.shape[1]} dims) to {args.output}"
    )


def cmd_mosaic(args: argparse.Namespace) -> None:
    data = np.load(args.embeddings, allow_pickle=False)
    features = data["features"]
    path_strings = data["paths"]
    image_paths = [Path(str(p)) for p in path_strings]

    print(f"Loaded {len(image_paths)} embeddings from {args.embeddings}")

    N = features.shape[0]
    if args.thumb_size == "auto":
        # 1.1 ensures grid cells < N after integer division, so every cell gets filled
        thumb_size = max(1, int(sqrt(args.width * args.height / N) * 1.1))
        print(f"Auto thumb size: {thumb_size}px")
    else:
        thumb_size = args.thumb_size

    grid_cols = args.width // thumb_size
    grid_rows = args.height // thumb_size
    n_cells = grid_cols * grid_rows
    if args.thumb_size != "auto" and N < n_cells:
        print(
            f"Warning: only {N} images but grid has {n_cells} cells. "
            "Consider --thumb-size auto, increasing --thumb-size, or reducing --width/--height."
        )

    if args.load_coords:
        coords = np.load(args.load_coords)
        print(f"Loaded 2D coordinates from {args.load_coords}")
    else:
        coords = reduce_to_2d(
            features,
            backend=args.reducer,
            random_state=42,
        )
        print(f"Reduced to 2D with {args.reducer}")

    if args.save_coords:
        np.save(args.save_coords, coords)
        print(f"Saved 2D coordinates to {args.save_coords}")

    assignments = assign_grid(
        coords,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        method=args.layout,
    )
    print(
        f"Assigned {len(assignments)} images to grid "
        f"({grid_cols}x{grid_rows}) using {args.layout}"
    )

    render_mosaic(
        image_paths,
        assignments,
        canvas_width=args.width,
        canvas_height=args.height,
        thumb_size=thumb_size,
        output_path=Path(args.output),
        jpeg_quality=90,
    )
    print(f"Mosaic saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNN embedding mosaic: embed images and render a 2D layout mosaic"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser(
        "embed", help="Extract CNN features from images"
    )
    embed_parser.add_argument("image_dir", type=str, help="Directory of input images")
    embed_parser.add_argument(
        "-o", "--output", type=str, default="embeddings.npz", help="Output .npz path"
    )
    embed_parser.add_argument(
        "--backend",
        type=str,
        default="resnet50",
        choices=["resnet50", "clip"],
        help="Feature backbone",
    )
    embed_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    embed_parser.add_argument(
        "--ext", type=str, default=None, metavar="EXT",
        help="Limit to images with single extension, e.g. .jpeg (faster than scanning all types)",
    )
    embed_parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)"
    )

    mosaic_parser = subparsers.add_parser(
        "mosaic", help="Render a mosaic from embeddings"
    )
    mosaic_parser.add_argument(
        "embeddings", type=str, help="Path to .npz embeddings file"
    )
    mosaic_parser.add_argument(
        "-o", "--output", type=str, default="mosaic.jpg", help="Output image path"
    )
    mosaic_parser.add_argument(
        "--reducer",
        type=str,
        default="tsne",
        choices=["tsne", "umap"],
        help="Dimensionality reduction backend",
    )
    mosaic_parser.add_argument(
        "--layout",
        type=str,
        default="full_grid",
        choices=["full_grid", "first_come"],
        help="Grid assignment method",
    )
    mosaic_parser.add_argument(
        "--width", type=int, default=2000, help="Output canvas width in pixels"
    )
    mosaic_parser.add_argument(
        "--height", type=int, default=2000, help="Output canvas height in pixels"
    )
    mosaic_parser.add_argument(
        "--thumb-size",
        type=lambda v: v if v == "auto" else int(v),
        default=50,
        metavar="N|auto",
        help="Thumbnail size in pixels, or 'auto' to fit all images",
    )
    mosaic_parser.add_argument(
        "--save-coords", type=str, default=None, help="Save 2D coords to .npy"
    )
    mosaic_parser.add_argument(
        "--load-coords",
        type=str,
        default=None,
        help="Load pre-computed 2D coords from .npy",
    )

    args = parser.parse_args()

    if args.command == "embed":
        cmd_embed(args)
    elif args.command == "mosaic":
        cmd_mosaic(args)


if __name__ == "__main__":
    main()
