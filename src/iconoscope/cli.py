from __future__ import annotations

import argparse
from pathlib import Path

from iconoscope.embed import run_embed
from iconoscope.mosaic import run_mosaic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNN embedding mosaic: embed images and render a 2D layout mosaic"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Extract CNN features from images")
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
        help="Limit to images with single extension, e.g. .jpeg",
    )
    embed_parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)"
    )

    mosaic_parser = subparsers.add_parser("mosaic", help="Render a mosaic from embeddings")
    mosaic_parser.add_argument("embeddings", type=str, help="Path to .npz embeddings file")
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
    mosaic_parser.add_argument("--width", type=int, default=2000, help="Output canvas width in pixels")
    mosaic_parser.add_argument("--height", type=int, default=2000, help="Output canvas height in pixels")
    mosaic_parser.add_argument(
        "--thumb-size",
        type=lambda v: v if v == "auto" else int(v),
        default=50,
        metavar="N|auto",
        help="Thumbnail size in pixels, or 'auto' to fit all images",
    )
    mosaic_parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG output quality (1-95)")
    mosaic_parser.add_argument("--save-coords", type=str, default=None, help="Save 2D coords to .npy")
    mosaic_parser.add_argument(
        "--load-coords", type=str, default=None, help="Load pre-computed 2D coords from .npy"
    )

    args = parser.parse_args()
    match args.command:
        case "embed":
            run_embed(
                image_dir=Path(args.image_dir),
                output=Path(args.output),
                backend=args.backend,
                batch_size=args.batch_size,
                ext=args.ext,
                device=args.device,
            )
        case "mosaic":
            run_mosaic(
                embeddings=Path(args.embeddings),
                output=Path(args.output),
                reducer=args.reducer,
                layout=args.layout,
                width=args.width,
                height=args.height,
                thumb_size=args.thumb_size,
                load_coords=Path(args.load_coords) if args.load_coords else None,
                save_coords=Path(args.save_coords) if args.save_coords else None,
                jpeg_quality=args.jpeg_quality,
            )


if __name__ == "__main__":
    main()
