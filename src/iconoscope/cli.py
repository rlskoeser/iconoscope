import argparse
from pathlib import Path

import polars as pl

from iconoscope.embed import extract_img_features
from iconoscope.mosaic import generate_mosaic


def main_embed(args: argparse.Namespace):
    if not args.image_dir.is_dir():
        raise SystemExit(f"{args.image_dir} is not a directory")
    extract_img_features(args.image_dir, args.output_path)


def main_mosaic(args: argparse.Namespace):
    generate_mosaic(
        embeddings_path=args.embeddings,
        output=args.output,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
    )


def embeddings_info(args: argparse.Namespace):
    if not args.embeddings.is_file():
        raise SystemExit(f"{args.embeddings} is not a file")
    df = pl.read_parquet(args.embeddings)

    info_details = [f"Details for {args.embeddings} :"]

    if "image_path" in df.columns and "features" in df.columns:
        info_details.append(f"  {df.height:,} images with features extracted")
    else:
        info_details.append(" image_path and features columns not found")

    if "umap" in df.columns:
        info_details.append("  UMAP coordinates")

    print("\n".join(info_details))


def main():
    parser = argparse.ArgumentParser(prog="iconoscope")
    subparsers = parser.add_subparsers()

    parser_embed = subparsers.add_parser("embed")
    parser_embed.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing images to embed (can be nested)",
    )
    parser_embed.add_argument(
        "output_path",
        type=Path,
        help="File path for saved embeddings (.parquet)",
    )
    parser_embed.set_defaults(func=main_embed)

    info_parser = subparsers.add_parser("info")
    info_parser.add_argument(
        "embeddings",
        type=Path,
        help="Embeddings file produced by iconoscope embed (.parquet)",
    )
    info_parser.set_defaults(func=embeddings_info)

    parser_mosaic = subparsers.add_parser("mosaic")
    parser_mosaic.add_argument(
        "embeddings",
        type=Path,
        help="Embeddings file produced by the embed command (.parquet)",
    )
    parser_mosaic.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path (default: embeddings stem + .jpg)",
    )
    parser_mosaic.add_argument("--width", type=int, default=2000)
    parser_mosaic.add_argument("--height", type=int, default=2000)
    parser_mosaic.add_argument(
        "--jpeg-quality", type=int, default=90, dest="jpeg_quality"
    )
    parser_mosaic.set_defaults(func=main_mosaic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
