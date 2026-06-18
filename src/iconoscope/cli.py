import argparse
from pathlib import Path

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
        thumb_size=args.thumb_size,
        jpeg_quality=args.jpeg_quality,
    )


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
    parser_mosaic.add_argument("--thumb-size", type=int, default=50, dest="thumb_size")
    parser_mosaic.add_argument(
        "--jpeg-quality", type=int, default=90, dest="jpeg_quality"
    )
    parser_mosaic.set_defaults(func=main_mosaic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
