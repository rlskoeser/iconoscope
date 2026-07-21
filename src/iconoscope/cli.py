import argparse
from pathlib import Path


from iconoscope.embed import extract_img_features
from iconoscope.mosaic import generate_mosaic
from iconoscope.storage import info


def main_embed(args: argparse.Namespace):
    if not args.image_dir.is_dir():
        raise SystemExit(f"{args.image_dir} is not a directory")
    extract_img_features(args.image_dir, args.output_path, args.max)


def main_mosaic(args: argparse.Namespace):
    generate_mosaic(
        embeddings_path=args.embeddings,
        output=args.output,
        width=args.size.width,
        height=args.size.height,
        jpeg_quality=args.jpeg_quality,
        sample_size=args.limit,
    )


def embeddings_info(args: argparse.Namespace):
    if not args.embeddings.is_file():
        raise SystemExit(f"{args.embeddings} is not a file")
    details = info(args.embeddings)

    info_details = [f"Details for {args.embeddings} :"]

    if "image_paths" in details:
        info_details.append(f"  {details['image_paths']:,} image paths")
    else:
        info_details.append(" image_path and features columns not found")

    for model, model_details in details["features"].items():
        # convert shape tuple into a readable dimension string
        embed_dimensions = "x".join([str(dim) for dim in model_details["embeddings"]])
        info_details.append(f"  {model} extracted features ({embed_dimensions})")

    # if "umap" in df.columns:
    #     info_details.append("  UMAP coordinates")

    print("\n".join(info_details))


def size_tuple(size_str: str | int) -> argparse.Namespace:
    try:
        parts = [int(side) for side in str(size_str).split("x")]
    except ValueError as err:
        raise ValueError(f"Could not parse `{size_str}` as a size: {err}") from err
    # first portion is the width
    width = parts.pop(0)
    # if a second value is present, use as height; otherwise width
    height = parts.pop() if parts else width
    # if there are parts leftover, raise value error
    if parts:
        raise ValueError(f"Could not parse `{size_str}` as a size")
    return argparse.Namespace(width=width, height=height)


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
        help="File path for saved embeddings (.hdf5)",
    )
    parser_embed.add_argument(
        "-m", "--max", type=int, help="Limit to specified number of images"
    )
    parser_embed.set_defaults(func=main_embed)

    info_parser = subparsers.add_parser("info")
    info_parser.add_argument(
        "embeddings",
        type=Path,
        help="Embeddings file produced by iconoscope embed (.hdf5)",
    )
    info_parser.set_defaults(func=embeddings_info)

    parser_mosaic = subparsers.add_parser("mosaic")
    parser_mosaic.add_argument(
        "embeddings",
        type=Path,
        help="Embeddings file produced by the embed command (.hdf5)",
    )
    parser_mosaic.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path (default: embeddings stem + .jpg)",
    )

    parser_mosaic.add_argument(
        "-s",
        "--size",
        type=size_tuple,
        default="2000",
        help="Output image size; specify a single dimension for a square or both width and height as 1000x250. (default: %(default)s)",
    )
    parser_mosaic.add_argument(
        "--jpeg-quality", type=int, default=90, dest="jpeg_quality"
    )
    parser_mosaic.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Limit to sample of images of the specified size",
    )
    parser_mosaic.set_defaults(func=main_mosaic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
