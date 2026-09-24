import argparse
from importlib import import_module
from pathlib import Path


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


HANDLERS = {
    "embed": ("iconoscope.commands.embed", "main"),
    "info": ("iconoscope.commands.info", "main"),
    "mosaic": ("iconoscope.commands.mosaic", "main"),
    "cluster": ("iconoscope.commands.cluster", "main"),
}


def dispatch(args: argparse.Namespace) -> None:
    module_name, handler_name = HANDLERS[args.command]
    handler = getattr(import_module(module_name), handler_name)
    handler(args)


def main():
    parser = argparse.ArgumentParser(prog="iconoscope")
    subparsers = parser.add_subparsers(dest="command")

    ## embed : create dataset and extract features
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

    ## dataset info
    info_parser = subparsers.add_parser("info")
    info_parser.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file created by iconoscope embed (.hdf5)",
    )

    ## generate mosaic
    parser_mosaic = subparsers.add_parser("mosaic")
    parser_mosaic.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file produced by the embed command (.hdf5)",
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

    ## cluster image features
    cluster_parser = subparsers.add_parser("cluster")
    cluster_parser.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file created by iconoscope embed (.hdf5)",
    )
    cluster_parser.add_argument("n_clusters", type=int, help="Number of clusters")
    parser.set_defaults(func=dispatch)

    # parse arguments and call the appropriate method
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    args.func(args)


if __name__ == "__main__":
    main()
