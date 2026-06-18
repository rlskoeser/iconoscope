import argparse
from pathlib import Path

from iconoscope.embed import extract_img_features


def main_embed(args: argparse.Namespace):
    if not args.image_dir.is_dir():
        raise SystemExit(f"{args.image_dir} is not a directory")

    # check if output file exists? does not end in parque?
    #
    extract_img_features(args.image_dir, args.output_path)


def main():
    parser = argparse.ArgumentParser(prog="iconoscope")
    subparsers = parser.add_subparsers()  # could add help

    # create the parser for the "embed" command
    parser_embed = subparsers.add_parser("embed")  # could add help
    parser_embed.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing images to be embed (can be nested)",
    )
    parser_embed.add_argument(
        "output_path",
        type=Path,
        help="File name where embeddings should be saved (currently only supports .parquet)",
    )
    parser_embed.set_defaults(func=main_embed)

    # parse args and call the default function with the parsed arguments
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
