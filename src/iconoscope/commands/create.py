import argparse

from iconoscope.dataset import ImageDataset


def main(args: argparse.Namespace) -> None:
    try:
        ImageDataset.create(
            args.image_dir,
            args.output_path,
            max_images=args.max,
        )
    except ValueError as err:
        raise SystemExit(err)
