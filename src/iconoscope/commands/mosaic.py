import argparse

from iconoscope.dataset import ImageDataset
from iconoscope.mosaic import generate_mosaic


def main(args: argparse.Namespace) -> None:
    img_dataset = ImageDataset(storage_path=args.dataset)
    generate_mosaic(
        img_dataset,
        output=args.output,
        width=args.size.width,
        height=args.size.height,
        jpeg_quality=args.jpeg_quality,
        sample_size=args.limit,
    )
