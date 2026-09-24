import argparse

from iconoscope.dataset import ImageDataset
from iconoscope.embed import extract_img_features


def main(args: argparse.Namespace) -> None:
    try:
        img_dataset = ImageDataset(
            storage_path=args.output_path,
            image_dir=args.image_dir,
            max_images=args.max,
        )
        img_dataset.save_features(extract_img_features(img_dataset), "dinov2")
    except ValueError as err:
        raise SystemExit(err)
