import argparse

from iconoscope.dataset import ImageDataset
from iconoscope.embed import extract_img_features


def main(args: argparse.Namespace) -> None:
    try:
        img_dataset = ImageDataset(storage_path=args.dataset)
        img_dataset.save_features(extract_img_features(img_dataset), "dinov2")
    except ValueError as err:
        raise SystemExit(err)
