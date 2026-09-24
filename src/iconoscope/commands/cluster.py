import argparse

from iconoscope.cluster import identify_clusters
from iconoscope.dataset import ImageDataset


def main(args: argparse.Namespace) -> None:
    img_dataset = ImageDataset(storage_path=args.dataset)
    labels = identify_clusters(img_dataset, args.n_clusters)
    print(labels)
    img_dataset.save_clusters(labels, args.n_clusters, "dinov2")
