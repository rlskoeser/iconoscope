import argparse
from pathlib import Path

from iconoscope.cluster import identify_clusters
from iconoscope.dataset import ImageDataset
from iconoscope.embed import extract_img_features
from iconoscope.mosaic import generate_mosaic


def main_embed(args: argparse.Namespace):
    try:
        img_dataset = ImageDataset(
            storage_path=args.output_path,
            image_dir=args.image_dir,
            max_images=args.max,
        )
        # dataset class validates storage / image dir on load
        # extract features and pass back to dataset to persist
        img_dataset.save_features(extract_img_features(img_dataset), "dinov2")
    except ValueError as err:
        raise SystemExit(err)


def main_mosaic(args: argparse.Namespace):
    img_dataset = ImageDataset(storage_path=args.dataset)
    generate_mosaic(
        img_dataset,
        output=args.output,
        width=args.size.width,
        height=args.size.height,
        jpeg_quality=args.jpeg_quality,
        sample_size=args.limit,
    )


def dimensions(shape: tuple[int, int]) -> str:
    # convert ndarray shape tuple into readable dimensions
    return "x".join([str(dim) for dim in shape])


def dataset_info(args: argparse.Namespace):
    img_dataset = ImageDataset(storage_path=args.dataset)
    # if not args.embeddings.is_file():
    # raise SystemExit(f"{args.embeddings} is not a file")
    details = img_dataset.info()  # args.embeddings)
    info_details = [f"Details for {args.dataset} :"]

    if "image_paths" in details:
        whence = ""
        if details.get(
            "image_dir"
        ):  # if image dir is set in attributes, include in info
            whence = f" from {details['image_dir']}"
        info_details.append(f"  {details['image_paths']:,} image paths{whence}")
    else:
        info_details.append(" image_path and features columns not found")

    for model, model_details in details["models"].items():
        # convert shape tuple into a readable dimension string
        embed_dims = dimensions(model_details["embeddings"])
        info_details.append(f"  {model} extracted features ({embed_dims})")
        if "umap" in model_details:
            umap_dims = dimensions(model_details["umap"])
            info_details.append(f"\tumap coordinates ({umap_dims})")
        if "cluster" in model_details:
            label_size = model_details["cluster"]["size"]
            cluster_k = model_details["cluster"]["k"]
            info_details.append(f"\t{label_size:,} cluster labels (k={cluster_k})")

    print("\n".join(info_details))


def dataset_cluster(args: argparse.Namespace):
    img_dataset = ImageDataset(storage_path=args.dataset)
    labels = identify_clusters(img_dataset, args.n_clusters)
    print(labels)
    img_dataset.save_clusters(labels, args.n_clusters, "dinov2")


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
    parser_embed.set_defaults(func=main_embed)

    ## dataset info
    info_parser = subparsers.add_parser("info")
    info_parser.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file created by iconoscope embed (.hdf5)",
    )
    info_parser.set_defaults(func=dataset_info)

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
    parser_mosaic.set_defaults(func=main_mosaic)

    ## cluster image features
    cluster_parser = subparsers.add_parser("cluster")
    cluster_parser.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file created by iconoscope embed (.hdf5)",
    )
    cluster_parser.add_argument("n_clusters", type=int, help="Number of clusters")
    cluster_parser.set_defaults(func=dataset_cluster)

    # parse arguments and call the appropriate method
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
