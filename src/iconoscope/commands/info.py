import argparse

from iconoscope.dataset import ImageDataset


def _dimensions(shape: tuple[int, int]) -> str:
    return "x".join(str(dim) for dim in shape)


def main(args: argparse.Namespace) -> None:
    img_dataset = ImageDataset(storage_path=args.dataset)
    details = img_dataset.info()
    info_details = [f"Details for {args.dataset} :"]

    if "image_paths" in details:
        whence = f" from {details['image_dir']}" if details.get("image_dir") else ""
        info_details.append(f"  {details['image_paths']:,} image paths{whence}")
    else:
        info_details.append(" image_path and features columns not found")

    for model, model_details in details["models"].items():
        embed_dims = _dimensions(model_details["embeddings"])
        info_details.append(f"  {model} extracted features ({embed_dims})")
        if "umap" in model_details:
            info_details.append(
                f"\tumap coordinates ({_dimensions(model_details['umap'])})"
            )
        if "cluster" in model_details:
            cluster = model_details["cluster"]
            info_details.append(
                f"\t{cluster['size']:,} cluster labels (k={cluster['k']})"
            )

    print("\n".join(info_details))
