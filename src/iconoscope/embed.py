from pathlib import Path
from typing import Generator

import polars as pl
import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from iconoscope.storage import save_features


class ImageDataset(IterableDataset):
    #: supported image extensions
    img_extensions = {".jpg", ".png", ".jpeg"}

    #: optional limit for number of images to find
    max_images: int | None = None

    def __init__(
        self,
        base_dir: Path,
        extensions: set | None = None,
        max_images: int | None = None,
    ):
        self.base_dir = base_dir
        if extensions:
            self.img_extensions = extensions

        self.max_images = max_images

    def __iter__(self) -> Generator[tuple[Image.Image, str]]:
        # by default, find all files with an extension and then filter by suffix
        rglob_pattern = "*.*"
        # if only a single extension, look for just that file type with rglob
        single_ext = len(self.img_extensions) == 1
        if single_ext:
            rglob_pattern = f"*{list(self.img_extensions)[0]}"

        for i, file_path in enumerate(self.base_dir.rglob(rglob_pattern), start=1):
            # return if found by single extension or if suffix is in the list
            if single_ext or file_path.suffix.lower() in self.img_extensions:
                # TODO: still needs error handling
                try:
                    img = Image.open(file_path).convert("RGB")
                except OSError as err:
                    # TODO: add logging
                    print(f"Error loading {file_path}: {err}")
                    continue
                # yield a tuple of image object and file path as string
                yield img, str(file_path)

            # stop after the requested maximum if there is one
            if self.max_images is not None and i >= self.max_images:
                break

    @staticmethod
    def collate(batch):
        # custom collate method for dataset collate_fn
        # image dataset batch is a list of tuples: [(img1, path1), (img2, path2), ...]
        # convert that to tuple of lists for this batch
        images, paths = zip(*batch)
        return list(images), list(paths)


def extract_img_features(img_dir: Path, outfile: Path, max_images: int | None = None):
    # autodetect which device to use
    device = Accelerator().device
    print(f"Using device={device}")

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    img_dataset = ImageDataset(img_dir, max_images=max_images)
    batch_size = 256
    dataloader = DataLoader(
        img_dataset,
        batch_size=batch_size,
        collate_fn=ImageDataset.collate,
    )

    img_feature_df = pl.DataFrame(
        schema={
            "image_path": pl.String,
            "features": pl.Array(pl.Float32, 768),  # does vector length vary by model?
        },
    )

    # batch size depends on model and available GPU/CPU memory
    # hf chat agent suggestion for ViT-Base with 224×224 images:
    # 8 GB -> 8–16; 16 GB -> 32–64; 24 GB -> 64–128; 40+ GB -> 128–256+

    progbar = tqdm(desc="Extracting features")
    for images, paths in dataloader:
        inputs = processor(images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            results = outputs.pooler_output.cpu()
            img_feature_df.extend(
                pl.DataFrame(
                    data={
                        "image_path": paths,
                        "features": results,
                    }
                )
            )
            # update progress bar (how many to increase, not the total count)
            progbar.update(len(images))

    save_features(outfile, img_feature_df, "dinov2")
    progbar.close()
    print(
        f"Successfully extracted features from {img_feature_df.height:,} images and saved to {outfile}"
    )
