from pathlib import Path

import polars as pl
import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


class ImageDataset(IterableDataset):
    img_extensions = {".jpg", ".png", ".jpeg"}

    def __init__(self, base_dir: Path, extensions: set | None = None):
        self.base_dir = base_dir
        if extensions:
            self.img_extensions = extensions

    def __iter__(self):
        # by default, find all files with an extension and then filter by suffix
        rglob_pattern = "*.*"
        # if only a single extension, look for just that file type with rglob
        single_ext = len(self.img_extensions) == 1
        if single_ext:
            rglob_pattern = f"*{list(self.img_extensions)[0]}"

        for file_path in self.base_dir.rglob(rglob_pattern):
            # return if found by single extension or if suffix is in the list
            if single_ext or file_path.suffix.lower() in self.img_extensions:
                # TODO: still needs error handling
                img = Image.open(file_path).convert("RGB")
                # yield a tuple of image object and file path as string
                yield img, str(file_path)

    @staticmethod
    def collate(batch):
        # custom collate method for dataset collate_fn
        # image dataset batch is a list of tuples: [(img1, path1), (img2, path2), ...]
        # convert that to tuple of lists for this batch
        images, paths = zip(*batch)
        return list(images), list(paths)


def extract_img_features(img_dir: Path, outfile: Path):
    # autodetect which device to use
    device = Accelerator().device
    print(f"Using device={device}")

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    img_dataset = ImageDataset(img_dir)
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

    img_feature_df.write_parquet(outfile)
    progbar.close()
    print(f"Successfully extracted features from {img_feature_df.height:,} images.")
