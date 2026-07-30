from pathlib import Path

import polars as pl
import torch
from accelerate import Accelerator

from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader

from iconoscope.storage import save_features
from iconoscope.dataset import ImageDataset


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
