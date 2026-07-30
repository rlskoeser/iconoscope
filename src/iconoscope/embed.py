import polars as pl
import torch
from accelerate import Accelerator

from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader

from iconoscope.dataset import ImageDataset


def extract_img_features(img_dataset: ImageDataset) -> pl.DataFrame:
    """ "
    Takes an image dataset and extracts features. Returns  a DataFrame with image_path, features.
    """
    # params to add later: model, boolean for progress bar

    # autodetect which device to use
    device = Accelerator().device
    print(f"Using device={device}")

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

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

    progbar.close()
    print(f"Successfully extracted features from {img_feature_df.height:,} images")
    return img_feature_df
