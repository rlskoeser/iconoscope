from __future__ import annotations

import sys
import warnings
from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EmbedModel(StrEnum):
    DINOV2 = "dinov2"
    DINOV3 = "dinov3"
    CLIP = "clip"


_HF_MODEL_IDS: dict[EmbedModel, str] = {
    EmbedModel.DINOV2: "facebook/dinov2-base",
    EmbedModel.DINOV3: "facebook/dinov3-vitb16-pretrain-lvd1689m",
}


class _ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform, center_crop: float = 0.0):
        self.paths = paths
        self.transform = transform
        self.center_crop = center_crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.center_crop > 0.0:
                w, h = img.size
                cw, ch = int(w * self.center_crop), int(h * self.center_crop)
                left, top = (w - cw) // 2, (h - ch) // 2
                img = img.crop((left, top, left + cw, top + ch))
            return self.transform(img), idx
        except Exception:
            return None, idx



def _collate_fn(batch):
    import torch

    batch = [(img, idx) for img, idx in batch if img is not None]
    if not batch:
        return None, []
    imgs, idxs = zip(*batch)
    return torch.stack(imgs), list(idxs)


def _load_backend(backend: str, device) -> tuple[object, object]:
    """Return (backbone, transform) for the given backend."""
    match backend:
        case EmbedModel.DINOV2 | EmbedModel.DINOV3:
            try:
                from transformers import AutoImageProcessor, AutoModel
            except ImportError:
                raise SystemExit("Error: DINO models require: pip install 'transformers>=4.56.0'")
            model_id = _HF_MODEL_IDS[backend]
            processor = AutoImageProcessor.from_pretrained(model_id)
            backbone = AutoModel.from_pretrained(model_id)
            transform = lambda img: processor(images=img, return_tensors="pt")["pixel_values"][0]
        case EmbedModel.CLIP:
            try:
                import clip
            except ImportError:
                raise SystemExit(
                    "Error: CLIP backend requires: "
                    "pip install git+https://github.com/openai/CLIP.git"
                )
            backbone, transform = clip.load("ViT-B/32", device=device)
            backbone = backbone.visual
        case _:
            raise ValueError(f"Unknown backend: {backend!r}")
    backbone = backbone.to(device).float().eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, transform


def run_embed(
    image_dir: Path,
    output: Path,
    model: EmbedModel = EmbedModel.DINOV2,
    batch_size: int = 64,
    ext: str | None = None,
    device: str = "auto",
    center_crop: float = 0.0,
    show_progress: bool = True,
) -> None:
    from iconoscope.utils import find_images

    if not image_dir.is_dir():
        sys.exit(f"error: {image_dir} is not a directory")

    image_paths = find_images(image_dir, ext)
    if not image_paths:
        sys.exit(f"error: no images found in {image_dir}")

    if show_progress:
        print(f"Found {len(image_paths)} images")

    from iconoscope.cache import save_embeddings

    features, paths = extract_features(
        image_paths,
        model=model,
        batch_size=batch_size,
        device=device,
        center_crop=center_crop,
        show_progress=show_progress,
    )
    save_embeddings(output, model, features, paths)
    if show_progress:
        print(f"Saved embeddings ({features.shape[0]} images, {features.shape[1]} dims) to {output}")


def extract_features(
    image_paths: list[Path],
    model: EmbedModel = EmbedModel.DINOV2,
    batch_size: int = 64,
    device: str = "auto",
    num_workers: int = 4,
    center_crop: float = 0.0,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[Path]]:
    """Extract embedding vectors from image_paths; return (features float32 [N,D], valid_paths)."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = device
    torch_device = torch.device(device_str)

    backbone, transform = _load_backend(model, torch_device)

    valid_paths: list[Path] = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                img.verify()
            valid_paths.append(p)
        except Exception as e:
            warnings.warn(f"Skipping {p}: {e}")

    if device_str == "mps":
        num_workers = 0  # MPS + multiprocessing fork causes hangs

    loader = DataLoader(
        _ImageDataset(valid_paths, transform, center_crop=center_crop),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=(device_str == "cuda"),
    )

    features_list: list[np.ndarray] = []
    iterator = tqdm(loader, desc="Extracting features") if show_progress else loader
    for batch in iterator:
        imgs, _ = batch
        if imgs is None:
            continue
        imgs = imgs.to(torch_device)
        with torch.no_grad():
            match model:
                case EmbedModel.DINOV2 | EmbedModel.DINOV3:
                    feats = backbone(pixel_values=imgs).pooler_output
                case EmbedModel.CLIP:
                    feats = backbone(imgs)
        feats = feats / feats.norm(dim=1, keepdim=True)  # L2 normalise to unit sphere
        features_list.append(feats.cpu().numpy())

    features = np.concatenate(features_list, axis=0).astype(np.float32)
    return features, valid_paths
