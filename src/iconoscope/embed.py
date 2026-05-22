from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class _ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), idx
        except Exception:
            return None, idx


def _make_loader(paths, transform, batch_size, num_workers, collate_fn, pin_memory):
    return DataLoader(
        _ImageDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def collate_fn(batch):
    import torch

    batch = [(img, idx) for img, idx in batch if img is not None]
    if not batch:
        return None, []
    imgs, idxs = zip(*batch)
    return torch.stack(imgs), list(idxs)


def _build_backbone(backend: str, device: str) -> object:
    import torch
    import torch.nn as nn

    if backend == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(*list(model.children())[:-2])
    elif backend == "clip":
        try:
            import clip
        except ImportError:
            raise SystemExit(
                "Error: CLIP backend requires: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        model, _ = clip.load("ViT-B/32", device=device)
        backbone = model.visual
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def _get_transform(backend: str) -> object:
    import torchvision.transforms as T

    if backend == "resnet50":
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif backend == "clip":
        return T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def extract_features(
    image_paths: list[Path],
    backend: str = "resnet50",
    batch_size: int = 64,
    device: str = "auto",
    num_workers: int = 4,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[Path]]:
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

    backbone = _build_backbone(backend, torch_device)
    transform = _get_transform(backend)

    valid_paths: list[Path] = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img.close()
            valid_paths.append(p)
        except Exception as e:
            warnings.warn(f"Skipping {p}: {e}")

    loader = _make_loader(
        valid_paths,
        transform,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device_str == "cuda"),
    )

    features_list: list[np.ndarray] = []
    iterator = tqdm(loader, desc="Extracting features") if show_progress else loader
    for batch in iterator:
        if batch is None:
            continue
        imgs, _ = batch
        imgs = imgs.to(torch_device)
        with torch.no_grad():
            feats = backbone(imgs)
            if backend == "resnet50":
                feats = feats.mean(dim=[2, 3])
        feats = feats / feats.norm(dim=1, keepdim=True)
        features_list.append(feats.cpu().numpy())

    features = np.concatenate(features_list, axis=0).astype(np.float32)
    return features, valid_paths
