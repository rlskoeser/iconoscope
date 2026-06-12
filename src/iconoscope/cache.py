from __future__ import annotations

from pathlib import Path

import numpy as np


def save_embeddings(path: Path, model: str, features: np.ndarray, image_paths: list[Path]) -> None:
    """Write features and paths to an HDF5 file; feature dataset is scoped by model name."""
    import h5py

    with h5py.File(path, "a") as f:
        name = f"features_{model}"
        if name in f:
            del f[name]
        f.create_dataset(name, data=features, compression="gzip")
        if "paths" in f:
            del f["paths"]
        f.create_dataset(
            "paths",
            data=np.array([str(p) for p in image_paths], dtype=object),
            dtype=h5py.string_dtype(),
        )


def load_embeddings(path: Path, model: str) -> tuple[np.ndarray, list[Path]]:
    """Load model-scoped features and shared paths from an HDF5 file."""
    import h5py

    with h5py.File(path, "r") as f:
        name = f"features_{model}"
        if name not in f:
            available = [k for k in f.keys() if k.startswith("features_")]
            raise KeyError(
                f"No features for model '{model}' in {path}. "
                f"Available: {available or '(none)'}"
            )
        features = f[name][:]
        paths_raw = f["paths"][:]

    image_paths = [Path(p.decode() if isinstance(p, bytes) else p) for p in paths_raw]
    return features, image_paths


def save_coords(path: Path, reducer: str, coords: np.ndarray) -> None:
    """Write reducer-scoped coords to an HDF5 file."""
    import h5py

    with h5py.File(path, "a") as f:
        name = f"coords_{reducer}"
        if name in f:
            del f[name]
        f.create_dataset(name, data=coords)


def load_coords(path: Path, reducer: str) -> np.ndarray | None:
    """Return cached coords for the given reducer, or None if not present."""
    import h5py

    with h5py.File(path, "r") as f:
        name = f"coords_{reducer}"
        if name not in f:
            return None
        return f[name][:]


def load_paths(path: Path) -> list[Path]:
    """Load image paths from an HDF5 file."""
    import h5py

    with h5py.File(path, "r") as f:
        paths_raw = f["paths"][:]
    return [Path(p.decode() if isinstance(p, bytes) else p) for p in paths_raw]


def describe(path: Path) -> dict:
    """Return a summary dict of what is stored in an HDF5 embeddings file."""
    import h5py

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        n_images = int(f["paths"].shape[0]) if "paths" in f else None
        features = [
            {"model": k[len("features_"):], "shape": tuple(f[k].shape)}
            for k in keys if k.startswith("features_")
        ]
        coords = [
            {"reducer": k[len("coords_"):], "shape": tuple(f[k].shape)}
            for k in keys if k.startswith("coords_")
        ]
        clusters = []
        for k in keys:
            if k.startswith("clusters_"):
                rest = k[len("clusters_"):]
                model_part, _, k_part = rest.rpartition("_")
                clusters.append({
                    "dataset": k,
                    "model": model_part or rest,
                    "k": int(k_part) if k_part.isdigit() else "?",
                    "n": int(f[k].shape[0]),
                })
    return {"n_images": n_images, "features": features, "coords": coords, "clusters": clusters}
