from __future__ import annotations

from pathlib import Path

import numpy as np


def run_cluster(
    path: Path,
    model: str,
    k: int,
    show_progress: bool = True,
) -> None:
    """K-means cluster the embeddings; store assignments as clusters_{model}_{k} in the HDF5 file."""
    import h5py
    from sklearn.cluster import KMeans

    from iconoscope.cache import load_embeddings

    features, _ = load_embeddings(path, model)
    N = features.shape[0]

    if k >= N:
        raise SystemExit(f"error: k={k} must be less than the number of images ({N})")

    if show_progress:
        print(f"Clustering {N} embeddings ({model}) into {k} clusters…")

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(features).astype(np.int32)

    dataset = f"clusters_{model}_{k}"
    with h5py.File(path, "a") as f:
        if dataset in f:
            del f[dataset]
        f.create_dataset(dataset, data=labels)

    if show_progress:
        counts = np.bincount(labels)
        for i, c in enumerate(counts):
            print(f"  cluster {i:>2}: {c} images")
        print(f"Saved {dataset} to {path}")
