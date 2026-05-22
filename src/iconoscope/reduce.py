from __future__ import annotations

from math import sqrt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_to_2d(
    features: np.ndarray,
    pca_dims: int | None = 50,
    backend: str = "tsne",
    perplexity: float | None = None,
    random_state: int = 42,
) -> np.ndarray:
    N = features.shape[0]
    if pca_dims is not None and pca_dims < features.shape[1]:
        pca = PCA(n_components=pca_dims, random_state=random_state)
        features_2d = pca.fit_transform(features)
    else:
        features_2d = features

    if perplexity is None:
        perplexity = min(50, max(5, int(sqrt(N))))

    if backend == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
    elif backend == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP backend requires `pip install umap-learn`")
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown reducer backend: {backend!r}")

    coords = reducer.fit_transform(features_2d)
    coords = coords - coords.min(axis=0)
    span = coords.max(axis=0)
    span[span == 0] = 1.0  # avoid division by zero if all points share an axis value
    coords = coords / span
    return coords.astype(np.float32)
