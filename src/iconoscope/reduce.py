from __future__ import annotations

from math import sqrt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_to_2d(
    features: np.ndarray,
    pca_dims: int | None = 50,
    reducer: str = "tsne",
    perplexity: float | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Optional PCA pre-reduction, then t-SNE/UMAP; returns float32 coords normalised to [0,1]²."""
    N = features.shape[0]
    if pca_dims is not None and pca_dims < features.shape[1]:
        pca = PCA(n_components=pca_dims, random_state=random_state)
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features

    if perplexity is None:
        perplexity = min(50, max(5, int(sqrt(N))))  # sqrt(N) heuristic; clamped to [5, 50]

    if reducer == "tsne":
        perplexity = min(perplexity, N - 1)
        if perplexity < 1:
            raise ValueError(f"Too few images ({N}) for t-SNE; need at least 2.")
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
    elif reducer == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP reducer requires `pip install umap-learn`")
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown reducer reducer: {reducer!r}")

    coords = reducer.fit_transform(features_reduced)
    coords = coords - coords.min(axis=0)
    span = coords.max(axis=0)
    span[span == 0] = 1.0  # avoid division by zero if all points share an axis value
    coords = coords / span
    return coords.astype(np.float32)
