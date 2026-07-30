import numpy as np
import umap
from sklearn.decomposition import PCA


def reduce_features(features: np.ndarray) -> np.ndarray:
    """Takes an array of embedding feature vectors and returns normalized coordinates.
    Uses PCA to reduce, UMAP to transform to two dimensions, then normalize from 0 to 1 for
    both axes. Returns an array of x,y coordinates for each feature vector in the input."""
    # use PCA to reduce vectors from 768 to 50 (but handle small datasets < 50)
    n_components = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=n_components).fit_transform(features)
    # use umap to project the reduced vectors into two dimensions
    coords = umap.UMAP(n_components=2).fit_transform(reduced)

    # determine smalleest and largest coordinates, and then
    # scale all coordinates to normalize from 0 to 1.0
    min_coords, max_coords = coords.min(0), coords.max(0)
    span = np.where(max_coords - min_coords > 0, max_coords - min_coords, 1.0)
    return (coords - min_coords) / span
