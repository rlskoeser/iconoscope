import numpy as np
from sklearn.cluster import KMeans

from iconoscope.dataset import ImageDataset


def identify_clusters(img_dataset: ImageDataset, n_clusters: int) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=100,
        n_init="auto",
        random_state=0,
    )
    df = img_dataset.load_data(features=True)
    # mini-batch kmeans allows fitting in batches
    # compute the centroids and
    cluster_labels = kmeans.fit_predict(df["features"])
    return cluster_labels
