from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import h5py
from PIL import Image
from torch.utils.data import IterableDataset

#: default image extensions
DEFAULT_IMG_EXTENSIONS = {".jpg", ".png", ".jpeg"}

# base class for iterable dataset without storage?


@dataclass
class ImageDataset(IterableDataset):
    """Iterable image dataset for use with torch DataLoader."""

    #: base directory for images in this dataset
    image_dir: Path

    #: image extensions; if not specified, uses the defaults
    extensions: set[str] = field(default_factory=DEFAULT_IMG_EXTENSIONS.copy)

    #: optional limit for number of images to find
    max_images: int | None = None

    def __post_init__(self):
        if not self.image_dir.is_dir():
            raise SystemExit(f"Image directory `{self.image_dir}` is not a directory")

    def __iter__(self) -> Iterator[tuple[Image.Image, str]]:
        # by default, find all files with an extension and then filter by suffix
        rglob_pattern = "*.*"
        # if only a single extension, look for just that file type with rglob
        single_ext = len(self.extensions) == 1
        if single_ext:
            rglob_pattern = f"*{list(self.extensions)[0]}"

        for i, file_path in enumerate(self.image_dir.rglob(rglob_pattern), start=1):
            # return if found by single extension or if suffix is in the list
            if single_ext or file_path.suffix.lower() in self.extensions:
                # TODO: still needs error handling
                try:
                    img = Image.open(file_path).convert("RGB")
                except OSError as err:
                    # TODO: switch to logging/warning
                    print(f"Error loading {file_path}: {err}")
                    continue
                # yield a tuple of image object and file path as string
                yield img, str(file_path)

            # stop after the requested maximum if there is one
            if self.max_images is not None and i >= self.max_images:
                break

    @staticmethod
    def collate(batch):
        # custom collate method for dataset collate_fn
        # image dataset batch is a list of tuples: [(img1, path1), (img2, path2), ...]
        # convert that to tuple of lists for this batch
        images, paths = zip(*batch)
        return list(images), list(paths)


@dataclass
class PersistentImageDataset(IterableDataset):
    """
    Persistent image dataset. Wraps image dataset with logic for saving and
    loading from HDF5.
    """

    #: path to hdf5 storage for image dataset
    storage_path: Path

    #: base directory for images in this dataset - OPTIONAL (save as an attr)
    image_dir: Path | None = None

    def save_features(self, df: pl.DataFrame, model_name: str) -> None:
        # handle save/update

        # check in post init?
        # if outfile.exists():
        # print(f"warning: {outfile} already exists")
        # check expected columns in dataframe?
        # TODO: handle updating existing file more carefully

        #  validation / checks:
        # - required/expected columns
        with h5py.File(self.storage_path, "w") as f:
            # create a group for image information
            img_grp = f.create_group("image")
            # save image paths as one dataset
            img_grp.create_dataset(
                "paths", data=df["image_path"].to_numpy(), compression="gzip"
            )
            # for each model, create a group to gather related/downstream information
            model_grp = img_grp.create_group(f"models/{model_name}")
            # save extracted features as a dataset
            model_grp.create_dataset(
                "features", data=df["features"].to_numpy(), compression="gzip"
            )
            # img_grp.attrs["last_modified"] = datetime.now().isoformat()
            # create a features dataset by model name
            # NOTE: support storing umap + clusters, and keep model feature derivatives together
            # seems easiest to store each column as a dataset

            # TODO: save umap for associated model/features
            # model_grp

            remainder_cols = set(df.columns) - {"image_path", "features"}
            if remainder_cols:
                print("Warning: unsaved columns (%s)" % ",".join(remainder_cols))
