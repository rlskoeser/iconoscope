from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import polars as pl
from PIL import Image
from torch.utils.data import IterableDataset

#: default image extensions
DEFAULT_IMG_EXTENSIONS = {".jpg", ".png", ".jpeg", ".tiff"}

# base class for iterable dataset without storage?


def find_images(
    base_dir: Path,
    extensions: set[str] | None = None,
    max: int | None = None,
) -> Iterator[Path]:
    """
    Find images within the specified directory. Can optionally limit by file extension
    or stop when a specified maximum number of images is found.
    """

    # if no extensions are specified, use the defaults
    if extensions is None:
        extensions = DEFAULT_IMG_EXTENSIONS.copy()

    # because we want case-insensitive matching and will usually look for multiple
    # image extensions, by default look for any file extension and filter in the loop
    rglob_pattern = "*.*"
    single_ext = False
    # if  there is only one extension, use that as the pattern (loses the case-insensitivity)
    if len(extensions) == 1:
        rglob_pattern = f"*{list(extensions)[0]}"
        # ext-specific glob pattern means we can bypass the extra check
        single_ext = True

    found = 0
    for file_path in base_dir.rglob(rglob_pattern):
        # return if found by single extension or if suffix is in the list
        if single_ext or file_path.suffix.lower() in extensions:
            yield file_path
            found += 1

        # stop after the requested maximum if there is one
        if max is not None and found >= max:
            break


@dataclass
class ImageDataset(IterableDataset):
    """
    Persistent image dataset. Provides iterable image dataset logic for use with torch DataLoader,
    but also provides logic for saving and loading from HDF5 file.
    """

    #: path to hdf5 storage for image dataset
    storage_path: Path

    #: base directory for images in this dataset - OPTIONAL (save as an attr)
    image_dir: Path | None = None

    #: image extensions; if not specified, uses the defaults
    extensions: set[str] = field(default_factory=DEFAULT_IMG_EXTENSIONS.copy)

    #: optional limit for number of images to find
    max_images: int | None = None

    def __post_init__(self):
        # when creating a new collection, storage will not exist so image dir is required
        if not self.storage_path.exists():
            if self.image_dir is None:
                raise ValueError(
                    f"image_dir is required when creating new image dataset ({self.storage_path} does not exist)"
                )
            elif not self.image_dir.is_dir():
                raise ValueError(
                    f"Image directory `{self.image_dir}` is not a directory"
                )
        # otherwise, storage exists and we will load paths from it

    def get_image_paths(self) -> Iterator[Path]:
        if self.storage_path.exists():
            print("loading images from storage")
            for row in self.load_image_paths().iter_rows(named=True):
                yield Path(row["image_path"])
        elif self.image_dir:
            print("finding images on disk")
            yield from find_images(self.image_dir, max=self.max_images)

    ## iterable dataset logic

    def __iter__(self) -> Iterator[tuple[Image.Image, str]]:
        for file_path in self.get_image_paths():
            # TODO: still needs better error handling
            try:
                img = Image.open(file_path).convert("RGB")
            except OSError as err:
                # TODO: switch to logging/warning
                print(f"Error loading {file_path}: {err}")
                continue
            # yield a tuple of image object and file path as string
            yield img, str(file_path)

    @staticmethod
    def collate(batch):
        # custom collate method for dataset collate_fn
        # image dataset batch is a list of tuples: [(img1, path1), (img2, path2), ...]
        # convert that to tuple of lists for this batch
        images, paths = zip(*batch)
        return list(images), list(paths)

    ## storage functionality

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

            # save image directory when first creating dataset
            # TODO: also save max if specified and extensions if not default
            if self.image_dir is not None:
                # convert path to string
                img_grp.attrs["image_dir"] = str(self.image_dir)

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
            remainder_cols = set(df.columns) - {"image_path", "features"}
            if remainder_cols:
                print("Warning: unsaved columns (%s)" % ",".join(remainder_cols))

    def info(self) -> dict:
        with h5py.File(self.storage_path, "r") as f:
            img_grp = f["image"]
            orig_img_dir = img_grp.attrs.get("image_dir")

            img_dataset = img_grp["paths"]
            info = {
                "image_paths": img_dataset.size,
                "image_dir": orig_img_dir,
                "models": {},
            }
            model_grp = img_grp["models"]
            for model in model_grp.keys():
                features = model_grp[f"{model}/features"]
                # check that they match?
                info["models"][model] = {"embeddings": features.shape}
            return info

    def load_image_paths(self) -> pl.DataFrame:
        with h5py.File(self.storage_path, "r") as f:
            img_grp = f["image"]
            img_dataset = img_grp["paths"]

            # [:] = retrieve all scalar data
            return pl.DataFrame(
                data={
                    # load image paths as string instead of binary string
                    "image_path": img_dataset[:].astype("T")[:],
                }
            )
