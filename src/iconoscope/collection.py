from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageCollection:
    #: base dir for images in this collection
    image_dir: Path

    #: path to hdf5 storage for image collection
    storage_path: Path

    def __post_init__(self):
        if not self.image_dir.is_dir():
            raise SystemExit(f"Image directory `{self.image_dir}` is not a directory")

    # if images not already in the collection
    def find_images():
        pass

    def extract_features(self, max_images: int | None = None):
        pass


# extract_img_features(args.image_dir, args.output_path, args.max)

# does image dataset belong here?
# image dataset == image collection ? 🤯
