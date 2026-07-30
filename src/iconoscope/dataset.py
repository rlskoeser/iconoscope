from pathlib import Path

from PIL import Image
from torch.utils.data import IterableDataset


class ImageDataset(IterableDataset):
    #: supported image extensions
    img_extensions = {".jpg", ".png", ".jpeg"}

    #: optional limit for number of images to find
    max_images: int | None = None

    def __init__(
        self,
        base_dir: Path,
        extensions: set | None = None,
        max_images: int | None = None,
    ):
        self.base_dir = base_dir
        if extensions:
            self.img_extensions = extensions

        self.max_images = max_images

    def __iter__(self):  #  -> Generator[tuple[Image.Image, str]]:
        # by default, find all files with an extension and then filter by suffix
        rglob_pattern = "*.*"
        # if only a single extension, look for just that file type with rglob
        single_ext = len(self.img_extensions) == 1
        if single_ext:
            rglob_pattern = f"*{list(self.img_extensions)[0]}"

        for i, file_path in enumerate(self.base_dir.rglob(rglob_pattern), start=1):
            # return if found by single extension or if suffix is in the list
            if single_ext or file_path.suffix.lower() in self.img_extensions:
                # TODO: still needs error handling
                try:
                    img = Image.open(file_path).convert("RGB")
                except OSError as err:
                    # TODO: add logging
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
