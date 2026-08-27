from pathlib import Path

import pytest
from PIL import Image

from iconoscope.dataset import ImageDataset, find_images


def test_init_validation(tmp_path):
    image_dir = tmp_path / "images"
    # non-existent
    with pytest.raises(ValueError, match="not a directory"):
        ImageDataset(image_dir=image_dir, storage_path=tmp_path / "a.h5")
    # file instead of dir
    img_file = tmp_path / "img.png"
    img_file.touch()
    with pytest.raises(ValueError, match="not a directory"):
        ImageDataset(image_dir=img_file, storage_path=tmp_path / "a.h5")

    # image dir is required when h5 file doesn't exist
    with pytest.raises(ValueError, match="image_dir is required"):
        ImageDataset(storage_path=tmp_path / "a.h5")

    # existing directory - no error, returns new object
    image_dir.mkdir()
    assert ImageDataset(image_dir=image_dir, storage_path=tmp_path / "a.h5")

    # storage path exists - image dir is not required
    h5_datafile = tmp_path / "z.h5"
    h5_datafile.touch()  # currently doesn't validate (probably should in future)
    assert ImageDataset(storage_path=h5_datafile)


def test_collate_returns_lists():
    imgs = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    paths = ["/a/1.jpg", "/a/2.jpg"]
    # batch is a list of tuples of image and image path
    batch = [(imgs[i], paths[i]) for i in range(len(imgs))]
    # collate returns list of images and list of image paths
    result_imgs, result_paths = ImageDataset.collate(batch)
    assert result_imgs == imgs
    assert result_paths == paths


def test_collate_single_item():
    img = Image.new("RGB", (4, 4))
    imgs, paths = ImageDataset.collate([(img, "/x.png")])
    assert len(imgs) == 1
    assert paths == ["/x.png"]


## test find images utility method


def test_find_images(tmp_image_dir: Path):
    items = list(find_images(tmp_image_dir))
    assert len(items) == 3
    for path in items:
        assert isinstance(path, Path)
        assert path.suffix == ".jpg"


def test_find_images_empty_dir(tmp_path: Path):
    assert list(find_images(tmp_path)) == []


def test_find_images_skips_non_image(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("hello")
    Image.new("RGB", (8, 8)).save(tmp_path / "photo.jpg")
    items = list(find_images(tmp_path))
    assert len(items) == 1


def test_find_images_custom_extensions(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    items = list(find_images(tmp_path, extensions={".png"}))
    assert len(items) == 1
    assert items[0].suffix == ".png"


def test_find_images_recurses_subdirs(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "top.jpg")
    Image.new("RGB", (8, 8)).save(sub / "nested.jpg")
    items = find_images(tmp_path)
    assert len(list(items)) == 2


def test_find_images_max(tmp_image_dir: Path):
    items = list(find_images(tmp_image_dir, max=2))
    assert len(items) == 2


def test_find_images_default_extensions(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "c.jpeg")
    dataset = find_images(tmp_path)
    assert len(list(dataset)) == 3


def test_find_images_extension_case_insensitive(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "upper.JPG", format="JPEG")
    items = find_images(tmp_path)
    assert len(list(items)) == 1
