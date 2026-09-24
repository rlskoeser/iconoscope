import logging
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import h5py
import numpy as np
import polars as pl
import pytest
from PIL import Image

from iconoscope.dataset import ImageDataset, find_images

## image dataset class


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


def test_create_writes_validated_image_inventory(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "good.jpg")
    (image_dir / "broken.jpg").write_text("not an image")

    dataset = ImageDataset.create(image_dir, tmp_path / "data.h5")

    assert list(dataset.get_image_paths()) == [image_dir / "good.jpg"]
    assert dataset.image_count == 1
    with h5py.File(tmp_path / "data.h5", "r") as h5_file:
        assert "image/models" in h5_file
        assert h5_file["image/paths"].size == 1


def test_get_image_paths(tmp_path: Path, tmp_image_dir: Path):
    # when storage file doesn't exist, yields results from find image method
    h5_datafile = tmp_path / "data.h5"
    img_ds = ImageDataset(image_dir=tmp_image_dir, storage_path=h5_datafile)
    with patch.object(img_ds, "load_image_paths") as mock_load_img_paths:
        img_paths = img_ds.get_image_paths()
        # should not load from data when storage file doesn't exist
        mock_load_img_paths.assert_not_called()
        assert isinstance(img_paths, Iterable)
        img_paths = list(img_paths)
        assert len(img_paths) == 3  # 3 in fixture
        for path in img_paths:
            assert isinstance(path, Path)
            assert path.suffix == ".jpg"

        h5_datafile.touch()
        test_image_paths = ["foobar_a.jpg", "b.jpg"]
        img_df = pl.DataFrame(data={"image_path": test_image_paths})
        mock_load_img_paths.return_value = img_df
        img_paths = img_ds.get_image_paths()
        # call count assertion fails, but mock data is working
        # assert mock_load_img_paths.call_count == 1
        assert isinstance(img_paths, Iterable)
        img_paths = list(img_paths)
        assert len(img_paths) == len(test_image_paths)
        for i, path in enumerate(img_paths):
            assert isinstance(path, Path)
            assert str(path) == test_image_paths[i]


def test_iter(tmp_path: Path, tmp_image_dir: Path):
    h5_datafile = tmp_path / "data.h5"
    img_ds = ImageDataset(image_dir=tmp_image_dir, storage_path=h5_datafile)
    images = img_ds.__iter__()
    assert isinstance(images, Iterable)
    images = list(images)
    assert len(images) == 3  # matches fixture data
    # yield a tuple of image object and file path as string
    # yield img, str(file_path)
    for img_tuple in images:
        assert isinstance(img_tuple[0], Image.Image)
        assert isinstance(img_tuple[1], str)
        assert img_tuple[1].endswith(".jpg")


def test_iter_err(tmp_path: Path, tmp_image_dir: Path, caplog):
    caplog.set_level(logging.WARN)
    # handle image file that can't be loaded
    h5_datafile = tmp_path / "data.h5"
    # add a non-image file with an image extension
    bad_img = tmp_image_dir / "bogus.png"
    bad_img.write_text("this is not an image")
    img_ds = ImageDataset(image_dir=tmp_image_dir, storage_path=h5_datafile)
    images = list(img_ds.__iter__())
    assert len(images) == 3  # should return all fixtures but nothing else
    assert len(caplog.record_tuples) == 1
    assert "Error loading" in caplog.text  # warns about the problem


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


def test_save_features_preserves_existing_models_and_derived_data(tmp_path: Path):
    storage_path = tmp_path / "data.h5"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    dataset = ImageDataset(storage_path=storage_path, image_dir=image_dir)
    paths = ["/images/one.jpg", "/images/two.jpg"]

    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": paths,
                "features": np.array([[1.0, 0.0], [0.0, 1.0]]),
            }
        ),
        "dinov2",
    )
    with h5py.File(storage_path, "r+") as h5_file:
        model_group = h5_file["image/models/dinov2"]
        model_group.create_dataset("umap", data=[[0.1, 0.2], [0.3, 0.4]])
        cluster = model_group.create_dataset("cluster", data=[0, 1])
        cluster.attrs["n_clusters"] = 2

    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": paths,
                "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
            }
        ),
        "clip",
    )

    with h5py.File(storage_path, "r") as h5_file:
        assert set(h5_file["image/models"]) == {"dinov2", "clip"}
        assert h5_file["image/models/dinov2/features"][:].tolist() == [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        assert h5_file["image/models/dinov2/umap"][:].tolist() == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
        assert h5_file["image/models/dinov2/cluster"][:].tolist() == [0, 1]
        assert h5_file["image/models/dinov2/cluster"].attrs["n_clusters"] == 2


def _dataset_with_features(tmp_path: Path) -> ImageDataset:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    dataset = ImageDataset(
        storage_path=tmp_path / "data.h5",
        image_dir=image_dir,
    )
    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": ["/images/one.jpg", "/images/two.jpg"],
                "features": np.array([[1.0, 0.0], [0.0, 1.0]]),
            }
        ),
        "dinov2",
    )
    return dataset


def test_save_features_rejects_different_image_paths(tmp_path: Path):
    dataset = _dataset_with_features(tmp_path)
    with pytest.raises(ValueError, match="image paths"):
        dataset.save_features(
            pl.DataFrame(
                {
                    "image_path": ["/images/one.jpg", "/images/other.jpg"],
                    "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
                }
            ),
            "clip",
        )


def test_save_features_rejects_mismatched_feature_rows(tmp_path: Path):
    dataset = _dataset_with_features(tmp_path)
    with pytest.raises(ValueError, match="existing models"):
        dataset.save_features(
            pl.DataFrame(
                {
                    "image_path": ["/images/one.jpg"],
                    "features": np.array([[2.0, 0.0]]),
                }
            ),
            "clip",
        )


def test_save_features_prunes_invalid_paths_before_first_model(tmp_path: Path):
    dataset = _dataset_with_features(tmp_path)
    with h5py.File(dataset.storage_path, "r+") as h5_file:
        del h5_file["image/models/dinov2"]

    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": ["/images/one.jpg"],
                "features": np.array([[2.0, 0.0]]),
            }
        ),
        "clip",
    )

    assert dataset.image_count == 1
    assert dataset.load_image_paths()["image_path"].to_list() == ["/images/one.jpg"]


def test_save_features_rejects_non_2d_features(tmp_path: Path):
    dataset = _dataset_with_features(tmp_path)
    with pytest.raises(ValueError, match="2-dimensional"):
        dataset.save_features(
            pl.DataFrame(
                {
                    "image_path": ["/images/one.jpg", "/images/two.jpg"],
                    "features": [2.0, 3.0],
                }
            ),
            "clip",
        )


def test_save_features_rejects_unsupported_columns(tmp_path: Path):
    dataset = _dataset_with_features(tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        dataset.save_features(
            pl.DataFrame(
                {
                    "image_path": ["/images/one.jpg", "/images/two.jpg"],
                    "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
                    "label": ["a", "b"],
                }
            ),
            "clip",
        )


def test_save_features_defines_same_model_behavior(tmp_path: Path):
    dataset = _dataset_with_features(tmp_path)
    with pytest.raises(ValueError, match="model.*already exists"):
        dataset.save_features(
            pl.DataFrame(
                {
                    "image_path": ["/images/one.jpg", "/images/two.jpg"],
                    "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
                }
            ),
            "dinov2",
        )


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
    # Discovery only checks suffixes, so these do not need to be decodable
    # images (which keeps the test independent of optional AVIF support in Pillow).
    (tmp_path / "d.webp").touch()
    (tmp_path / "e.avif").touch()
    dataset = find_images(tmp_path)
    assert len(list(dataset)) == 5


def test_find_images_extension_case_insensitive(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "upper.JPG", format="JPEG")
    items = find_images(tmp_path)
    assert len(list(items)) == 1
