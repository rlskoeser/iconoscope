from pathlib import Path

from PIL import Image

from iconoscope.embed import ImageDataset


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


def test_iter_finds_images(tmp_image_dir: Path):
    dataset = ImageDataset(tmp_image_dir)
    items = list(dataset)
    assert len(items) == 3
    for img, path in items:
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert Path(path).suffix == ".jpg"


def test_iter_empty_dir(tmp_path: Path):
    dataset = ImageDataset(tmp_path)
    assert list(dataset) == []


def test_iter_skips_non_image(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("hello")
    Image.new("RGB", (8, 8)).save(tmp_path / "photo.jpg")
    dataset = ImageDataset(tmp_path)
    items = list(dataset)
    assert len(items) == 1


def test_iter_custom_extensions(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    dataset = ImageDataset(tmp_path, extensions={".png"})
    items = list(dataset)
    assert len(items) == 1
    assert items[0][1].endswith(".png")


def test_iter_recurses_subdirs(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "top.jpg")
    Image.new("RGB", (8, 8)).save(sub / "nested.jpg")
    dataset = ImageDataset(tmp_path)
    assert len(list(dataset)) == 2
