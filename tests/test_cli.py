import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from iconoscope import cli


@patch("iconoscope.cli.ImageDataset")
@patch("iconoscope.cli.extract_img_features")
def test_embed_args(mock_extract_features, mock_img_dataset, tmp_path: Path):
    # test cli args are passed correctly for embed function
    out = tmp_path / "out.h5"
    with patch("sys.argv", ["iconoscope", "embed", str(tmp_path), str(out)]):
        cli.main()

    mock_img_dataset.assert_called_with(
        storage_path=out, image_dir=tmp_path, max_images=None
    )
    mock_extract_features.assert_called_with(mock_img_dataset.return_value)
    mock_img_dataset.return_value.save_features.assert_called_with(
        mock_extract_features.return_value, "dinov2"
    )


def test_main_embed_missing_dir(tmp_path: Path):
    missing_dir = tmp_path / "no_such_dir"
    args = argparse.Namespace(
        image_dir=missing_dir, output_path=tmp_path / "out.h5", max=None
    )
    with pytest.raises(SystemExit):
        cli.main_embed(args)


## custom size type for argparse to support specifying size as wxh


def test_size_tuple():
    # single dimension -> square
    assert cli.size_tuple("200") == argparse.Namespace(width=200, height=200)
    assert cli.size_tuple(200) == argparse.Namespace(width=200, height=200)
    # width x height
    assert cli.size_tuple("150x250") == argparse.Namespace(width=150, height=250)

    # errors
    # too many dimensions
    with pytest.raises(ValueError, match="Could not parse"):
        cli.size_tuple("100x200x300")
    # too few
    with pytest.raises(ValueError, match="invalid literal"):
        cli.size_tuple("")
    # non-numeric
    with pytest.raises(ValueError, match="invalid literal"):
        cli.size_tuple("a")
    with pytest.raises(ValueError, match="invalid literal"):
        cli.size_tuple("10xb")
