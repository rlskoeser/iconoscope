import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from iconoscope import cli


def test_embed_parses_positional_args(tmp_path: Path):
    out = tmp_path / "out.parquet"
    with (
        patch.object(cli, "extract_img_features"),
        patch("sys.argv", ["iconoscope", "embed", str(tmp_path), str(out)]),
    ):
        cli.main()


def test_main_embed_raises_on_missing_dir(tmp_path: Path):
    missing_dir = tmp_path / "no_such_dir"
    args = argparse.Namespace(
        image_dir=missing_dir, output_path=tmp_path / "out.parquet"
    )
    with pytest.raises(SystemExit):
        cli.main_embed(args)


def test_main_embed_calls_extract(tmp_path: Path):
    output_path = tmp_path / "out.parquet"
    args = argparse.Namespace(image_dir=tmp_path, output_path=output_path, max=None)
    with patch.object(cli, "extract_img_features") as mock_extract:
        cli.main_embed(args)
        mock_extract.assert_called_once_with(tmp_path, output_path, args.max)


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
