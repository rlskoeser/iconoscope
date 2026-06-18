import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from iconoscope import cli


def test_embed_parses_positional_args(tmp_path: Path):
    out = tmp_path / "out.parquet"
    with (
        patch("iconoscope.cli.extract_img_features"),
        patch("sys.argv", ["iconoscope", "embed", str(tmp_path), str(out)]),
    ):
        cli.main()


def test_main_embed_raises_on_missing_dir(tmp_path: Path):
    missing = tmp_path / "no_such_dir"
    args = argparse.Namespace(image_dir=missing, output_path=tmp_path / "out.parquet")
    with pytest.raises(SystemExit):
        cli.main_embed(args)


def test_main_embed_calls_extract(tmp_path: Path):
    out = tmp_path / "out.parquet"
    args = argparse.Namespace(image_dir=tmp_path, output_path=out)
    with patch("iconoscope.cli.extract_img_features") as mock_extract:
        cli.main_embed(args)
        mock_extract.assert_called_once_with(tmp_path, out)
