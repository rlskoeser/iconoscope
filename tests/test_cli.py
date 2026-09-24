import argparse
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from iconoscope import cli

logger = logging.getLogger(__name__)


def test_embed_args(tmp_path: Path):
    # Test parser arguments without importing the embedding stack.
    out = tmp_path / "out.h5"
    handler = MagicMock()
    command_module = ModuleType("iconoscope.commands.embed")
    command_module.main = handler
    with (
        patch("iconoscope.cli.import_module", return_value=command_module) as importer,
        patch("sys.argv", ["iconoscope", "embed", str(tmp_path), str(out)]),
    ):
        cli.main()

    importer.assert_called_once_with("iconoscope.commands.embed")
    args = handler.call_args.args[0]
    assert args.image_dir == tmp_path
    assert args.output_path == out
    assert args.max is None


def test_embed_missing_dir_does_not_import_handler(tmp_path: Path):
    with (
        patch("iconoscope.cli.import_module") as importer,
        patch("sys.argv", ["iconoscope", "embed", str(tmp_path / "missing"), "out.h5"]),
        pytest.raises(SystemExit),
    ):
        cli.main()

    importer.assert_not_called()


def test_cli_lazy_load():
    # for speed, calling the cli should not import heavy dependencies
    code = """
import sys
import time
started = time.perf_counter()
import iconoscope.cli

heavy = {"torch", "transformers", "umap", "sklearn"}
loaded = heavy.intersection(sys.modules)
assert not loaded, f"heavy dependencies loaded: {sorted(loaded)}"
print(f"{time.perf_counter() - started:.4f}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    logger.info("iconoscope.cli import: %ss", result.stdout.strip())


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
