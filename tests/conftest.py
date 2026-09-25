from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Temp directory with a few small RGB images."""
    for i in range(3):
        img = Image.new("RGB", (32, 32), color=(i * 80, i * 40, 100))
        img.save(tmp_path / f"img_{i}.jpg")
    return tmp_path
