from __future__ import annotations

from pathlib import Path


def find_images(image_dir: Path, ext: str | None = None) -> list[Path]:
    if ext:
        ext = ext.lstrip(".")
        return sorted(image_dir.rglob(f"*.{ext}"))
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in extensions]
