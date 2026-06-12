from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


# 20-colour qualitative palette
_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]


def _load_template() -> str:
    """Return the built Svelte viewer HTML. Looks for bundled copy first, then dev build."""
    bundled = Path(__file__).parent / "viewer_template.html"
    if bundled.exists():
        return bundled.read_text(encoding="utf-8")
    dev = Path(__file__).parent.parent.parent / "viewer" / "dist" / "index.html"
    if dev.exists():
        return dev.read_text(encoding="utf-8")
    raise FileNotFoundError(
        "Viewer template not found. "
        "Run `npm install && npm run build` in the viewer/ directory first."
    )


def _autodetect_clusters(path: Path) -> str | None:
    """Return the name of the first clusters_* dataset found, or None."""
    import h5py
    with h5py.File(path, "r") as f:
        names = sorted(k for k in f.keys() if k.startswith("clusters_"))
    return names[0] if names else None


def run_viewer(
    path: Path,
    output: Path,
    reducer: str = "tsne",
    cluster_dataset: str | None = "auto",
    thumb_size: int = 64,
    show_progress: bool = True,
) -> None:
    """Generate a self-contained HTML scatter viewer from an embeddings file."""
    import h5py

    from iconoscope.cache import load_coords, load_paths

    image_paths = load_paths(path)
    coords = load_coords(path, reducer)
    if coords is None:
        raise SystemExit(
            f"No {reducer} coordinates in {path}. "
            "Run `iconoscope mosaic --save-coords` first."
        )

    if cluster_dataset == "auto":
        cluster_dataset = _autodetect_clusters(path)

    cluster_labels = None
    if cluster_dataset:
        with h5py.File(path, "r") as f:
            if cluster_dataset not in f:
                available = sorted(k for k in f.keys() if k.startswith("clusters_"))
                raise SystemExit(
                    f"No dataset '{cluster_dataset}' in {path}. "
                    f"Available: {available or '(none)'}"
                )
            cluster_labels = f[cluster_dataset][:].tolist()

    if show_progress:
        print(f"Generating {thumb_size}px thumbnails for {len(image_paths)} images…")

    points = []
    for i, (p, (x, y)) in enumerate(zip(image_paths, coords)):
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=75, optimize=True)
            thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            thumb = ""
        point = {"id": i, "x": float(x), "y": float(y), "path": str(p), "thumb": thumb}
        if cluster_labels is not None:
            point["cluster"] = cluster_labels[i]
        points.append(point)

    # Build cluster list and assign per-point colours
    cluster_list = []
    if cluster_labels is not None:
        unique = sorted(set(cluster_labels))
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(unique))]
        color_map = {uid: colors[j] for j, uid in enumerate(unique)}
        for pt in points:
            pt["color"] = color_map[pt["cluster"]]
        cluster_list = [
            {"id": uid, "color": colors[j], "label": f"Cluster {uid}"}
            for j, uid in enumerate(unique)
        ]

    data_json = json.dumps({"points": points, "clusters": cluster_list})
    template = _load_template()
    html = template.replace("window.__DATA__ = {};", f"window.__DATA__ = {data_json};")
    output.write_text(html, encoding="utf-8")

    if show_progress:
        size_mb = len(html.encode()) / 1e6
        print(f"Viewer saved to {output} ({size_mb:.1f} MB)")
