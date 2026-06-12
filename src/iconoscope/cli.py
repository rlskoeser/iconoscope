from __future__ import annotations

import argparse
from pathlib import Path

from iconoscope.cache import describe
from iconoscope.cluster import run_cluster
from iconoscope.embed import EmbedModel, run_embed
from iconoscope.mosaic import run_mosaic
from iconoscope.viewer import run_viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNN embedding mosaic: embed images and render a 2D layout mosaic"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── embed ──────────────────────────────────────────────────────────────
    embed_parser = subparsers.add_parser(
        "embed", help="Extract CNN features from images"
    )
    embed_parser.add_argument("image_dir", type=str, help="Directory of input images")
    embed_parser.add_argument(
        "embeddings", type=str, help="Output embeddings file (e.g. mycollection.h5)"
    )
    embed_parser.add_argument(
        "--model",
        type=str,
        default=EmbedModel.DINOV3,
        choices=list(EmbedModel),
        help="Embedding model",
    )
    embed_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    embed_parser.add_argument(
        "--ext",
        type=str,
        default=None,
        metavar="EXT",
        help="Limit to images with single extension, e.g. .jpeg",
    )
    embed_parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)"
    )
    embed_parser.add_argument(
        "--center-crop",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="Crop to this fraction of the image centre before embedding, e.g. 0.7 to keep the inner 70%%",
    )

    # ── info ───────────────────────────────────────────────────────────────
    info_parser = subparsers.add_parser(
        "info", help="Report what is stored in an embeddings file"
    )
    info_parser.add_argument("embeddings", type=str, help="Embeddings file to inspect")

    # ── mosaic ─────────────────────────────────────────────────────────────
    mosaic_parser = subparsers.add_parser(
        "mosaic", help="Render a mosaic from embeddings"
    )
    mosaic_parser.add_argument(
        "embeddings", type=str, help="Embeddings file (e.g. mycollection.h5)"
    )
    mosaic_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <embeddings-stem>.jpg)",
    )
    mosaic_parser.add_argument(
        "--model",
        type=str,
        default=EmbedModel.DINOV3,
        choices=list(EmbedModel),
        help="Embedding model (must match the model used during embed)",
    )
    mosaic_parser.add_argument(
        "--reducer",
        type=str,
        default="tsne",
        choices=["tsne", "umap"],
        help="Dimensionality reduction backend",
    )
    mosaic_parser.add_argument(
        "--layout",
        type=str,
        default="full_grid",
        choices=["full_grid", "first_come"],
        help="Grid assignment method",
    )
    mosaic_parser.add_argument(
        "--width", type=int, default=2000, help="Output canvas width in pixels"
    )
    mosaic_parser.add_argument(
        "--height", type=int, default=2000, help="Output canvas height in pixels"
    )
    mosaic_parser.add_argument(
        "--thumb-size",
        type=lambda v: v if v == "auto" else int(v),
        default=50,
        metavar="N|auto",
        help="Thumbnail size in pixels, or 'auto' to fit all images",
    )
    mosaic_parser.add_argument(
        "--jpeg-quality", type=int, default=90, help="JPEG output quality (1-95)"
    )
    mosaic_parser.add_argument(
        "--save-coords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache reduced coordinates in the embeddings file (default: on)",
    )
    mosaic_parser.add_argument(
        "--load-coords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cached coordinates from the embeddings file if available (default: on)",
    )

    # ── cluster ────────────────────────────────────────────────────────────
    cluster_parser = subparsers.add_parser("cluster", help="K-means cluster embeddings")
    cluster_parser.add_argument("embeddings", type=str, help="Embeddings file")
    cluster_parser.add_argument(
        "-k", type=int, required=True, help="Number of clusters"
    )
    cluster_parser.add_argument(
        "--model",
        type=str,
        default=EmbedModel.DINOV3,
        choices=list(EmbedModel),
        help="Embedding model to cluster",
    )

    # ── viewer ─────────────────────────────────────────────────────────────
    viewer_parser = subparsers.add_parser(
        "viewer", help="Generate an interactive HTML scatter viewer"
    )
    viewer_parser.add_argument("embeddings", type=str, help="Embeddings file")
    viewer_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output HTML path (default: <embeddings-stem>.html)",
    )
    viewer_parser.add_argument(
        "--reducer",
        type=str,
        default="tsne",
        choices=["tsne", "umap"],
        help="Coordinates to plot",
    )
    viewer_parser.add_argument(
        "--clusters",
        type=str,
        default="auto",
        metavar="DATASET",
        help="Cluster dataset name to colour by, 'auto' to use first available, 'none' to disable",
    )
    viewer_parser.add_argument(
        "--thumb-size",
        type=int,
        default=64,
        metavar="N",
        help="Thumbnail size in pixels (default: 64)",
    )

    args = parser.parse_args()
    match args.command:
        case "info":
            path = Path(args.embeddings)
            if not path.exists():
                parser.error(f"{path} not found")
            info = describe(path)
            print(f"{path}")
            print(
                f"  {info['n_images']} images"
                if info["n_images"] is not None
                else "  (no paths)"
            )
            if info["features"]:
                print("\n  Features:")
                for f in info["features"]:
                    n, d = f["shape"]
                    print(f"    {f['model']:<12}  {n} × {d}")
            if info["coords"]:
                print("\n  Coordinates:")
                for c in info["coords"]:
                    n, d = c["shape"]
                    print(f"    {c['reducer']:<12}  {n} × {d}")
            if info["clusters"]:
                print("\n  Clusters:")
                for c in info["clusters"]:
                    print(f"    {c['dataset']:<24}  k={c['k']}")
        case "embed":
            run_embed(
                image_dir=Path(args.image_dir),
                output=Path(args.embeddings),
                model=args.model,
                batch_size=args.batch_size,
                ext=args.ext,
                device=args.device,
                center_crop=args.center_crop,
            )
        case "mosaic":
            embeddings = Path(args.embeddings)
            output = (
                Path(args.output) if args.output else embeddings.with_suffix(".jpg")
            )
            run_mosaic(
                embeddings=embeddings,
                output=output,
                model=args.model,
                reducer=args.reducer,
                layout=args.layout,
                width=args.width,
                height=args.height,
                thumb_size=args.thumb_size,
                load_coords=args.load_coords,
                save_coords=args.save_coords,
                jpeg_quality=args.jpeg_quality,
            )
        case "cluster":
            run_cluster(
                path=Path(args.embeddings),
                model=args.model,
                k=args.k,
            )
        case "viewer":
            embeddings = Path(args.embeddings)
            output = (
                Path(args.output) if args.output else embeddings.with_suffix(".html")
            )
            cluster_dataset = None if args.clusters == "none" else args.clusters
            run_viewer(
                path=embeddings,
                output=output,
                reducer=args.reducer,
                cluster_dataset=cluster_dataset,
                thumb_size=args.thumb_size,
            )


if __name__ == "__main__":
    main()
