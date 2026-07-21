from datetime import datetime
from pathlib import Path

import h5py
import polars as pl


def save_features(outfile: Path, df: pl.DataFrame, model_name: str = "dinov2") -> None:
    if outfile.exists():
        print(f"warning: {outfile} already exists")
    # check expected columns in dataframe?
    # TODO: handle updating existing file more carefully

    with h5py.File(outfile, "w") as f:
        # create a group for image information
        img_grp = f.create_group("image")
        # numpy can't do ndarray of mixed type, so save image paths as one dataset
        print(df.head())
        img_grp.create_dataset(
            "paths", data=df["image_path"].to_numpy(), compression="gzip"
        )
        img_grp.attrs["last_modified"] = datetime.now().isoformat()
        # create a features dataset by model name
        # NOTE: support storing umap + clusters, and keep model feature derivatives together
        # seems easiest to store each column as a dataset
        img_grp.create_dataset(
            f"features/{model_name}",
            data=df["features"].to_numpy(),
            compression="gzip",
        )
        # TODO: save umap for associated model/features

        remainder_df = df.drop("image_path", "features")
        if remainder_df.columns:
            print("Warning: unsaved columns (%s)" % ",".join(remainder_df.columns))


def info(hfile: Path) -> dict:
    with h5py.File(hfile, "r") as f:
        img_dataset = f["image/paths"]
        info = {"image_paths": img_dataset.size, "features": {}}
        feature_grp = f["image/features"]
        for model in feature_grp.keys():
            features = f[f"image/features/{model}"]
            # check that they match?
            info["features"][model] = {"embeddings": features.shape}
        return info


def load_features(datafile: Path, model_name: str | None = None) -> pl.DataFrame:
    if not datafile.exists():
        raise ValueError(f"File not found: {datafile}")

    with h5py.File(datafile, "r") as f:
        print(f.keys())
        img_dataset = f["image/paths"]
        feature_grp = f["image/features"]
        if model_name is None:
            model_name = list(feature_grp.keys())[0]
        else:
            if model_name not in feature_grp.keys():
                raise ValueError(f"Features for {model_name} not found")
        features = feature_grp[model_name]

        # [:] = retrieve all scalar data
        return pl.DataFrame(
            data={
                "image_path": img_dataset[:],
                "features": features[:],
            }
        ).with_columns(
            # image path is loaded as binary string; convert to string
            image_path=pl.col.image_path.cast(pl.String)
        )


# df = pl.read_parquet(args.embeddings)

# info_details = [f"Details for {args.embeddings} :"]

# if "image_path" in df.columns and "features" in df.columns:
#     info_details.append(f"  {df.height:,} images with features extracted")
# else:
#     info_details.append(" image_path and features columns not found")

# if "umap" in df.columns:
#     info_details.append("  UMAP coordinates")

# print("\n".join(info_details))
