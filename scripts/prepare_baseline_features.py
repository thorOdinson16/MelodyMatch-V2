from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "fma_metadata" / "features.csv"
DATA_DIR = PROJECT_ROOT / "data"

SPLITS = [
    "clean_train",
    "clean_validation",
    "clean_test",
]


def load_features() -> pd.DataFrame:
    print("Loading FMA features...")

    features = pd.read_csv(
        FEATURES_PATH,
        header=[0, 1, 2],
        low_memory=False,
    )

    # --------------------------------------------------------
    # First column is the track ID
    # --------------------------------------------------------

    track_id = pd.to_numeric(
        features.iloc[:, 0],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remaining 518 columns are audio features
    # --------------------------------------------------------

    feature_values = features.iloc[:, 1:].copy()

    # Flatten MultiIndex feature names
    feature_values.columns = [
        "_".join(
            str(level)
            for level in column
            if str(level) != "nan"
        )
        for column in feature_values.columns
    ]

    feature_values.insert(
        0,
        "track_id",
        track_id,
    )

    feature_values = feature_values.dropna(
        subset=["track_id"]
    )

    feature_values["track_id"] = (
        feature_values["track_id"]
        .astype(int)
    )

    return feature_values


def prepare_split(
    features: pd.DataFrame,
    split_name: str,
) -> None:

    manifest_path = (
        DATA_DIR / f"{split_name}.csv"
    )

    output_path = (
        DATA_DIR / f"{split_name}_features.csv"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    print(
        f"\nPreparing {split_name}: "
        f"{len(manifest):,} tracks"
    )

    merged = manifest.merge(
        features,
        on="track_id",
        how="left",
        validate="one_to_one",
    )

    feature_columns = [
        column
        for column in merged.columns
        if column not in {
            "track_id",
            "genre",
            "audio_path",
        }
    ]

    missing = merged[
        feature_columns
    ].isna().any(axis=1)

    print(
        "Missing feature rows:",
        int(missing.sum()),
    )

    if missing.any():
        missing_ids = merged.loc[
            missing,
            "track_id",
        ].tolist()

        raise ValueError(
            f"Missing features for tracks: "
            f"{missing_ids[:20]}"
        )

    merged.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )


def main():
    features = load_features()

    print(
        f"Feature rows: "
        f"{len(features):,}"
    )

    print(
        f"Feature columns: "
        f"{len(features.columns) - 1}"
    )

    for split_name in tqdm(
        SPLITS,
        desc="Preparing splits",
        unit="split",
    ):
        prepare_split(
            features,
            split_name,
        )


if __name__ == "__main__":
    main()