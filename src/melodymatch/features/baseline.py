from pathlib import Path

import pandas as pd


def load_fma_features(features_path: str | Path) -> pd.DataFrame:
    """
    Load the FMA-provided feature table.

    The FMA features.csv file uses a three-level header.
    The first column contains track IDs.
    """

    features_path = Path(features_path)

    df = pd.read_csv(
        features_path,
        header=[0, 1, 2],
    )

    # The first column is the track ID.
    track_ids = df.iloc[:, 0]

    # Flatten the remaining MultiIndex feature names.
    feature_columns = []

    for column in df.columns[1:]:
        parts = [
            str(part)
            for part in column
            if str(part) != "nan"
        ]

        feature_columns.append("_".join(parts))

    features = df.iloc[:, 1:].copy()
    features.columns = feature_columns

    features.insert(
        0,
        "track_id",
        track_ids.astype(int),
    )

    return features


def select_tracks(
    features: pd.DataFrame,
    track_ids,
) -> pd.DataFrame:
    """
    Select features for a specific set of track IDs.
    """

    track_ids = set(int(track_id) for track_id in track_ids)

    selected = features[
        features["track_id"].isin(track_ids)
    ].copy()

    return selected