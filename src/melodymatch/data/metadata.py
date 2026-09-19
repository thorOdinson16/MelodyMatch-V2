from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    ("track", "bit_rate"),
    ("track", "duration"),
    ("track", "number"),
]

CATEGORICAL_COLUMNS = [
    ("track", "license"),
]


class FMAMetadataProcessor:

    def __init__(self):
        self.scaler = StandardScaler()

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        self.numeric_columns = NUMERIC_COLUMNS
        self.categorical_columns = CATEGORICAL_COLUMNS

    def load_metadata(self, metadata_path):

        df = pd.read_csv(
            metadata_path,
            header=[0, 1],
            index_col=0,
        )

        return df

    def fit(self, metadata_df):

        numeric = (
            metadata_df[
                self.numeric_columns
            ]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        categorical = (
            metadata_df[
                self.categorical_columns
            ]
            .fillna("unknown")
            .astype(str)
        )

        self.scaler.fit(numeric)

        self.encoder.fit(categorical)

        return self

    def transform(self, metadata_df):

        numeric = (
            metadata_df[
                self.numeric_columns
            ]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        categorical = (
            metadata_df[
                self.categorical_columns
            ]
            .fillna("unknown")
            .astype(str)
        )

        numeric_features = (
            self.scaler.transform(numeric)
        )

        categorical_features = (
            self.encoder.transform(categorical)
        )

        features = np.concatenate(
            [
                numeric_features,
                categorical_features,
            ],
            axis=1,
        )

        return features.astype(
            np.float32
        )

    @property
    def output_dim(self):

        return (
            len(self.numeric_columns)
            + len(
                self.encoder.get_feature_names_out(
                    self.categorical_columns
                )
            )
        )