from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import audio_to_mel


class FMAMelMetadataDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        project_root,
        genre_to_index,
        metadata_features,
        cache_dir=None,
    ):
        self.manifest_path = Path(
            manifest_path
        )

        self.project_root = Path(
            project_root
        )

        self.genre_to_index = (
            genre_to_index
        )

        self.df = pd.read_csv(
            self.manifest_path
        )

        self.metadata_features = torch.tensor(
            metadata_features,
            dtype=torch.float32,
        )

        if len(self.metadata_features) != len(
            self.df
        ):
            raise ValueError(
                "Number of metadata rows does "
                "not match manifest rows."
            )

        if cache_dir is not None:
            self.cache_dir = Path(
                cache_dir
            )

            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:
            self.cache_dir = None

    def __len__(self):
        return len(self.df)

    def _cache_path(self, track_id):

        if self.cache_dir is None:
            return None

        return (
            self.cache_dir
            / f"{int(track_id):06d}.pt"
        )

    def __getitem__(self, index):

        row = self.df.iloc[index]

        track_id = int(
            row["track_id"]
        )

        audio_path = (
            self.project_root
            / row["audio_path"]
        )

        cache_path = self._cache_path(
            track_id
        )

        if (
            cache_path is not None
            and cache_path.exists()
        ):
            mel = torch.load(
                cache_path,
                weights_only=True,
            )
        else:
            mel = audio_to_mel(
                audio_path
            )

            if cache_path is not None:
                torch.save(
                    mel,
                    cache_path,
                )

        label = self.genre_to_index[
            row["genre"]
        ]

        metadata = self.metadata_features[
            index
        ]

        return {
            "mel": mel,
            "metadata": metadata,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "track_id": track_id,
        }