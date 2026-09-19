from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import audio_to_mel


class FMAMelDataset(Dataset):
    """
    PyTorch Dataset for FMA Medium Mel spectrograms.

    Expected CSV columns:

        track_id
        genre
        audio_path
        split
    """

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        genre_to_index: dict[str, int],
        cache_dir: str | Path | None = None,
    ):

        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.genre_to_index = genre_to_index

        self.df = pd.read_csv(
            self.manifest_path
        )

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        if cache_dir is not None:

            self.cache_dir = Path(cache_dir)

            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:

            self.cache_dir = None

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.df)

    # ========================================================
    # Cache path
    # ========================================================

    def _cache_path(self, track_id):

        if self.cache_dir is None:
            return None

        return self.cache_dir / f"{int(track_id):06d}.pt"

    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(self, index):

        row = self.df.iloc[index]

        track_id = int(row["track_id"])

        audio_path = (
            self.project_root
            / row["audio_path"]
        )

        # ----------------------------------------------------
        # Load cached spectrogram
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = self.genre_to_index[
            row["genre"]
        ]

        return {
            "mel": mel,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "track_id": track_id,
        }