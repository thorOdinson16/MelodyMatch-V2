from pathlib import Path

import torch

from melodymatch.data.preprocessing import audio_to_mel


def extract_mel_feature(
    audio_path: str | Path,
) -> torch.Tensor:
    """
    Convert an audio file into a normalized Mel-spectrogram tensor.
    """

    return audio_to_mel(audio_path)


def load_cached_mel(
    cache_path: str | Path,
) -> torch.Tensor:
    """
    Load a cached Mel-spectrogram tensor.
    """

    return torch.load(
        cache_path,
        weights_only=True,
    )