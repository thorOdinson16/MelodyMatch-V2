from pathlib import Path

import librosa
import numpy as np
import torch


# ============================================================
# Audio configuration
# ============================================================

SAMPLE_RATE = 22050
DURATION_SECONDS = 30.0

NUM_SAMPLES = int(
    SAMPLE_RATE * DURATION_SECONDS
)


# ============================================================
# Mel-spectrogram configuration
# ============================================================

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

FMIN = 20
FMAX = SAMPLE_RATE // 2


# ============================================================
# Preprocessing configuration
# ============================================================

PREPROCESSING_CONFIG = {
    "sample_rate": SAMPLE_RATE,
    "duration_seconds": DURATION_SECONDS,
    "num_samples": NUM_SAMPLES,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "fmin": FMIN,
    "fmax": FMAX,
    "mono": True,
    "power": 2.0,
    "normalization": "per_spectrogram_zscore",
}


# ============================================================
# Load audio
# ============================================================

def load_audio(
    audio_path: str | Path,
) -> np.ndarray:
    """
    Load audio as mono at the fixed target sample rate.

    Audio shorter than 30 seconds is zero-padded.
    Audio longer than 30 seconds is truncated.
    """

    y, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    # --------------------------------------------------------
    # Fix length
    # --------------------------------------------------------

    if len(y) < NUM_SAMPLES:

        y = np.pad(
            y,
            (
                0,
                NUM_SAMPLES - len(y),
            ),
            mode="constant",
        )

    elif len(y) > NUM_SAMPLES:

        y = y[:NUM_SAMPLES]

    return y.astype(
        np.float32
    )


# ============================================================
# Audio → Mel spectrogram
# ============================================================

def audio_to_mel(
    audio_path: str | Path,
) -> torch.Tensor:
    """
    Convert an audio file into a normalized log-Mel
    spectrogram.

    Returns:
        Tensor with shape:

            [1, N_MELS, TIME]
    """

    y = load_audio(
        audio_path
    )

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    # --------------------------------------------------------
    # Convert power → decibels
    # --------------------------------------------------------

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # --------------------------------------------------------
    # Per-spectrogram normalization
    # --------------------------------------------------------

    mean = mel_db.mean()
    std = mel_db.std()

    mel_db = (
        mel_db - mean
    ) / (
        std + 1e-8
    )

    # --------------------------------------------------------
    # Add channel dimension
    #
    # [Mel, Time]
    #       ↓
    # [1, Mel, Time]
    # --------------------------------------------------------

    mel_tensor = torch.from_numpy(
        mel_db.astype(
            np.float32
        )
    ).unsqueeze(0)

    return mel_tensor