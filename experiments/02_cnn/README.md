# CNN Baseline

## Purpose

Evaluate a convolutional neural network directly on Mel spectrograms.

## Input

- Audio duration: 30 seconds
- Sample rate: 22,050 Hz
- Mel bins: 128
- FFT size: 2048
- Hop length: 512
- Input shape: `[1, 128, 1292]`
- Normalization: per-spectrogram z-score

## Architecture

The CNN contains four convolutional blocks:

- 1 → 32 channels
- 32 → 64 channels
- 64 → 128 channels
- 128 → 256 channels

Each block uses:

- 3×3 convolution
- BatchNorm
- ReLU
- 2×2 max pooling

Adaptive average pooling is followed by dropout and a 16-class classifier.

## Training

- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4
- Batch size: 32
- Maximum epochs: 30
- Early stopping patience: 7
- Class-weighted Cross Entropy
- Model selection: validation Macro-F1

## Results

| Metric | Test |
|---|---:|
| Accuracy | 60.61% |
| Balanced Accuracy | 51.58% |
| Macro-F1 | 45.27% |
| Weighted-F1 | 62.43% |

## Role

This model serves as the main deep-learning baseline against which the subsequent architectural and imbalance experiments are compared.