# Metadata Fusion

## Purpose

Investigate whether track metadata provides useful information beyond the Mel-spectrogram representation.

## Model

CNN + Transformer + Metadata

### Audio branch

Mel spectrogram
→ CNN
→ Transformer
→ audio embedding

### Metadata branch

Selected metadata:

- `track.bit_rate`
- `track.duration`
- `track.number`
- `track.license`

Excluded fields include genre-derived fields and engagement/popularity variables to avoid label leakage and popularity-driven signals.

Missing license values are represented as `unknown`.

Numeric metadata is standardized and license is one-hot encoded using the training split.

The audio and metadata embeddings are concatenated before classification.

## Results

| Metric | CNN + Transformer | CNN + Transformer + Metadata |
|---|---:|---:|
| Accuracy | 48.60% | 40.98% |
| Balanced Accuracy | 36.80% | 32.21% |
| Macro-F1 | 27.67% | 29.38% |
| Weighted-F1 | 48.45% | 44.92% |

## Observation

Metadata fusion increases Macro-F1 by 1.71 percentage points but decreases accuracy, balanced accuracy, and weighted-F1.

Therefore, metadata does not provide a uniform performance improvement under this configuration.