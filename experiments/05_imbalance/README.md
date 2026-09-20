# Class Imbalance Experiments

## Purpose

Investigate different approaches for handling the severe class imbalance present in FMA Medium.

The dataset contains 16 genres with highly different numbers of examples.

## Experiments

### Smoothed Class Weights

Weights:

w_c = (N / N_c)^alpha

Evaluated:

- α = 0.25
- α = 0.50
- α = 0.75
- α = 1.00

Best results by metric:

| α | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|---:|
| 0.25 | 63.80% | 42.96% | 40.99% | 60.54% |
| 0.50 | 62.36% | 50.15% | 44.21% | 62.64% |
| 0.75 | 62.36% | 50.91% | 44.15% | 62.54% |
| 1.00 | 60.26% | 50.62% | 44.74% | 61.65% |

### SpecAugment

Training-time frequency and time masking were applied.

| Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 58.20% | 49.50% | 42.79% | 60.07% |

### Mixup

Mixup was applied during training with α = 0.4 and probability 0.5.

| Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 60.93% | 52.45% | 44.40% | 62.23% |

### Focal Loss

Evaluated with γ ∈ {1, 2, 3}.

| γ | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|---:|
| 1 | 56.22% | 43.33% | 36.08% | 57.44% |
| 2 | 55.40% | 46.86% | 40.13% | 57.96% |
| 3 | 55.56% | 42.93% | 35.27% | 56.30% |

### Balanced Sampling

Inverse-frequency WeightedRandomSampler was used during training.

| Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 62.79% | 47.94% | 44.52% | 63.81% |

### Decoupled Training

Training was separated into two stages.

#### Stage 1

The CNN representation was trained using the natural imbalanced training distribution.

#### Stage 2

The convolutional feature extractor was frozen and the classifier was reinitialized and trained using balanced sampling.

Test results:

| Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 62.52% | 54.37% | 45.65% | 64.50% |

## Observation

Decoupled training improves all four primary test metrics relative to the CNN baseline.

Compared with the CNN baseline:

- Accuracy: +1.91 pp
- Balanced Accuracy: +2.79 pp
- Macro-F1: +0.38 pp
- Weighted-F1: +2.07 pp

It is the strongest overall CNN-based configuration evaluated in this experiment set.