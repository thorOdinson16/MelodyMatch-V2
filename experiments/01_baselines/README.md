# Traditional Baselines

## Purpose

Establish traditional machine-learning baselines using the 518-dimensional features provided by the FMA dataset.

## Input

- Dataset: FMA Medium
- Features: FMA-provided `features.csv`
- Number of features: 518
- Classes: 16 genres

## Models

1. Logistic Regression
2. SVM with RBF kernel
3. Multi-Layer Perceptron (MLP)

## Preprocessing

- StandardScaler applied to input features.
- Class weighting used to address class imbalance.
- Official FMA train/validation/test split preserved.

## Results

| Model | Test Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 53.23% | 38.78% | 35.28% | 55.72% |
| SVM (RBF) | 63.65% | 41.61% | 43.05% | 61.87% |
| MLP | 58.20% | 43.87% | 41.49% | 59.54% |

## Observation

The SVM achieves the highest raw test accuracy among the traditional baselines, while the MLP provides slightly higher Macro-F1 than SVM.

The traditional models operate on the FMA-provided engineered features rather than directly modeling the audio spectrogram.