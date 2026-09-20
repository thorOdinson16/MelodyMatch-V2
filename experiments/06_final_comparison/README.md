# Final Model Comparison

## Complete Test-Set Comparison

| # | Model | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---|---:|---:|---:|---:|
| 1 | Logistic Regression | 53.23% | 38.78% | 35.28% | 55.72% |
| 2 | SVM (RBF) | 63.65% | 41.61% | 43.05% | 61.87% |
| 3 | MLP | 58.20% | 43.87% | 41.49% | 59.54% |
| 4 | CNN | 60.61% | 51.58% | 45.27% | 62.43% |
| 5 | CNN + BiLSTM | 57.50% | 45.65% | 41.16% | 59.19% |
| 6 | CNN + Transformer | 48.60% | 36.80% | 27.67% | 48.45% |
| 7 | CNN + Transformer + Metadata | 40.98% | 32.21% | 29.38% | 44.92% |
| 8 | CNN + BiLSTM + Attention | 58.55% | 44.31% | 39.32% | 59.75% |
| 9 | CNN + Smoothed Weights | 60.26–63.80% | 42.96–50.91% | 40.99–44.74% | 60.54–62.64% |
| 10 | CNN + SpecAugment | 58.20% | 49.50% | 42.79% | 60.07% |
| 11 | CNN + Mixup | 60.93% | 52.45% | 44.40% | 62.23% |
| 12 | CNN + Focal Loss | 55.40–56.22% | 42.93–46.86% | 35.27–40.13% | 56.30–57.96% |
| 13 | CNN + Balanced Sampling | 62.79% | 47.94% | 44.52% | 63.81% |
| 14 | CNN + Decoupled Training | 62.52% | 54.37% | 45.65% | 64.50% |
| 15 | CNN + Multi-Scale | 59.84% | 47.99% | 41.62% | 60.98% |

## Key Findings

- SVM achieves the highest test accuracy: 63.65%.
- Decoupled CNN achieves the highest balanced accuracy: 54.37%.
- Decoupled CNN achieves the highest Macro-F1: 45.65%.
- Decoupled CNN achieves the highest Weighted-F1: 64.50%.
- The original CNN remains a strong baseline despite its substantially smaller complexity than the temporal and Transformer models.
- More complex temporal and Transformer architectures did not improve performance under the evaluated configuration.
- Decoupled training improves all four primary metrics over the CNN baseline.

## Per-Class Analysis

The decoupled model improves F1 for several classes, including:

- Country
- International
- Instrumental
- Electronic
- Experimental
- Rock
- Soul-RnB
- Folk
- Hip-Hop

The largest F1 decreases occur for:

- Blues
- Spoken
- Pop

The confusion matrices should be used together with these per-class results to interpret these changes.