# Temporal and Attention Models

## Purpose

Investigate whether explicitly modeling temporal dependencies in the Mel-spectrogram representation improves genre classification.

## Experiments

### CNN + BiLSTM

The CNN feature extractor produces a temporal sequence which is processed using a bidirectional LSTM.

Test results:

| Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 57.50% | 45.65% | 41.16% | 59.19% |

### CNN + Transformer

CNN features are projected into Transformer tokens and processed using Transformer encoder layers.

Test results:

| Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 48.60% | 36.80% | 27.67% | 48.45% |

### CNN + BiLSTM + Attention

A learned attention mechanism is added to the BiLSTM representation before classification.

Test results:

| Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|
| 58.55% | 44.31% | 39.32% | 59.75% |

## Observation

Under the current experimental setup, the temporal and attention-based architectures do not improve over the CNN baseline.

The results indicate that increased architectural complexity does not automatically produce better genre classification performance for this dataset and representation.