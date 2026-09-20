# MelodyMatch-V2

Research-oriented music genre classification using the **FMA Medium** dataset, with comparisons between traditional machine-learning methods, convolutional neural networks, temporal models, Transformers, metadata fusion, and class-imbalance strategies.

## Project Evolution

**MelodyMatch-V2** is the research-focused continuation of the original [MelodyMatch](https://github.com/thorOdinson16/MelodyMatch) project.

The original MelodyMatch project explored music genre classification using the **FMA Small** dataset, consisting of 8,000 tracks across 8 genres.

MelodyMatch-V2 extends that work to **FMA Medium**, using 25,000 locally available tracks across 16 genres. It introduces a redesigned experimental pipeline with:

- stricter dataset validation and cleaning
- official FMA train/validation/test splits
- Mel-spectrogram-based deep learning
- additional temporal and Transformer architectures
- metadata fusion
- systematic class-imbalance experiments
- decoupled representation and classifier training
- reproducible evaluation and analysis

The goal of V2 is to move from the original proof-of-concept toward a more systematic study of music genre classification under severe class imbalance.

---

## Overview

**MelodyMatch-V2** investigates how different representations, architectures, and training strategies affect multi-class music genre classification under a highly imbalanced dataset.

The project evaluates:

- Traditional ML using FMA-provided audio features
- CNN-based learning from Mel spectrograms
- CNN + BiLSTM temporal modeling
- CNN + Transformer modeling
- Audio + metadata fusion
- Multiple class-imbalance strategies
- Decoupled representation/classifier training

The primary evaluation metric is **Macro-F1**, with Accuracy, Balanced Accuracy, and Weighted-F1 also reported.

---

## Research Questions

The experiments investigate two main questions:

1. **How do different model architectures affect music genre classification?**
   - CNN
   - CNN + BiLSTM
   - CNN + Transformer
   - CNN + Transformer + metadata

2. **How do different strategies for handling severe class imbalance affect performance?**
   - Class-weighted Cross Entropy
   - Smoothed class weights
   - SpecAugment
   - Mixup
   - Focal Loss
   - Balanced sampling
   - Decoupled training
   - Multi-scale CNN

---

## Dataset

The project uses the **FMA Medium** subset of the Free Music Archive dataset.

### Dataset statistics

| Property | Value |
|---|---:|
| Original local tracks | 25,000 |
| Final usable tracks | 24,980 |
| Genres | 16 |
| Training | 19,904 |
| Validation | 2,504 |
| Test | 2,572 |
| Audio duration | ~30 seconds |
| Sample rate | 22,050 Hz |
| Mel bins | 128 |

The official FMA train/validation/test assignments are used.

### Dataset cleaning

20 tracks were excluded:

- 15 invalid/unreadable audio files
- 5 tracks shorter than 10 seconds

The original audio files are not modified or deleted. Cleaning is performed through the project manifests.

The final dataset remains highly imbalanced, with **Rock** and **Electronic** representing the largest classes and several genres containing very few examples.

---

## Audio Representation

Each track is converted into a fixed-length Mel spectrogram.

```text
30-second audio
      │
      ▼
Mono waveform
      │
      ▼
22,050 Hz
      │
      ▼
128-bin Mel spectrogram
      │
      ▼
Log-power / dB conversion
      │
      ▼
Per-spectrogram Z-score normalization
      │
      ▼
[1 × 128 × 1292]
````

Configuration:

```text
Sample rate : 22,050 Hz
Duration    : 30 seconds
N_MELS      : 128
N_FFT       : 2048
HOP_LENGTH  : 512
FMIN        : 20 Hz
FMAX        : 11,025 Hz
```

Mel spectrograms are cached locally to avoid repeatedly decoding the audio during model training.

---

## Models

### Traditional Baselines

The FMA-provided 518-dimensional feature representation is used for:

1. Logistic Regression
2. RBF SVM
3. MLP

### Deep Learning Models

Mel spectrograms are used for:

4. CNN
5. CNN + BiLSTM
6. CNN + Transformer
7. CNN + Transformer + Metadata

Additional experiments investigate:

* CNN + BiLSTM + Attention
* Smoothed class weights
* SpecAugment
* Mixup
* Focal Loss
* Balanced sampling
* Decoupled training
* Multi-scale CNN

---

## Final Results

Results below are from the held-out test set.

| Model                        |   Accuracy | Balanced Accuracy |   Macro-F1 | Weighted-F1 |
| ---------------------------- | ---------: | ----------------: | ---------: | ----------: |
| Logistic Regression          |     53.23% |            38.78% |     35.28% |      55.72% |
| SVM (RBF)                    | **63.65%** |            41.61% |     43.05% |      61.87% |
| MLP                          |     58.20% |            43.87% |     41.49% |      59.54% |
| CNN                          |     60.61% |            51.58% |     45.27% |      62.43% |
| CNN + BiLSTM                 |     57.50% |            45.65% |     41.16% |      59.19% |
| CNN + Transformer            |     48.60% |            36.80% |     27.67% |      48.45% |
| CNN + Transformer + Metadata |     40.98% |            32.21% |     29.38% |      44.92% |
| CNN + BiLSTM + Attention     |     58.55% |            44.31% |     39.32% |      59.75% |
| CNN + SpecAugment            |     58.20% |            49.50% |     42.79% |      60.07% |
| CNN + Mixup                  |     60.93% |            52.45% |     44.40% |      62.23% |
| CNN + Balanced Sampling      |     62.79% |            47.94% |     44.52% |      63.81% |
| **CNN + Decoupled Training** |     62.52% |        **54.37%** | **45.65%** |  **64.50%** |
| CNN + Multi-Scale            |     59.84% |            47.99% |     41.62% |      60.98% |

The focal-loss and smoothed-class-weight experiments are reported separately in the detailed experiment results.

### Main observations

* The RBF SVM obtained the highest test accuracy among the evaluated models.
* The decoupled CNN obtained the highest test Balanced Accuracy, Macro-F1, and Weighted-F1 in the evaluated experiments.
* The original CNN provides a strong baseline despite having a relatively small architecture.
* Adding BiLSTM or Transformer layers did not improve over the CNN baseline.
* Metadata fusion improved the CNN + Transformer Macro-F1 slightly but reduced the other reported test metrics.
* Several imbalance strategies changed the precision/recall trade-off without consistently improving all metrics.
* Decoupled training improved all four reported test metrics relative to the original CNN.

---

## Decoupled Training

The final decoupled experiment separates representation learning from classifier rebalancing.

### Stage 1 — Representation learning

The CNN is trained on the original imbalanced training distribution using standard Cross Entropy.

### Stage 2 — Classifier rebalancing

The learned CNN feature extractor is frozen.

The final classifier is reinitialized and trained using balanced sampling.

```text
Stage 1
Audio → CNN feature extractor → Classifier
                 │
                 ▼
       learned representation

Stage 2
Audio → Frozen CNN → Reinitialized Classifier
                         │
                         ▼
                  balanced sampling
```

This produced:

```text
Accuracy          62.52%
Balanced Accuracy 54.37%
Macro-F1          45.65%
Weighted-F1       64.50%
```

---

## Evaluation

All models are evaluated using:

* Accuracy
* Balanced Accuracy
* Macro-F1
* Weighted-F1
* Per-class Precision
* Per-class Recall
* Per-class F1
* Confusion Matrix

**Macro-F1 is the primary metric** because the dataset is highly imbalanced.

---

## Repository Structure

```text
MelodyMatch/
│
├── configs/
│   └── default.yaml
│
├── data/
│   └── ...
│
├── experiments/
│   ├── 01_baselines/
│   ├── 02_cnn/
│   ├── 03_temporal_models/
│   ├── 04_metadata/
│   ├── 05_imbalance/
│   └── 06_final_comparison/
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_baseline_comparison.ipynb
│   ├── 03_deep_model_comparison.ipynb
│   ├── 04_imbalance_analysis.ipynb
│   ├── 05_confusion_analysis.ipynb
│   └── 06_final_results.ipynb
│
├── results/
│   ├── checkpoints/
│   ├── figures/
│   └── metrics/
│
├── scripts/
│   ├── build_manifest.py
│   ├── validate_audio.py
│   ├── create_splits.py
│   ├── clean_manifest.py
│   ├── preprocess_audio.py
│   ├── prepare_baseline_features.py
│   └── ...
│
├── src/
│   └── melodymatch/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       └── training/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/thorOdinson16/MelodyMatch-V2.git
cd MelodyMatch-V2
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install MelodyMatch as an editable package:

```bash
python -m pip install -e .
```

---

## Dataset Setup

Download the FMA Medium audio and metadata separately.

The project expects:

```text
fma_medium/
fma_metadata/
```

Then build the initial manifest:

```bash
python scripts/build_manifest.py
```

Create the official dataset splits:

```bash
python scripts/create_splits.py
```

Validate the audio:

```bash
python scripts/validate_audio.py
```

Create the final cleaned manifests:

```bash
python scripts/clean_manifest.py
```

---

## Preprocessing

Generate the cached Mel spectrograms:

```bash
python scripts/preprocess_audio.py
```

Prepare the FMA-provided features for the traditional baselines:

```bash
python scripts/prepare_baseline_features.py
```

---

## Training

Examples:

```bash
python scripts/train_logistic_baseline.py
python scripts/train_svm_baseline.py
python scripts/train_mlp_baseline.py

python scripts/train_cnn.py
python scripts/train_cnn_bilstm.py
python scripts/train_cnn_transformer.py
python scripts/train_transformer_metadata.py

python scripts/train_cnn_decoupled.py
```

All long-running training and preprocessing operations report progress using `tqdm`.

---

## Analysis

The notebooks provide the analysis pipeline:

```text
01_dataset_analysis
        ↓
02_baseline_comparison
        ↓
03_deep_model_comparison
        ↓
04_imbalance_analysis
        ↓
05_confusion_analysis
        ↓
06_final_results
```

Generated figures are stored under:

```text
results/figures/
```

Metrics and classification reports are stored under:

```text
results/metrics/
```

---

## Reproducibility

The repository contains:

* Dataset manifests
* Configuration files
* Preprocessing configuration
* Training scripts
* Evaluation utilities
* Experiment documentation
* Metrics
* Analysis notebooks
* Final model checkpoints
* Generated figures

The original FMA audio files and large generated feature/cache files are not included in the repository.

---

## Research Report

A detailed LaTeX report documents the complete methodology, experimental setup, results, analysis, limitations, and conclusions.

See:

```text
report/
```

for the full research report.

---

## License

This project is intended for research and educational use.

The FMA dataset is distributed under its own licensing terms. Users should consult the original FMA dataset documentation and individual track licenses before redistributing or using the audio commercially.

---

## Acknowledgements

This project uses the **Free Music Archive (FMA)** dataset and its associated metadata and precomputed features.