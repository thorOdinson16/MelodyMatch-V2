from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from melodymatch.data.dataset import FMAMelDataset
from melodymatch.data.labels import GENRES, GENRE_TO_INDEX
from melodymatch.models.cnn import CNNBaseline


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST = PROJECT_ROOT / "data" / "clean_train.csv"
VAL_MANIFEST = PROJECT_ROOT / "data" / "clean_validation.csv"
TEST_MANIFEST = PROJECT_ROOT / "data" / "clean_test.csv"

MEL_CACHE_DIR = PROJECT_ROOT / "data" / "mel_cache"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 16

BATCH_SIZE = 32
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

# ------------------------------------------------------------
# SpecAugment
# ------------------------------------------------------------

# Mel input shape:
# [batch, 1, 128, 1292]

FREQ_MASK_PARAM = 12
TIME_MASK_PARAM = 80

AUGMENT_PROBABILITY = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SpecAugment
# ============================================================

def spec_augment(
    mel,
    freq_mask_param=FREQ_MASK_PARAM,
    time_mask_param=TIME_MASK_PARAM,
    probability=AUGMENT_PROBABILITY,
):
    """
    Apply SpecAugment independently to each sample.

    Input:
        mel: [B, 1, n_mels, time]

    Returns:
        augmented mel with the same shape.
    """

    if probability <= 0:
        return mel

    augmented = mel.clone()

    batch_size = augmented.size(0)
    n_mels = augmented.size(2)
    time_steps = augmented.size(3)

    for i in range(batch_size):

        # Randomly decide whether to augment this sample.
        if torch.rand(
            1,
            device=augmented.device,
        ).item() > probability:
            continue

        # ----------------------------------------------------
        # Frequency masking
        # ----------------------------------------------------

        max_freq_width = min(
            freq_mask_param,
            n_mels,
        )

        freq_width = torch.randint(
            low=0,
            high=max_freq_width + 1,
            size=(1,),
            device=augmented.device,
        ).item()

        if freq_width > 0:

            freq_start_max = (
                n_mels - freq_width
            )

            if freq_start_max > 0:

                freq_start = torch.randint(
                    low=0,
                    high=freq_start_max + 1,
                    size=(1,),
                    device=augmented.device,
                ).item()

                augmented[
                    i,
                    :,
                    freq_start:freq_start + freq_width,
                    :,
                ] = 0.0

        # ----------------------------------------------------
        # Time masking
        # ----------------------------------------------------

        max_time_width = min(
            time_mask_param,
            time_steps,
        )

        time_width = torch.randint(
            low=0,
            high=max_time_width + 1,
            size=(1,),
            device=augmented.device,
        ).item()

        if time_width > 0:

            time_start_max = (
                time_steps - time_width
            )

            if time_start_max > 0:

                time_start = torch.randint(
                    low=0,
                    high=time_start_max + 1,
                    size=(1,),
                    device=augmented.device,
                ).item()

                augmented[
                    i,
                    :,
                    :,
                    time_start:time_start + time_width,
                ] = 0.0

    return augmented


# ============================================================
# Class weights
# ============================================================

def calculate_class_weights(train_df):

    counts = (
        train_df["genre"]
        .value_counts()
        .reindex(GENRES)
    )

    counts = counts.values.astype(
        np.float64
    )

    total = counts.sum()

    # Same weighting strategy as the original CNN.
    weights = total / (
        len(GENRES) * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            mel = batch["mel"].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch["label"].to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(mel)

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions,
        )
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Experiment #10 — CNN + SpecAugment")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("\nSpecAugment configuration:")

    print(
        f"  Frequency mask: "
        f"{FREQ_MASK_PARAM} Mel bins"
    )

    print(
        f"  Time mask: "
        f"{TIME_MASK_PARAM} frames"
    )

    print(
        f"  Augmentation probability: "
        f"{AUGMENT_PROBABILITY}"
    )

    set_seed(42)

    start_time = time.time()

    # ========================================================
    # Load manifests
    # ========================================================

    train_df = pd.read_csv(
        TRAIN_MANIFEST
    )

    val_df = pd.read_csv(
        VAL_MANIFEST
    )

    test_df = pd.read_csv(
        TEST_MANIFEST
    )

    print(
        f"\nTrain samples: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_df):,}"
    )

    print(
        f"Test samples: "
        f"{len(test_df):,}"
    )

    # ========================================================
    # Datasets
    # ========================================================

    train_dataset = FMAMelDataset(
        manifest_path=TRAIN_MANIFEST,
        project_root=PROJECT_ROOT,
        genre_to_index=GENRE_TO_INDEX,
        cache_dir=MEL_CACHE_DIR,
    )

    val_dataset = FMAMelDataset(
        manifest_path=VAL_MANIFEST,
        project_root=PROJECT_ROOT,
        genre_to_index=GENRE_TO_INDEX,
        cache_dir=MEL_CACHE_DIR,
    )

    test_dataset = FMAMelDataset(
        manifest_path=TEST_MANIFEST,
        project_root=PROJECT_ROOT,
        genre_to_index=GENRE_TO_INDEX,
        cache_dir=MEL_CACHE_DIR,
    )

    print(
        "\nDatasets loaded."
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # ========================================================
    # Class weights
    # ========================================================

    class_weights = (
        calculate_class_weights(
            train_df
        ).to(DEVICE)
    )

    print(
        "\nClass weights:"
    )

    for genre, weight in zip(
        GENRES,
        class_weights.cpu().numpy(),
    ):

        print(
            f"  {genre:<22} "
            f"{weight:.4f}"
        )

    # ========================================================
    # Model
    # ========================================================

    model = CNNBaseline(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nParameters: "
        f"{parameter_count:,}"
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    # ========================================================
    # Training
    # ========================================================

    best_val_macro_f1 = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    checkpoint_path = (
        CHECKPOINT_DIR
        / "cnn_specaugment.pt"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        train_predictions = []
        train_labels = []

        progress = tqdm(
            train_loader,
            desc=(
                f"SpecAugment | "
                f"Epoch {epoch:02d}/"
                f"{MAX_EPOCHS}"
            ),
            leave=True,
        )

        for batch in progress:

            mel = batch["mel"].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch["label"].to(
                DEVICE,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Apply SpecAugment ONLY during training.
            # Validation and test are untouched.
            # ------------------------------------------------

            mel = spec_augment(
                mel
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(mel)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            train_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            train_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # ====================================================
        # Training metrics
        # ====================================================

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions,
        )

        train_balanced_accuracy = (
            balanced_accuracy_score(
                train_labels,
                train_predictions,
            )
        )

        train_macro_f1 = f1_score(
            train_labels,
            train_predictions,
            average="macro",
            zero_division=0,
        )

        train_weighted_f1 = f1_score(
            train_labels,
            train_predictions,
            average="weighted",
            zero_division=0,
        )

        # ====================================================
        # Validation
        # ====================================================

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_balanced_accuracy": (
                    train_balanced_accuracy
                ),
                "train_macro_f1": train_macro_f1,
                "train_weighted_f1": (
                    train_weighted_f1
                ),
                "val_loss": val_metrics["loss"],
                "val_accuracy": (
                    val_metrics["accuracy"]
                ),
                "val_balanced_accuracy": (
                    val_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "val_macro_f1": (
                    val_metrics["macro_f1"]
                ),
                "val_weighted_f1": (
                    val_metrics["weighted_f1"]
                ),
            }
        )

        print(
            f"\nEpoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Train Bal Acc: "
            f"{train_balanced_accuracy:.4f} | "
            f"Train Macro-F1: "
            f"{train_macro_f1:.4f}"
        )

        print(
            f"           "
            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "
            f"Val Acc: "
            f"{val_metrics['accuracy']:.4f} | "
            f"Val Bal Acc: "
            f"{val_metrics['balanced_accuracy']:.4f} | "
            f"Val Macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        # ====================================================
        # Best checkpoint
        # ====================================================

        if (
            val_metrics["macro_f1"]
            > best_val_macro_f1
        ):

            best_val_macro_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1":
                        best_val_macro_f1,
                    "class_weights":
                        class_weights.cpu(),
                },
                checkpoint_path,
            )

            print(
                f"  ✓ New best checkpoint "
                f"(Val Macro-F1: "
                f"{best_val_macro_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

        # ====================================================
        # Early stopping
        # ====================================================

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"\nEarly stopping at "
                f"epoch {epoch}."
            )

            break

    # ========================================================
    # Load best checkpoint
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # Final evaluation
    # ========================================================

    train_metrics = evaluate(
        model,
        train_loader,
        criterion,
    )

    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
    )

    runtime_minutes = (
        time.time() - start_time
    ) / 60.0

    # ========================================================
    # Final results
    # ========================================================

    print()
    print("-" * 70)
    print("FINAL RESULTS — CNN + SpecAugment")
    print("-" * 70)

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Best Validation Macro-F1: "
        f"{best_val_macro_f1:.4f}"
    )

    print("\nTrain:")

    print(
        f"  Loss: "
        f"{train_metrics['loss']:.4f}"
    )

    print(
        f"  Accuracy: "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{train_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{train_metrics['macro_f1']:.4f}"
    )

    print(
        f"  Weighted-F1: "
        f"{train_metrics['weighted_f1']:.4f}"
    )

    print("\nValidation:")

    print(
        f"  Loss: "
        f"{val_metrics['loss']:.4f}"
    )

    print(
        f"  Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{val_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{val_metrics['macro_f1']:.4f}"
    )

    print(
        f"  Weighted-F1: "
        f"{val_metrics['weighted_f1']:.4f}"
    )

    print("\nTest:")

    print(
        f"  Loss: "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"  Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"  Weighted-F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        f"\nRuntime: "
        f"{runtime_minutes:.2f} minutes"
    )

    # ========================================================
    # Save metrics
    # ========================================================

    metrics_row = {
        "model": "CNN_SpecAugment",
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_val_macro_f1":
            best_val_macro_f1,

        "freq_mask_param":
            FREQ_MASK_PARAM,

        "time_mask_param":
            TIME_MASK_PARAM,

        "augmentation_probability":
            AUGMENT_PROBABILITY,

        "train_loss":
            train_metrics["loss"],

        "train_accuracy":
            train_metrics["accuracy"],

        "train_balanced_accuracy":
            train_metrics[
                "balanced_accuracy"
            ],

        "train_macro_f1":
            train_metrics["macro_f1"],

        "train_weighted_f1":
            train_metrics[
                "weighted_f1"
            ],

        "val_loss":
            val_metrics["loss"],

        "val_accuracy":
            val_metrics["accuracy"],

        "val_balanced_accuracy":
            val_metrics[
                "balanced_accuracy"
            ],

        "val_macro_f1":
            val_metrics["macro_f1"],

        "val_weighted_f1":
            val_metrics[
                "weighted_f1"
            ],

        "test_loss":
            test_metrics["loss"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_balanced_accuracy":
            test_metrics[
                "balanced_accuracy"
            ],

        "test_macro_f1":
            test_metrics["macro_f1"],

        "test_weighted_f1":
            test_metrics[
                "weighted_f1"
            ],

        "runtime_minutes":
            runtime_minutes,
    }

    metrics_path = (
        METRICS_DIR
        / "cnn_specaugment.csv"
    )

    pd.DataFrame(
        [metrics_row]
    ).to_csv(
        metrics_path,
        index=False,
    )

    # ========================================================
    # Save training history
    # ========================================================

    history_path = (
        METRICS_DIR
        / "cnn_specaugment_training_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    # ========================================================
    # Test classification report
    # ========================================================

    report = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=GENRES,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "class"
            }
        )
    )

    report_path = (
        METRICS_DIR
        / "cnn_specaugment_test_classification_report.csv"
    )

    report_df.to_csv(
        report_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("EXPERIMENT #10 COMPLETE")
    print("=" * 70)

    print(
        "\nSaved:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {history_path}"
    )

    print(
        f"  {report_path}"
    )

    print(
        f"  {checkpoint_path}"
    )


if __name__ == "__main__":
    main()