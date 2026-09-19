from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from melodymatch.data.dataset import FMAMelDataset
from melodymatch.data.labels import GENRES
from melodymatch.models.cnn_bilstm_attention import (
    CNNBiLSTMAttention,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST = PROJECT_ROOT / "data" / "clean_train.csv"
VAL_MANIFEST = PROJECT_ROOT / "data" / "clean_validation.csv"
TEST_MANIFEST = PROJECT_ROOT / "data" / "clean_test.csv"

MEL_CACHE = PROJECT_ROOT / "data" / "mel_cache"

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

LR_FACTOR = 0.5
LR_PATIENCE = 2

NUM_CLASSES = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Utility functions
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def evaluate(
    model,
    loader,
    criterion,
    split_name,
):
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    progress = tqdm(
        loader,
        desc=f"{split_name}",
        leave=False,
    )

    with torch.no_grad():
        for batch in progress:
            mel = batch["mel"].to(
                DEVICE,
                non_blocking=True,
            )
            labels = batch["label"].to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(mel)
            loss = criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )
            y_pred.extend(
                predictions.cpu().numpy()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    average_loss = (
        running_loss / len(loader.dataset)
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    metrics["loss"] = average_loss

    return metrics, y_true, y_pred


def calculate_class_weights(train_df):
    counts = (
        train_df["genre"]
        .value_counts()
        .reindex(GENRES)
    )

    counts = counts.fillna(0)

    total = counts.sum()
    num_classes = len(counts)

    weights = total / (
        num_classes * counts
    )

    return torch.tensor(
        weights.values,
        dtype=torch.float32,
    )


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("CNN + BiLSTM + Attention")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

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
        f"Train: {len(train_df)}"
    )
    print(
        f"Validation: {len(val_df)}"
    )
    print(
        f"Test: {len(test_df)}"
    )

    # --------------------------------------------------------
    # Genre mapping
    # --------------------------------------------------------

    genre_to_index = {
        genre: index
        for index, genre in enumerate(GENRES)
    }

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = FMAMelDataset(
        manifest_path=TRAIN_MANIFEST,
        project_root=PROJECT_ROOT,
        genre_to_index=genre_to_index,
        cache_dir=MEL_CACHE,
    )

    val_dataset = FMAMelDataset(
        manifest_path=VAL_MANIFEST,
        project_root=PROJECT_ROOT,
        genre_to_index=genre_to_index,
        cache_dir=MEL_CACHE,
    )

    test_dataset = FMAMelDataset(
        manifest_path=TEST_MANIFEST,
        project_root=PROJECT_ROOT,
        genre_to_index=genre_to_index,
        cache_dir=MEL_CACHE,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNNBiLSTMAttention(
        num_classes=NUM_CLASSES,
        lstm_hidden=128,
        lstm_layers=2,
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: {parameter_count:,}"
    )

    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_df
    ).to(DEVICE)

    print("\nClass weights:")

    for genre, weight in zip(
        GENRES,
        class_weights.cpu().numpy(),
    ):
        print(
            f"  {genre}: {weight:.4f}"
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_macro_f1 = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    checkpoint_path = (
        CHECKPOINT_DIR
        / "cnn_bilstm_attention.pt"
    )

    print("\nStarting training...\n")

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        y_true = []
        y_pred = []

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{MAX_EPOCHS}",
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

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(mel)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            y_pred.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        train_metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            "Validation",
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        epoch_time = (
            time.time()
            - epoch_start
        )

        history_row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_metrics[
                "accuracy"
            ],
            "train_balanced_accuracy": train_metrics[
                "balanced_accuracy"
            ],
            "train_macro_f1": train_metrics[
                "macro_f1"
            ],
            "train_weighted_f1": train_metrics[
                "weighted_f1"
            ],
            "val_loss": val_metrics[
                "loss"
            ],
            "val_accuracy": val_metrics[
                "accuracy"
            ],
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
            "val_macro_f1": val_metrics[
                "macro_f1"
            ],
            "val_weighted_f1": val_metrics[
                "weighted_f1"
            ],
            "epoch_time_seconds": epoch_time,
        }

        history.append(
            history_row
        )

        print(
            f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        )
        print(
            f"  Train Loss: "
            f"{train_loss:.4f}"
        )
        print(
            f"  Train Accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Train Balanced Accuracy: "
            f"{train_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"  Train Macro-F1: "
            f"{train_metrics['macro_f1']:.4f}"
        )
        print(
            f"  Train Weighted-F1: "
            f"{train_metrics['weighted_f1']:.4f}"
        )

        print(
            f"  Val Loss: "
            f"{val_metrics['loss']:.4f}"
        )
        print(
            f"  Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val Balanced Accuracy: "
            f"{val_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"  Val Macro-F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )
        print(
            f"  Val Weighted-F1: "
            f"{val_metrics['weighted_f1']:.4f}"
        )

        print(
            f"  Learning Rate: "
            f"{current_lr:.6f}"
        )
        print(
            f"  Epoch Time: "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

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
                    "genres": GENRES,
                    "parameter_count":
                        parameter_count,
                },
                checkpoint_path,
            )

            print(
                f"  ✓ New best model "
                f"(Val Macro-F1: "
                f"{best_val_macro_f1:.4f})"
            )

        else:
            epochs_without_improvement += 1

            print(
                f"  No improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        METRICS_DIR
        / "cnn_bilstm_attention_training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print(
        f"\nLoading best model from epoch "
        f"{best_epoch}..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print("\nFinal evaluation...\n")

    train_metrics, _, _ = evaluate(
        model,
        train_loader,
        criterion,
        "Train",
    )

    val_metrics, _, _ = evaluate(
        model,
        val_loader,
        criterion,
        "Validation",
    )

    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        "Test",
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"\nBest Epoch: {best_epoch}"
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

    # --------------------------------------------------------
    # Per-class classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=GENRES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        METRICS_DIR
        / "cnn_bilstm_attention_test_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    # --------------------------------------------------------
    # Save final summary
    # --------------------------------------------------------

    metrics_row = {
        "model": "CNN + BiLSTM + Attention",
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,

        "train_loss": train_metrics[
            "loss"
        ],
        "train_accuracy": train_metrics[
            "accuracy"
        ],
        "train_balanced_accuracy":
            train_metrics[
                "balanced_accuracy"
            ],
        "train_macro_f1": train_metrics[
            "macro_f1"
        ],
        "train_weighted_f1": train_metrics[
            "weighted_f1"
        ],

        "val_loss": val_metrics[
            "loss"
        ],
        "val_accuracy": val_metrics[
            "accuracy"
        ],
        "val_balanced_accuracy":
            val_metrics[
                "balanced_accuracy"
            ],
        "val_macro_f1": val_metrics[
            "macro_f1"
        ],
        "val_weighted_f1": val_metrics[
            "weighted_f1"
        ],

        "test_loss": test_metrics[
            "loss"
        ],
        "test_accuracy": test_metrics[
            "accuracy"
        ],
        "test_balanced_accuracy":
            test_metrics[
                "balanced_accuracy"
            ],
        "test_macro_f1": test_metrics[
            "macro_f1"
        ],
        "test_weighted_f1": test_metrics[
            "weighted_f1"
        ],
    }

    metrics_path = (
        METRICS_DIR
        / "cnn_bilstm_attention.csv"
    )

    pd.DataFrame(
        [metrics_row]
    ).to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    print(
        f"\nTotal runtime: "
        f"{total_time / 60:.2f} minutes"
    )

    print("\nSaved:")
    print(
        f"  Checkpoint: "
        f"{checkpoint_path}"
    )
    print(
        f"  Metrics: "
        f"{metrics_path}"
    )
    print(
        f"  History: "
        f"{history_path}"
    )
    print(
        f"  Classification report: "
        f"{report_path}"
    )

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()