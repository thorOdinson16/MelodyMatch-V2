from pathlib import Path

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

from melodymatch.data.dataset import FMAMelDataset
from melodymatch.data.labels import GENRE_TO_INDEX, GENRES
from melodymatch.models.cnn import CNNBaseline
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "mel_cache"

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
PATIENCE = 7


# ============================================================
# Dataset
# ============================================================

def create_dataset(split):

    return FMAMelDataset(
        manifest_path=DATA_DIR / f"clean_{split}.csv",
        project_root=PROJECT_ROOT,
        genre_to_index=GENRE_TO_INDEX,
        cache_dir=CACHE_DIR,
    )


# ============================================================
# Class weights
# ============================================================

def compute_class_weights(dataset):

    labels = [
        GENRE_TO_INDEX[genre]
        for genre in dataset.df["genre"]
    ]

    counts = np.bincount(
        labels,
        minlength=len(GENRES),
    )

    weights = (
        len(labels)
        / (
            len(GENRES)
            * counts
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    split_name,
):

    model.eval()

    all_predictions = []
    all_targets = []

    total_loss = 0.0

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
                * mel.size(0)
            )

            predictions = (
                logits.argmax(dim=1)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

    predictions = np.array(
        all_predictions
    )

    targets = np.array(
        all_targets
    )

    metrics = {
        "split": split_name,
        "loss": (
            total_loss
            / len(loader.dataset)
        ),
        "accuracy": accuracy_score(
            targets,
            predictions,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                targets,
                predictions,
            )
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics, targets, predictions


# ============================================================
# Main
# ============================================================

def main():

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = create_dataset(
        "train"
    )

    val_dataset = create_dataset(
        "validation"
    )

    test_dataset = create_dataset(
        "test"
    )

    print(
        "Train:",
        len(train_dataset),
    )

    print(
        "Validation:",
        len(val_dataset),
    )

    print(
        "Test:",
        len(test_dataset),
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
    # Class weights
    # --------------------------------------------------------

    class_weights = compute_class_weights(
        train_dataset
    )

    print("\nClass weights:")

    for genre, weight in zip(
        GENRES,
        class_weights.cpu().numpy(),
    ):
        print(
            f"{genre:22s}: {weight:.4f}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNNBaseline(
        num_classes=len(GENRES)
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: "
        f"{parameters:,}"
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "cnn_baseline.pt"
    )

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch:02d}/"
                f"{MAX_EPOCHS}"
            ),
            unit="batch",
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

            logits = model(mel)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * mel.size(0)
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            "validation",
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nEpoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics,
                "lr": current_lr,
            }
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "best_val_macro_f1": (
                        best_macro_f1
                    ),
                    "genres": GENRES,
                    "preprocessing": {
                        "sample_rate": 22050,
                        "duration_seconds": 30.0,
                        "n_mels": 128,
                        "n_fft": 2048,
                        "hop_length": 512,
                        "fmin": 20,
                        "fmax": 11025,
                        "normalization": (
                            "per_spectrogram_zscore"
                        ),
                    },
                },
                checkpoint_path,
            )

            print(
                f"  ✓ New best checkpoint "
                f"(Macro-F1={best_macro_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping at "
                f"epoch {epoch}."
            )

            break

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print(
        f"\nLoading best checkpoint "
        f"from epoch {best_epoch}..."
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

    all_results = []

    for split_name, loader in [
        ("train", train_loader),
        ("validation", val_loader),
        ("test", test_loader),
    ]:

        metrics, targets, predictions = (
            evaluate(
                model,
                loader,
                criterion,
                split_name,
            )
        )

        all_results.append(
            metrics
        )

        print(
            f"\n{split_name.upper()} RESULTS"
        )

        print("-" * 50)

        print(
            f"Loss: "
            f"{metrics['loss']:.4f}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro-F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Weighted-F1: "
            f"{metrics['weighted_f1']:.4f}"
        )

        if split_name == "test":

            print(
                "\nPer-class report:"
            )

            print(
                classification_report(
                    targets,
                    predictions,
                    target_names=GENRES,
                    zero_division=0,
                )
            )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        all_results
    )

    metrics_path = (
        RESULTS_DIR
        / "cnn_baseline.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    history_path = (
        RESULTS_DIR
        / "cnn_training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print(
        f"\nMetrics saved to: "
        f"{metrics_path}"
    )

    print(
        f"Training history saved to: "
        f"{history_path}"
    )

    print(
        f"Best checkpoint saved to: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()