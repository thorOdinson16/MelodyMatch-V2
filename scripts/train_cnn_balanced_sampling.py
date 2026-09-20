from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melodymatch.data.dataset import FMAMelDataset
from melodymatch.data.labels import GENRE_TO_INDEX
from melodymatch.models.cnn import CNNBaseline


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
LR_PATIENCE = 2

NUM_CLASSES = 16


# ============================================================
# Paths
# ============================================================

TRAIN_MANIFEST = (
    PROJECT_ROOT / "data" / "clean_train.csv"
)

VAL_MANIFEST = (
    PROJECT_ROOT / "data" / "clean_validation.csv"
)

TEST_MANIFEST = (
    PROJECT_ROOT / "data" / "clean_test.csv"
)

CACHE_DIR = (
    PROJECT_ROOT / "data" / "mel_cache"
)

METRICS_DIR = (
    PROJECT_ROOT / "results" / "metrics"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT / "results" / "checkpoints"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Balanced sampler
# ============================================================

def create_balanced_sampler(dataset):
    """
    Give each training example a sampling probability
    inversely proportional to its class frequency.

    Rare classes are sampled more frequently.
    Common classes are sampled less frequently.

    The sampler draws len(dataset) samples per epoch,
    with replacement.
    """

    labels = dataset.df["genre"].map(
        GENRE_TO_INDEX
    ).to_numpy()

    class_counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    class_weights = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    for class_index in range(NUM_CLASSES):

        if class_counts[class_index] > 0:
            class_weights[class_index] = (
                1.0 /
                class_counts[class_index]
            )

    sample_weights = class_weights[labels]

    sample_weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return sampler, class_counts


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
    description="Evaluating"
):

    model.eval()

    running_loss = 0.0

    y_true = []
    y_pred = []

    progress = tqdm(
        loader,
        desc=description,
        leave=False
    )

    with torch.no_grad():

        for batch in progress:

            mel = batch["mel"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )

            logits = model(mel)

            loss = criterion(
                logits,
                labels
            )

            running_loss += (
                loss.item() *
                mel.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    avg_loss = (
        running_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics["loss"] = avg_loss

    return (
        metrics,
        y_true,
        y_pred
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for batch in progress:

        mel = batch["mel"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(mel)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            mel.size(0)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        running_loss /
        len(loader.dataset)
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("CNN + BALANCED SAMPLING")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    train_dataset = FMAMelDataset(
        TRAIN_MANIFEST,
        PROJECT_ROOT,
        GENRE_TO_INDEX,
        CACHE_DIR
    )

    val_dataset = FMAMelDataset(
        VAL_MANIFEST,
        PROJECT_ROOT,
        GENRE_TO_INDEX,
        CACHE_DIR
    )

    test_dataset = FMAMelDataset(
        TEST_MANIFEST,
        PROJECT_ROOT,
        GENRE_TO_INDEX,
        CACHE_DIR
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Balanced sampler
    # --------------------------------------------------------

    train_sampler, class_counts = (
        create_balanced_sampler(
            train_dataset
        )
    )

    print(
        "\nTraining class counts:"
    )

    index_to_genre = {
        index: genre
        for genre, index
        in GENRE_TO_INDEX.items()
    }

    for class_index in range(
        NUM_CLASSES
    ):

        print(
            f"  "
            f"{index_to_genre[class_index]:<22}"
            f"{class_counts[class_index]:>6}"
        )

    print(
        "\nUsing WeightedRandomSampler:"
    )

    print(
        "  Sampling: inverse class frequency"
    )

    print(
        "  Replacement: True"
    )

    print(
        f"  Samples per epoch: "
        f"{len(train_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNNBaseline(
        num_classes=NUM_CLASSES
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nParameters: "
        f"{total_params:,}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Normal CrossEntropy — NO class weights.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    print(
        "Loss: standard CrossEntropyLoss"
    )

    print(
        "Class weights: NONE"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=LR_PATIENCE
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_macro_f1 = -float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    checkpoint_path = (
        CHECKPOINT_DIR /
        "cnn_balanced_sampling.pt"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print(
            f"\nEpoch "
            f"{epoch}/{MAX_EPOCHS}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            description="Validation"
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "
            f"Acc: "
            f"{val_metrics['accuracy']:.4f} | "
            f"Bal Acc: "
            f"{val_metrics['balanced_accuracy']:.4f} | "
            f"Macro-F1: "
            f"{val_metrics['macro_f1']:.4f} | "
            f"Weighted-F1: "
            f"{val_metrics['weighted_f1']:.4f} | "
            f"LR: "
            f"{current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy":
                val_metrics["accuracy"],
            "val_balanced_accuracy":
                val_metrics["balanced_accuracy"],
            "val_macro_f1":
                val_metrics["macro_f1"],
            "val_weighted_f1":
                val_metrics["weighted_f1"],
            "learning_rate": current_lr,
        })

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
                    "sampling":
                        "inverse_frequency",
                    "loss":
                        "CrossEntropyLoss",
                },
                checkpoint_path
            )

            print(
                f"Saved best model "
                f"(Val Macro-F1: "
                f"{best_val_macro_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch}"
            )

            break

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"\nBest epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print(
        "\nFinal evaluation..."
    )

    train_metrics, _, _ = evaluate(
        model,
        train_loader,
        criterion,
        device,
        description="Train evaluation"
    )

    val_metrics, _, _ = evaluate(
        model,
        val_loader,
        criterion,
        device,
        description="Validation evaluation"
    )

    test_metrics, test_true, test_pred = (
        evaluate(
            model,
            test_loader,
            criterion,
            device,
            description="Test evaluation"
        )
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print(
        "\nFinal Results"
    )

    for split_name, metrics in [
        ("Train", train_metrics),
        ("Validation", val_metrics),
        ("Test", test_metrics),
    ]:

        print(
            f"{split_name}: "
            f"Loss={metrics['loss']:.4f}, "
            f"Accuracy="
            f"{metrics['accuracy']:.4f}, "
            f"Balanced Accuracy="
            f"{metrics['balanced_accuracy']:.4f}, "
            f"Macro-F1="
            f"{metrics['macro_f1']:.4f}, "
            f"Weighted-F1="
            f"{metrics['weighted_f1']:.4f}"
        )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    rows = []

    for split_name, metrics in [
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics),
    ]:

        rows.append({
            "model":
                "CNN + Balanced Sampling",
            "split": split_name,
            "loss": metrics["loss"],
            "accuracy":
                metrics["accuracy"],
            "balanced_accuracy":
                metrics["balanced_accuracy"],
            "macro_f1":
                metrics["macro_f1"],
            "weighted_f1":
                metrics["weighted_f1"],
        })

    metrics_path = (
        METRICS_DIR /
        "cnn_balanced_sampling.csv"
    )

    pd.DataFrame(rows).to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        METRICS_DIR /
        "cnn_balanced_sampling_training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    target_names = [
        index_to_genre[i]
        for i in range(NUM_CLASSES)
    ]

    report = classification_report(
        test_true,
        test_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        METRICS_DIR /
        "cnn_balanced_sampling_test_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print(
        f"\nSaved metrics to: "
        f"{metrics_path}"
    )

    print(
        f"Saved history to: "
        f"{history_path}"
    )

    print(
        f"Saved classification report to: "
        f"{report_path}"
    )

    print(
        f"Saved checkpoint to: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()