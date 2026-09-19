from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

MIXUP_ALPHA = 0.4
MIXUP_PROBABILITY = 0.5

NUM_CLASSES = 16


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
# Paths
# ============================================================

TRAIN_MANIFEST = PROJECT_ROOT / "data" / "clean_train.csv"
VAL_MANIFEST = PROJECT_ROOT / "data" / "clean_validation.csv"
TEST_MANIFEST = PROJECT_ROOT / "data" / "clean_test.csv"

CACHE_DIR = PROJECT_ROOT / "data" / "mel_cache"

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Class weights
# ============================================================

def compute_class_weights(manifest_path):
    df = pd.read_csv(manifest_path)

    counts = (
        df["genre"]
        .map(GENRE_TO_INDEX)
        .value_counts()
        .sort_index()
    )

    counts = counts.to_numpy(dtype=np.float64)

    weights = len(df) / (NUM_CLASSES * counts)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# Mixup
# ============================================================

def mixup_batch(x, y, alpha=0.4):
    """
    Mix a batch of inputs and labels.

    x: [B, C, H, W]
    y: [B]
    """

    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)

    batch_size = x.size(0)

    index = torch.randperm(
        batch_size,
        device=x.device
    )

    mixed_x = lam * x + (1.0 - lam) * x[index]

    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
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

def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Evaluating",
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

            logits = model(mel)

            loss = criterion(
                logits,
                labels
            )

            running_loss += (
                loss.item() * mel.size(0)
            )

            predictions = logits.argmax(dim=1)

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    avg_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics["loss"] = avg_loss

    return metrics, y_true, y_pred


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

        # ----------------------------------------------------
        # Apply Mixup with probability 0.5
        # ----------------------------------------------------

        if random.random() < MIXUP_PROBABILITY:

            mixed_mel, y_a, y_b, lam = mixup_batch(
                mel,
                labels,
                MIXUP_ALPHA
            )

            logits = model(mixed_mel)

            loss = (
                lam * criterion(logits, y_a)
                + (1.0 - lam) * criterion(logits, y_b)
            )

        else:

            logits = model(mel)

            loss = criterion(
                logits,
                labels
            )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * mel.size(0)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / len(loader.dataset)


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

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    print(f"Parameters: {total_params:,}")

    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    class_weights = compute_class_weights(
        TRAIN_MANIFEST
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=LR_PATIENCE
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
        "cnn_mixup.pt"
    )

    for epoch in range(1, MAX_EPOCHS + 1):

        print(
            f"\nEpoch {epoch}/{MAX_EPOCHS}"
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
            device
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Bal Acc: {val_metrics['balanced_accuracy']:.4f} | "
            f"Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Weighted-F1: {val_metrics['weighted_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "learning_rate": current_lr,
        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_metrics["macro_f1"] > best_val_macro_f1:

            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_val_macro_f1,
                    "config": {
                        "mixup_alpha": MIXUP_ALPHA,
                        "mixup_probability": MIXUP_PROBABILITY,
                    },
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
                f"Early stopping at epoch {epoch}"
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
        f"\nBest epoch: {checkpoint['epoch']}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print("\nEvaluating final model...")

    train_metrics, _, _ = evaluate(
        model,
        train_loader,
        criterion,
        device
    )

    val_metrics, _, _ = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    test_metrics, test_true, test_pred = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nFinal Results")

    for split_name, metrics in [
        ("Train", train_metrics),
        ("Validation", val_metrics),
        ("Test", test_metrics),
    ]:

        print(
            f"{split_name}: "
            f"Loss={metrics['loss']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Balanced Accuracy={metrics['balanced_accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"Weighted-F1={metrics['weighted_f1']:.4f}"
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
            "model": "CNN + Mixup",
            "split": split_name,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })

    metrics_path = (
        METRICS_DIR /
        "cnn_mixup.csv"
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
        "cnn_mixup_training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # Save test classification report
    # --------------------------------------------------------

    index_to_genre = {
        index: genre
        for genre, index in GENRE_TO_INDEX.items()
    }

    target_names = [
        index_to_genre[i]
        for i in range(NUM_CLASSES)
    ]

    report = classification_report(
        test_true,
        test_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = (
        METRICS_DIR /
        "cnn_mixup_test_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print(
        f"\nSaved metrics to: {metrics_path}"
    )

    print(
        f"Saved history to: {history_path}"
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