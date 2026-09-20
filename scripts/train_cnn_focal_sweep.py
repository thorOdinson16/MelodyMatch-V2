from pathlib import Path
import sys
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

NUM_CLASSES = 16

# Overnight sweep
FOCAL_GAMMAS = [1.0, 2.0, 3.0]


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

    counts = counts.to_numpy(
        dtype=np.float64
    )

    weights = (
        len(df) /
        (NUM_CLASSES * counts)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        weight=None,
        gamma=2.0
    ):
        super().__init__()

        self.weight = weight
        self.gamma = gamma

    def forward(
        self,
        logits,
        targets
    ):

        log_probs = F.log_softmax(
            logits,
            dim=1
        )

        log_pt = log_probs.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        pt = log_pt.exp()

        ce_loss = F.nll_loss(
            log_probs,
            targets,
            weight=self.weight,
            reduction="none"
        )

        focal_factor = (
            1.0 - pt
        ).pow(self.gamma)

        loss = (
            focal_factor *
            ce_loss
        )

        return loss.mean()


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

            predictions = (
                logits.argmax(dim=1)
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
# Train one epoch
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
# Run one gamma experiment
# ============================================================

def run_experiment(
    gamma,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    device
):

    print("\n")
    print("=" * 70)
    print(
        f"FOCAL LOSS EXPERIMENT — gamma={gamma}"
    )
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    set_seed(SEED)

    model = CNNBaseline(
        num_classes=NUM_CLASSES
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {total_params:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = FocalLoss(
        weight=class_weights,
        gamma=gamma
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
    # Tracking
    # --------------------------------------------------------

    best_val_macro_f1 = -float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    gamma_string = str(gamma).replace(
        ".",
        "_"
    )

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"cnn_focal_gamma_{gamma_string}.pt"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print(
            f"\n[gamma={gamma}] "
            f"Epoch {epoch}/{MAX_EPOCHS}"
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
            "gamma": gamma,
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
        # Best checkpoint
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
                    "gamma": gamma,
                    "val_macro_f1":
                        best_val_macro_f1,
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
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{best_val_macro_f1:.4f}"
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
    # Print final results
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
    # Save individual metrics
    # --------------------------------------------------------

    metrics_rows = []

    for split_name, metrics in [
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics),
    ]:

        metrics_rows.append({
            "model":
                f"CNN + Focal Loss "
                f"(gamma={gamma})",
            "gamma": gamma,
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
        f"cnn_focal_gamma_{gamma_string}.csv"
    )

    pd.DataFrame(
        metrics_rows
    ).to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
        METRICS_DIR /
        f"cnn_focal_gamma_{gamma_string}"
        f"_training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    index_to_genre = {
        index: genre
        for genre, index
        in GENRE_TO_INDEX.items()
    }

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
        f"cnn_focal_gamma_{gamma_string}"
        f"_test_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    runtime_minutes = (
        time.time() - start_time
    ) / 60.0

    print(
        f"\nRuntime: "
        f"{runtime_minutes:.2f} minutes"
    )

    return {
        "gamma": gamma,
        "best_epoch": best_epoch,
        "best_val_macro_f1":
            best_val_macro_f1,
        "test_accuracy":
            test_metrics["accuracy"],
        "test_balanced_accuracy":
            test_metrics["balanced_accuracy"],
        "test_macro_f1":
            test_metrics["macro_f1"],
        "test_weighted_f1":
            test_metrics["weighted_f1"],
        "runtime_minutes":
            runtime_minutes,
    }


# ============================================================
# Main
# ============================================================

def main():

    overall_start = time.time()

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("CNN + FOCAL LOSS SWEEP")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Gamma values: "
        f"{FOCAL_GAMMAS}"
    )

    print(
        f"Train: "
        f"{TRAIN_MANIFEST}"
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
    # Class weights
    # --------------------------------------------------------

    class_weights = (
        compute_class_weights(
            TRAIN_MANIFEST
        ).to(device)
    )

    print(
        "\nClass weights loaded."
    )

    # --------------------------------------------------------
    # Run sweep
    # --------------------------------------------------------

    results = []

    for experiment_number, gamma in enumerate(
        FOCAL_GAMMAS,
        start=1
    ):

        print("\n")
        print("#" * 70)
        print(
            f"EXPERIMENT "
            f"{experiment_number}/"
            f"{len(FOCAL_GAMMAS)}"
        )
        print(
            f"Focal Loss gamma = {gamma}"
        )
        print("#" * 70)

        result = run_experiment(
            gamma,
            train_loader,
            val_loader,
            test_loader,
            class_weights,
            device
        )

        results.append(result)

        # ----------------------------------------------------
        # Save combined results after EVERY experiment
        # ----------------------------------------------------

        comparison_path = (
            METRICS_DIR /
            "cnn_focal_sweep_comparison.csv"
        )

        comparison_df = pd.DataFrame(
            results
        )

        comparison_df.to_csv(
            comparison_path,
            index=False
        )

        print(
            "\nCurrent sweep results:"
        )

        print(
            comparison_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    total_runtime = (
        time.time() - overall_start
    ) / 60.0

    comparison_path = (
        METRICS_DIR /
        "cnn_focal_sweep_comparison.csv"
    )

    comparison_df = pd.DataFrame(
        results
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FOCAL LOSS SWEEP COMPLETE")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print(
        f"\nTotal runtime: "
        f"{total_runtime:.2f} minutes"
    )

    print(
        f"\nSaved comparison to:"
        f"\n{comparison_path}"
    )


if __name__ == "__main__":
    main()