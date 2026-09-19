from pathlib import Path
import sys

import joblib
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

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

TRAIN_MANIFEST = (
    PROJECT_ROOT / "data" / "clean_train.csv"
)

VAL_MANIFEST = (
    PROJECT_ROOT / "data" / "clean_validation.csv"
)

TEST_MANIFEST = (
    PROJECT_ROOT / "data" / "clean_test.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "fma_metadata"
    / "tracks.csv"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "mel_cache"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------

from melodymatch.data.labels import (
    GENRE_TO_INDEX,
)

from melodymatch.data.metadata import (
    FMAMetadataProcessor,
)

from melodymatch.data.metadata_dataset import (
    FMAMelMetadataDataset,
)

from melodymatch.models.cnn_transformer_metadata import (
    CNNTransformerMetadata,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

NUM_CLASSES = 16

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def print_dataset_info(
    name,
    dataset,
):
    print(
        f"{name}: {len(dataset)} samples"
    )


def compute_class_weights(
    manifest_path,
):
    df = pd.read_csv(manifest_path)

    counts = (
        df["genre"]
        .map(GENRE_TO_INDEX)
        .value_counts()
        .sort_index()
    )

    counts = counts.reindex(
        range(NUM_CLASSES),
        fill_value=0,
    )

    total = counts.sum()

    weights = (
        total
        / (
            NUM_CLASSES
            * counts.replace(0, np.nan)
        )
    )

    weights = weights.fillna(0.0)

    return torch.tensor(
        weights.values,
        dtype=torch.float32,
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
    split_name,
):
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    progress = tqdm(
        loader,
        desc=f"{split_name} evaluation",
        leave=False,
    )

    with torch.no_grad():

        for batch in progress:

            mel = batch["mel"].to(
                device,
                non_blocking=True,
            )

            metadata = batch[
                "metadata"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            logits = model(
                mel,
                metadata,
            )

            loss = criterion(
                logits,
                labels,
            )

            predictions = (
                logits.argmax(dim=1)
            )

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    targets = np.array(
        all_targets
    )

    predictions = np.array(
        all_predictions
    )

    loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            targets,
            predictions,
        )
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(
        f"{split_name}: "
        f"loss={loss:.4f}, "
        f"acc={accuracy:.4f}, "
        f"bal_acc={balanced_accuracy:.4f}, "
        f"macro_f1={macro_f1:.4f}, "
        f"weighted_f1={weighted_f1:.4f}"
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "targets": targets,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------
# Metadata preparation
# ---------------------------------------------------------------------

def prepare_metadata_features():

    print("\nLoading FMA metadata...")

    metadata_processor = (
        FMAMetadataProcessor()
    )

    metadata_df = (
        metadata_processor.load_metadata(
            METADATA_PATH
        )
    )

    print(
        f"Metadata rows: "
        f"{len(metadata_df)}"
    )

    # -------------------------------------------------------------
    # Load manifests
    # -------------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_MANIFEST
    )

    val_df = pd.read_csv(
        VAL_MANIFEST
    )

    test_df = pd.read_csv(
        TEST_MANIFEST
    )

    # -------------------------------------------------------------
    # Metadata is indexed by track_id in
    # tracks.csv.
    # -------------------------------------------------------------

    train_ids = (
        train_df["track_id"]
        .astype(int)
        .tolist()
    )

    val_ids = (
        val_df["track_id"]
        .astype(int)
        .tolist()
    )

    test_ids = (
        test_df["track_id"]
        .astype(int)
        .tolist()
    )

    # -------------------------------------------------------------
    # Select metadata for each split.
    # -------------------------------------------------------------

    train_metadata = metadata_df.loc[
        train_ids
    ]

    val_metadata = metadata_df.loc[
        val_ids
    ]

    test_metadata = metadata_df.loc[
        test_ids
    ]

    # -------------------------------------------------------------
    # IMPORTANT:
    # Fit preprocessing ONLY on training metadata.
    # -------------------------------------------------------------

    print(
        "\nFitting metadata preprocessing "
        "on training split..."
    )

    metadata_processor.fit(
        train_metadata
    )

    train_features = (
        metadata_processor.transform(
            train_metadata
        )
    )

    val_features = (
        metadata_processor.transform(
            val_metadata
        )
    )

    test_features = (
        metadata_processor.transform(
            test_metadata
        )
    )

    print(
        f"Metadata feature dimension: "
        f"{metadata_processor.output_dim}"
    )

    print(
        f"Train metadata shape: "
        f"{train_features.shape}"
    )

    print(
        f"Validation metadata shape: "
        f"{val_features.shape}"
    )

    print(
        f"Test metadata shape: "
        f"{test_features.shape}"
    )

    # -------------------------------------------------------------
    # Save fitted processor for future inference.
    # -------------------------------------------------------------

    processor_path = (
        CHECKPOINT_DIR
        / "metadata_processor.joblib"
    )

    joblib.dump(
        metadata_processor,
        processor_path,
    )

    print(
        f"Saved metadata processor: "
        f"{processor_path}"
    )

    return (
        train_features,
        val_features,
        test_features,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "CNN + Transformer + Metadata"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    (
        train_metadata,
        val_metadata,
        test_metadata,
    ) = prepare_metadata_features()

    metadata_dim = (
        train_metadata.shape[1]
    )

    # -------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------

    print("\nCreating datasets...")

    train_dataset = (
        FMAMelMetadataDataset(
            manifest_path=TRAIN_MANIFEST,
            project_root=PROJECT_ROOT,
            genre_to_index=GENRE_TO_INDEX,
            metadata_features=train_metadata,
            cache_dir=CACHE_DIR,
        )
    )

    val_dataset = (
        FMAMelMetadataDataset(
            manifest_path=VAL_MANIFEST,
            project_root=PROJECT_ROOT,
            genre_to_index=GENRE_TO_INDEX,
            metadata_features=val_metadata,
            cache_dir=CACHE_DIR,
        )
    )

    test_dataset = (
        FMAMelMetadataDataset(
            manifest_path=TEST_MANIFEST,
            project_root=PROJECT_ROOT,
            genre_to_index=GENRE_TO_INDEX,
            metadata_features=test_metadata,
            cache_dir=CACHE_DIR,
        )
    )

    print_dataset_info(
        "Train",
        train_dataset,
    )

    print_dataset_info(
        "Validation",
        val_dataset,
    )

    print_dataset_info(
        "Test",
        test_dataset,
    )

    # -------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Class weights
    # -------------------------------------------------------------

    class_weights = (
        compute_class_weights(
            TRAIN_MANIFEST
        )
    )

    print(
        "\nClass weights:"
    )

    print(class_weights)

    class_weights = class_weights.to(
        DEVICE
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = (
        CNNTransformerMetadata(
            metadata_dim=metadata_dim,
            num_classes=NUM_CLASSES,
        )
        .to(DEVICE)
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{parameter_count:,}"
    )

    # -------------------------------------------------------------
    # Loss / optimizer / scheduler
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    history = []

    best_val_macro_f1 = -float(
        "inf"
    )

    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / "cnn_transformer_metadata.pt"
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        print(
            f"\nEpoch "
            f"{epoch}/{MAX_EPOCHS}"
        )

        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------

        model.train()

        running_loss = 0.0
        train_targets = []
        train_predictions = []

        progress = tqdm(
            train_loader,
            desc="Training",
            leave=True,
        )

        for batch in progress:

            mel = batch["mel"].to(
                DEVICE,
                non_blocking=True,
            )

            metadata = batch[
                "metadata"
            ].to(
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

            logits = model(
                mel,
                metadata,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            predictions = (
                logits.argmax(dim=1)
            )

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            train_targets.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            train_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_targets = np.array(
            train_targets
        )

        train_predictions = np.array(
            train_predictions
        )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        train_accuracy = accuracy_score(
            train_targets,
            train_predictions,
        )

        train_balanced_accuracy = (
            balanced_accuracy_score(
                train_targets,
                train_predictions,
            )
        )

        train_macro_f1 = f1_score(
            train_targets,
            train_predictions,
            average="macro",
            zero_division=0,
        )

        train_weighted_f1 = f1_score(
            train_targets,
            train_predictions,
            average="weighted",
            zero_division=0,
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            "Validation",
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Train: "
            f"loss={train_loss:.4f}, "
            f"acc={train_accuracy:.4f}, "
            f"bal_acc={train_balanced_accuracy:.4f}, "
            f"macro_f1={train_macro_f1:.4f}, "
            f"weighted_f1={train_weighted_f1:.4f}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.6g}"
        )

        # ---------------------------------------------------------
        # History
        # ---------------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_balanced_accuracy": train_balanced_accuracy,
                "train_macro_f1": train_macro_f1,
                "train_weighted_f1": train_weighted_f1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics[
                    "balanced_accuracy"
                ],
                "val_macro_f1": val_metrics[
                    "macro_f1"
                ],
                "val_weighted_f1": val_metrics[
                    "weighted_f1"
                ],
            }
        )

        # ---------------------------------------------------------
        # Best checkpoint
        # ---------------------------------------------------------

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
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": best_val_macro_f1,
                    "metadata_dim": metadata_dim,
                    "num_classes": NUM_CLASSES,
                },
                checkpoint_path,
            )

            print(
                f"✓ Best checkpoint saved "
                f"(Val Macro-F1: "
                f"{best_val_macro_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # ---------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # -------------------------------------------------------------
    # Save training history
    # -------------------------------------------------------------

    history_path = (
        METRICS_DIR
        / "cnn_transformer_metadata_training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print(
        f"\nTraining history saved: "
        f"{history_path}"
    )

    # -------------------------------------------------------------
    # Load best checkpoint
    # -------------------------------------------------------------

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
        checkpoint[
            "model_state_dict"
        ]
    )

    # -------------------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------------------

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
        "Test",
    )

    # -------------------------------------------------------------
    # Save overall metrics
    # -------------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "model": (
                    "CNN + Transformer + Metadata"
                ),
                "best_epoch": best_epoch,
                "best_val_macro_f1": (
                    best_val_macro_f1
                ),
                "test_loss": (
                    test_metrics["loss"]
                ),
                "test_accuracy": (
                    test_metrics["accuracy"]
                ),
                "test_balanced_accuracy": (
                    test_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "test_macro_f1": (
                    test_metrics[
                        "macro_f1"
                    ]
                ),
                "test_weighted_f1": (
                    test_metrics[
                        "weighted_f1"
                    ]
                ),
                "trainable_parameters": (
                    parameter_count
                ),
                "metadata_dimension": (
                    metadata_dim
                ),
            }
        ]
    )

    metrics_path = (
        METRICS_DIR
        / "cnn_transformer_metadata.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"Metrics saved: "
        f"{metrics_path}"
    )

    # -------------------------------------------------------------
    # Per-class classification report
    # -------------------------------------------------------------

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
        test_metrics["targets"],
        test_metrics["predictions"],
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    report_path = (
        METRICS_DIR
        / "cnn_transformer_metadata_test_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print(
        f"Classification report saved: "
        f"{report_path}"
    )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "CNN + Transformer + Metadata COMPLETE"
    )
    print("=" * 70)

    print(
        f"Best validation Macro-F1: "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test Weighted-F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )


if __name__ == "__main__":
    main()