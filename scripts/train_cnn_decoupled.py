from pathlib import Path
import copy
import json
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from melodymatch.data.dataset import FMAMelDataset
from melodymatch.data.labels import GENRE_TO_INDEX
from melodymatch.models.cnn import CNNBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "data" / "clean_train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "clean_validation.csv"
TEST_CSV = PROJECT_ROOT / "data" / "clean_test.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "mel_cache"

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_CHECKPOINT = CHECKPOINT_DIR / "cnn_decoupled_stage1.pt"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "cnn_decoupled.pt"

BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, criterion):
    model.eval()

    losses = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            logits = model(mel)
            loss = criterion(logits, labels)

            losses.append(loss.item() * labels.size(0))
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total = len(y_true)

    return {
        "loss": sum(losses) / total,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def make_loader(csv_path, shuffle=False, sampler=None):
    dataset = FMAMelDataset(
        manifest_path=csv_path,
        project_root=PROJECT_ROOT,
        genre_to_index=GENRE_TO_INDEX,
        cache_dir=CACHE_DIR,
    )

    return dataset, DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def build_balanced_sampler(dataset):
    labels = dataset.df["genre"].map(GENRE_TO_INDEX).to_numpy()

    class_counts = np.bincount(
        labels,
        minlength=len(GENRE_TO_INDEX),
    )

    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_stage1(model, train_loader, val_loader):
    """
    Stage 1:
    Standard CNN training on the original imbalanced dataset.
    No sampler and no class weighting.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_STAGE1,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Stage 1 Epoch {epoch}/{MAX_EPOCHS}",
            leave=True,
        )

        for batch in progress:
            mel = batch["mel"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(mel)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        val_result = evaluate(model, val_loader, criterion)

        scheduler.step(val_result["macro_f1"])

        print(
            f"Stage 1 Epoch {epoch}: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_result['loss']:.4f} | "
            f"Val Acc={val_result['accuracy']:.4f} | "
            f"Val Bal Acc={val_result['balanced_accuracy']:.4f} | "
            f"Val Macro-F1={val_result['macro_f1']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        history.append({
            "stage": 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "val_balanced_accuracy": val_result["balanced_accuracy"],
            "val_macro_f1": val_result["macro_f1"],
            "val_weighted_f1": val_result["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_result["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_result["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "stage": 1,
                },
                STAGE1_CHECKPOINT,
            )

            print(
                f"Saved Stage 1 checkpoint "
                f"(Val Macro-F1: {best_macro_f1:.4f})"
            )
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Stage 1 early stopping.")
            break

    model.load_state_dict(best_state)

    return model, history, best_epoch, best_macro_f1


def train_stage2(model, train_loader, val_loader):
    """
    Stage 2:
    Freeze CNN feature extractor.
    Train ONLY the final classifier using balanced sampling.
    """

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    # Reinitialize only the final Linear classifier.
    classifier_linear = model.classifier[-1]

    if not isinstance(classifier_linear, nn.Linear):
        raise RuntimeError(
            "Expected the final CNN classifier layer to be nn.Linear."
        )

    nn.init.xavier_uniform_(classifier_linear.weight)
    nn.init.zeros_(classifier_linear.bias)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR_STAGE2,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Stage 2 Epoch {epoch}/{MAX_EPOCHS}",
            leave=True,
        )

        for batch in progress:
            mel = batch["mel"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(mel)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        val_result = evaluate(model, val_loader, criterion)

        scheduler.step(val_result["macro_f1"])

        print(
            f"Stage 2 Epoch {epoch}: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_result['loss']:.4f} | "
            f"Val Acc={val_result['accuracy']:.4f} | "
            f"Val Bal Acc={val_result['balanced_accuracy']:.4f} | "
            f"Val Macro-F1={val_result['macro_f1']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        history.append({
            "stage": 2,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "val_balanced_accuracy": val_result["balanced_accuracy"],
            "val_macro_f1": val_result["macro_f1"],
            "val_weighted_f1": val_result["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_result["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_result["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "stage": 2,
                },
                FINAL_CHECKPOINT,
            )

            print(
                f"Saved Stage 2 checkpoint "
                f"(Val Macro-F1: {best_macro_f1:.4f})"
            )
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Stage 2 early stopping.")
            break

    model.load_state_dict(best_state)

    return model, history, best_epoch, best_macro_f1


def main():
    start_time = time.time()

    print("=" * 70)
    print("CNN + DECOUPLED TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading datasets...")

    train_dataset, train_loader_normal = make_loader(
        TRAIN_CSV,
        shuffle=True,
    )

    _, val_loader = make_loader(VAL_CSV)
    _, test_loader = make_loader(TEST_CSV)

    balanced_sampler = build_balanced_sampler(train_dataset)

    _, train_loader_balanced = make_loader(
        TRAIN_CSV,
        sampler=balanced_sampler,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = CNNBaseline(
        num_classes=len(GENRE_TO_INDEX)
    ).to(DEVICE)

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ---------------------------------------------------------------
    # STAGE 1
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 1 — REPRESENTATION LEARNING")
    print("=" * 70)
    print("Standard CrossEntropyLoss")
    print("Original imbalanced training distribution")
    print("All CNN parameters trainable")

    model, history1, best_epoch1, best_val1 = train_stage1(
        model,
        train_loader_normal,
        val_loader,
    )

    # ---------------------------------------------------------------
    # STAGE 2
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 2 — CLASSIFIER RE-BALANCING")
    print("=" * 70)
    print("CNN feature extractor: FROZEN")
    print("Classifier: TRAINABLE")
    print("WeightedRandomSampler: ENABLED")
    print("CrossEntropyLoss: UNWEIGHTED")

    model, history2, best_epoch2, best_val2 = train_stage2(
        model,
        train_loader_balanced,
        val_loader,
    )

    # ---------------------------------------------------------------
    # FINAL EVALUATION
    # ---------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    train_eval = evaluate(model, train_loader_normal, criterion)
    val_eval = evaluate(model, val_loader, criterion)
    test_eval = evaluate(model, test_loader, criterion)

    for name, result in [
        ("Train", train_eval),
        ("Validation", val_eval),
        ("Test", test_eval),
    ]:
        print(
            f"{name}: "
            f"Loss={result['loss']:.4f}, "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Balanced Accuracy={result['balanced_accuracy']:.4f}, "
            f"Macro-F1={result['macro_f1']:.4f}, "
            f"Weighted-F1={result['weighted_f1']:.4f}"
        )

    report = classification_report(
        test_eval["y_true"],
        test_eval["y_pred"],
        target_names=[
            genre
            for genre, _ in sorted(
                GENRE_TO_INDEX.items(),
                key=lambda x: x[1],
            )
        ],
        output_dict=True,
        zero_division=0,
    )

    metrics = pd.DataFrame([
        {
            "model": "CNN + Decoupled Training",
            "split": "train",
            "loss": train_eval["loss"],
            "accuracy": train_eval["accuracy"],
            "balanced_accuracy": train_eval["balanced_accuracy"],
            "macro_f1": train_eval["macro_f1"],
            "weighted_f1": train_eval["weighted_f1"],
        },
        {
            "model": "CNN + Decoupled Training",
            "split": "validation",
            "loss": val_eval["loss"],
            "accuracy": val_eval["accuracy"],
            "balanced_accuracy": val_eval["balanced_accuracy"],
            "macro_f1": val_eval["macro_f1"],
            "weighted_f1": val_eval["weighted_f1"],
        },
        {
            "model": "CNN + Decoupled Training",
            "split": "test",
            "loss": test_eval["loss"],
            "accuracy": test_eval["accuracy"],
            "balanced_accuracy": test_eval["balanced_accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "weighted_f1": test_eval["weighted_f1"],
        },
    ])

    metrics.to_csv(
        METRICS_DIR / "cnn_decoupled.csv",
        index=False,
    )

    history = pd.DataFrame(history1 + history2)
    history.to_csv(
        METRICS_DIR / "cnn_decoupled_training_history.csv",
        index=False,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        METRICS_DIR / "cnn_decoupled_test_classification_report.csv"
    )

    elapsed = (time.time() - start_time) / 60

    print("\nSaved:")
    print(f"  {METRICS_DIR / 'cnn_decoupled.csv'}")
    print(f"  {METRICS_DIR / 'cnn_decoupled_training_history.csv'}")
    print(f"  {METRICS_DIR / 'cnn_decoupled_test_classification_report.csv'}")
    print(f"  {FINAL_CHECKPOINT}")
    print(f"\nRuntime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()