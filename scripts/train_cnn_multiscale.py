from pathlib import Path
import copy
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
from melodymatch.data.labels import GENRE_TO_INDEX


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "data" / "clean_train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "clean_validation.csv"
TEST_CSV = PROJECT_ROOT / "data" / "clean_test.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "mel_cache"

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "cnn_multiscale.pt"

BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN for Mel spectrograms.

    Parallel kernels capture different local
    time-frequency patterns before the features are merged.
    """

    def __init__(self, num_classes=16):
        super().__init__()

        # First multi-scale block.
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.branch7 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool2d(2)

        # 48 -> 64 channels.
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 64 -> 128.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 128 -> 256.
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)

        x = torch.cat([b3, b5, b7], dim=1)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = self.classifier(x)

        return x


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


def make_loader(csv_path, shuffle=False):
    dataset = FMAMelDataset(
        manifest_path=csv_path,
        project_root=PROJECT_ROOT,
        genre_to_index=GENRE_TO_INDEX,
        cache_dir=CACHE_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, loader


def get_class_weights(dataset):
    labels = dataset.df["genre"].map(GENRE_TO_INDEX).to_numpy()

    counts = np.bincount(
        labels,
        minlength=len(GENRE_TO_INDEX),
    )

    total = len(labels)

    weights = total / (len(counts) * np.maximum(counts, 1))

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


def main():
    start_time = time.time()

    print("=" * 70)
    print("MULTI-SCALE CNN")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading datasets...")

    train_dataset, train_loader = make_loader(
        TRAIN_CSV,
        shuffle=True,
    )

    _, val_loader = make_loader(VAL_CSV)
    _, test_loader = make_loader(TEST_CSV)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = MultiScaleCNN(
        num_classes=len(GENRE_TO_INDEX)
    ).to(DEVICE)

    num_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {num_params:,}")

    class_weights = get_class_weights(train_dataset)

    print("\nLoss: class-weighted CrossEntropyLoss")
    print("Architecture: 3x3 + 5x5 + 7x7 parallel branches")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
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
            desc=f"Epoch {epoch}/{MAX_EPOCHS}",
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

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = running_loss / total

        val_result = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_result["macro_f1"]
        )

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_result['loss']:.4f} | "
            f"Acc={val_result['accuracy']:.4f} | "
            f"Bal Acc={val_result['balanced_accuracy']:.4f} | "
            f"Macro-F1={val_result['macro_f1']:.4f} | "
            f"Weighted-F1={val_result['weighted_f1']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        history.append({
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
                },
                CHECKPOINT_PATH,
            )

            print(
                f"Saved best model "
                f"(Val Macro-F1: {best_macro_f1:.4f})"
            )

        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}")
    print(
        f"Best validation Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print("\nFinal evaluation...")

    train_eval = evaluate(
        model,
        train_loader,
        criterion,
    )

    val_eval = evaluate(
        model,
        val_loader,
        criterion,
    )

    test_eval = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\nFinal Results")

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

    metrics = pd.DataFrame([
        {
            "model": "Multi-Scale CNN",
            "split": "train",
            "loss": train_eval["loss"],
            "accuracy": train_eval["accuracy"],
            "balanced_accuracy": train_eval["balanced_accuracy"],
            "macro_f1": train_eval["macro_f1"],
            "weighted_f1": train_eval["weighted_f1"],
        },
        {
            "model": "Multi-Scale CNN",
            "split": "validation",
            "loss": val_eval["loss"],
            "accuracy": val_eval["accuracy"],
            "balanced_accuracy": val_eval["balanced_accuracy"],
            "macro_f1": val_eval["macro_f1"],
            "weighted_f1": val_eval["weighted_f1"],
        },
        {
            "model": "Multi-Scale CNN",
            "split": "test",
            "loss": test_eval["loss"],
            "accuracy": test_eval["accuracy"],
            "balanced_accuracy": test_eval["balanced_accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "weighted_f1": test_eval["weighted_f1"],
        },
    ])

    metrics.to_csv(
        METRICS_DIR / "cnn_multiscale.csv",
        index=False,
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        METRICS_DIR / "cnn_multiscale_training_history.csv",
        index=False,
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

    pd.DataFrame(report).transpose().to_csv(
        METRICS_DIR /
        "cnn_multiscale_test_classification_report.csv"
    )

    elapsed = (time.time() - start_time) / 60

    print("\nSaved:")
    print(f"  {METRICS_DIR / 'cnn_multiscale.csv'}")
    print(
        f"  {METRICS_DIR / 'cnn_multiscale_training_history.csv'}"
    )
    print(
        f"  {METRICS_DIR / 'cnn_multiscale_test_classification_report.csv'}"
    )
    print(f"  {CHECKPOINT_PATH}")
    print(f"\nRuntime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()