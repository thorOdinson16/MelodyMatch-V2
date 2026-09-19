from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR = PROJECT_ROOT / "results" / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GENRES = [
    "Blues",
    "Classical",
    "Country",
    "Easy Listening",
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Jazz",
    "Old-Time / Historic",
    "Pop",
    "Rock",
    "Soul-RnB",
    "Spoken",
]

GENRE_TO_INDEX = {
    genre: i for i, genre in enumerate(GENRES)
}


class MLPBaseline(nn.Module):

    def __init__(self, num_classes=16):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(518, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_split(name):

    df = pd.read_csv(
        DATA_DIR / f"clean_{name}_features.csv"
    )

    X = df.drop(
        columns=[
            "track_id",
            "genre",
            "audio_path",
            "split",
        ]
    )

    y = df["genre"].map(
        GENRE_TO_INDEX
    )

    return X.values.astype(np.float32), y.values


def evaluate(model, loader, criterion):

    model.eval()

    all_predictions = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)

            loss = criterion(
                logits,
                y,
            )

            total_loss += (
                loss.item() * X.size(0)
            )

            predictions = (
                logits.argmax(dim=1)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y.cpu().numpy()
            )

    predictions = np.array(
        all_predictions
    )

    targets = np.array(
        all_targets
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(
            targets,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            targets,
            predictions,
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
        "targets": targets,
        "predictions": predictions,
    }


def main():

    print("Device:", DEVICE)

    print("\nLoading datasets...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    print("\nFitting StandardScaler...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_counts = np.bincount(
        y_train,
        minlength=len(GENRES),
    )

    weights = (
        len(y_train)
        / (
            len(GENRES)
            * class_counts
        )
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long(),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).long(),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MLPBaseline(
        num_classes=len(GENRES)
    ).to(DEVICE)

    print(
        "\nParameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    epochs = 30

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{epochs}",
            leave=False,
        )

        for X, y in progress:

            X = X.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * X.size(0)
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if (
            val_metrics["macro_f1"]
            > best_macro_f1
        ):
            best_macro_f1 = (
                val_metrics["macro_f1"]
            )

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    results = {}

    for name, loader in [
        ("train", train_loader),
        ("validation", val_loader),
        ("test", test_loader),
    ]:

        metrics = evaluate(
            model,
            loader,
            criterion,
        )

        results[name] = metrics

        print(
            f"\n{name.upper()} RESULTS"
        )
        print("-" * 50)

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

        if name == "test":

            print("\nPer-class report:")

            print(
                classification_report(
                    metrics["targets"],
                    metrics["predictions"],
                    target_names=GENRES,
                    zero_division=0,
                )
            )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "split": name,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics[
                "balanced_accuracy"
            ],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }
        for name, metrics in results.items()
    ])

    metrics_path = (
        RESULTS_DIR
        / "mlp_baseline.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR
        / "mlp_baseline.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "genres": GENRES,
            "input_dim": 518,
            "num_classes": 16,
        },
        model_path,
    )

    print(
        f"\nMetrics saved to: {metrics_path}"
    )

    print(
        f"Model saved to: {model_path}"
    )


if __name__ == "__main__":
    main()