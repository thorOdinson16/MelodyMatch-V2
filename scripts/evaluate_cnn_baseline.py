from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from melodymatch.data.dataset import FMAMelDataset
from melodymatch.data.labels import GENRES, GENRE_TO_INDEX
from melodymatch.models.cnn import CNNBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_MANIFEST = PROJECT_ROOT / "data" / "clean_test.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "mel_cache"

CHECKPOINT = PROJECT_ROOT / "results" / "checkpoints" / "cnn_baseline.pt"

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "cnn_baseline_test_classification_report.csv"
)

CONFUSION_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "cnn_baseline_test_confusion_matrix.csv"
)


def main():
    print("=" * 70)
    print("CNN BASELINE TEST EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    dataset = FMAMelDataset(
        manifest_path=TEST_MANIFEST,
        project_root=PROJECT_ROOT,
        genre_to_index=GENRE_TO_INDEX,
        cache_dir=CACHE_DIR,
    )

    print(f"Test samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = CNNBaseline(num_classes=len(GENRES))

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    # Handle either a raw state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Checkpoint loaded.")

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Evaluating CNN baseline",
            unit="batch",
        ):
            mel = batch["mel"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(mel)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions,
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

    print()
    print("TEST RESULTS")
    print("-" * 50)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(GENRES))),
        target_names=GENRES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH)

    print()
    print("Per-class results:")
    print(
        report_df[
            ["precision", "recall", "f1-score", "support"]
        ].to_string()
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(GENRES))),
    )

    cm_df = pd.DataFrame(
        cm,
        index=GENRES,
        columns=GENRES,
    )

    cm_df.to_csv(CONFUSION_PATH)

    print()
    print(f"Classification report saved to:")
    print(REPORT_PATH)

    print()
    print(f"Confusion matrix saved to:")
    print(CONFUSION_PATH)

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()