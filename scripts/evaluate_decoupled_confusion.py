from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from melodymatch.data.dataset import FMAMelDataset
from melodymatch.data.labels import GENRES, GENRE_TO_INDEX
from melodymatch.models.cnn import CNNBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_MANIFEST = PROJECT_ROOT / "data" / "clean_test.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "mel_cache"
CHECKPOINT = PROJECT_ROOT / "results" / "checkpoints" / "cnn_decoupled.pt"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "cnn_decoupled_test_confusion_matrix.csv"
)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DECOUPLED CNN — CONFUSION MATRIX")
    print("=" * 70)
    print(f"Device: {device}")

    # Dataset
    dataset = FMAMelDataset(
        manifest_path=TEST_MANIFEST,
        project_root=PROJECT_ROOT,
        genre_to_index=GENRE_TO_INDEX,
        cache_dir=CACHE_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Test samples: {len(dataset)}")

    # Model
    model = CNNBaseline(num_classes=len(GENRES))

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Checkpoint loaded.")

    # Inference
    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch in tqdm(
            loader,
            desc="Evaluating decoupled CNN",
            unit="batch",
        ):

            mel = batch["mel"].to(device, non_blocking=True)
            labels = batch["label"]

            logits = model(mel)
            predictions = torch.argmax(logits, dim=1).cpu()

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(GENRES))),
    )

    cm_df = pd.DataFrame(
        cm,
        index=GENRES,
        columns=GENRES,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(OUTPUT_PATH)

    print()
    print("CONFUSION MATRIX")
    print("-" * 70)
    print(cm_df.to_string())

    print()
    print(f"Saved to:")
    print(OUTPUT_PATH)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()