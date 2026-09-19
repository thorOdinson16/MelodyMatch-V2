from pathlib import Path

import pandas as pd
from tqdm import tqdm

from melodymatch.data.preprocessing import audio_to_mel


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "mel_cache"

SPLITS = [
    "clean_train.csv",
    "clean_validation.csv",
    "clean_test.csv",
]


# ============================================================
# Main preprocessing
# ============================================================

def preprocess_split(split_file: str) -> None:
    manifest_path = DATA_DIR / split_file
    df = pd.read_csv(manifest_path)

    processed = 0
    skipped = 0
    failed = 0

    print(f"\nProcessing: {split_file}")
    print(f"Tracks: {len(df):,}")

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_file,
        unit="track",
    ):
        track_id = int(row["track_id"])
        audio_path = PROJECT_ROOT / row["audio_path"]

        cache_path = CACHE_DIR / f"{track_id:06d}.pt"

        # ----------------------------------------------------
        # Skip existing cache
        # ----------------------------------------------------

        if cache_path.exists():
            skipped += 1
            continue

        try:
            mel = audio_to_mel(audio_path)

            cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            import torch

            torch.save(mel, cache_path)

            processed += 1

        except Exception as exc:
            failed += 1

            print(
                f"\nFailed track {track_id}: {exc}"
            )

    print(
        f"Completed {split_file} | "
        f"processed={processed:,}, "
        f"skipped={skipped:,}, "
        f"failed={failed:,}"
    )


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_file in SPLITS:
        preprocess_split(split_file)

    print("\nMel preprocessing complete.")


if __name__ == "__main__":
    main()