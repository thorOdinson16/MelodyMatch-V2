from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifest_with_splits.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "audio_validation.csv"
)

CLEAN_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean_manifest.csv"
)

CLEAN_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean_train.csv"
)

CLEAN_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean_validation.csv"
)

CLEAN_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean_test.csv"
)

REMOVED_PATH = (
    PROJECT_ROOT
    / "data"
    / "removed_tracks.csv"
)


# ============================================================
# Dataset cleaning rules
# ============================================================

MIN_DURATION_SECONDS = 10.0


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MelodyMatch - Final Dataset Cleaning")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load files
    # --------------------------------------------------------

    print("\n[1/5] Loading manifests...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(VALIDATION_PATH)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Original tracks: "
        f"{len(manifest):,}"
    )

    print(
        f"Validation records: "
        f"{len(validation):,}"
    )

    # --------------------------------------------------------
    # 2. Merge validation information
    # --------------------------------------------------------

    print("\n[2/5] Merging audio validation results...")

    validation_info = validation[
        [
            "track_id",
            "exists",
            "valid",
            "duration",
            "sample_rate",
            "channels",
            "error",
        ]
    ].copy()

    df = manifest.merge(
        validation_info,
        on="track_id",
        how="left",
        validate="one_to_one",
    )

    if df["valid"].isna().any():
        missing = df["valid"].isna().sum()

        raise RuntimeError(
            f"{missing} tracks have no validation record."
        )

    # --------------------------------------------------------
    # 3. Apply cleaning rules
    # --------------------------------------------------------

    print("\n[3/5] Applying cleaning rules...")

    removal_reasons = []

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Checking tracks",
        unit="track",
    ):

        reason = None

        # Rule 1: audio must be valid
        if not bool(row.valid):

            reason = "invalid_audio"

        # Rule 2: audio must be at least 10 seconds
        elif (
            pd.isna(row.duration)
            or row.duration < MIN_DURATION_SECONDS
        ):

            reason = "duration_under_10_seconds"

        removal_reasons.append(reason)

    df["removal_reason"] = removal_reasons

    # --------------------------------------------------------
    # Split into clean and removed
    # --------------------------------------------------------

    removed_df = df[
        df["removal_reason"].notna()
    ].copy()

    clean_df = df[
        df["removal_reason"].isna()
    ].copy()

    # --------------------------------------------------------
    # Remove validation-only columns from clean manifest
    # --------------------------------------------------------

    clean_columns = [
        "track_id",
        "genre",
        "audio_path",
        "split",
    ]

    clean_manifest = clean_df[
        clean_columns
    ].copy()

    # --------------------------------------------------------
    # 4. Save files
    # --------------------------------------------------------

    print("\n[4/5] Saving cleaned datasets...")

    clean_manifest.to_csv(
        CLEAN_MANIFEST_PATH,
        index=False,
    )

    split_files = {
        "train": CLEAN_TRAIN_PATH,
        "validation": CLEAN_VALIDATION_PATH,
        "test": CLEAN_TEST_PATH,
    }

    for split, output_path in tqdm(
        split_files.items(),
        desc="Saving splits",
        unit="split",
    ):

        split_df = clean_manifest[
            clean_manifest["split"] == split
        ]

        split_df.to_csv(
            output_path,
            index=False,
        )

    # Save removed tracks with reasons
    removed_columns = [
        "track_id",
        "genre",
        "split",
        "audio_path",
        "valid",
        "duration",
        "sample_rate",
        "channels",
        "removal_reason",
    ]

    removed_df[
        removed_columns
    ].to_csv(
        REMOVED_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    print("\n[5/5] Cleaning summary")

    print("\n" + "=" * 70)
    print("Final dataset cleaning complete")
    print("=" * 70)

    print(
        f"\nOriginal tracks: "
        f"{len(manifest):,}"
    )

    print(
        f"Clean tracks:    "
        f"{len(clean_manifest):,}"
    )

    print(
        f"Removed tracks:  "
        f"{len(removed_df):,}"
    )

    print("\nRemoval reasons:")

    reason_counts = (
        removed_df["removal_reason"]
        .value_counts()
    )

    for reason, count in reason_counts.items():

        print(
            f"  {reason:<30}"
            f"{count:>6,}"
        )

    # --------------------------------------------------------
    # Split counts
    # --------------------------------------------------------

    print("\nFinal split distribution:")

    split_counts = (
        clean_manifest["split"]
        .value_counts()
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"  {split:<12}"
            f"{split_counts.get(split, 0):>7,}"
        )

    # --------------------------------------------------------
    # Genre counts
    # --------------------------------------------------------

    print("\nFinal genre distribution:")

    genre_counts = (
        clean_manifest["genre"]
        .value_counts()
        .sort_index()
    )

    for genre, count in genre_counts.items():

        percentage = (
            count
            / len(clean_manifest)
            * 100
        )

        print(
            f"  {genre:<22}"
            f"{count:>6,}"
            f" ({percentage:>5.2f}%)"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\nOutput files:")

    print(
        f"  {CLEAN_MANIFEST_PATH}"
    )

    print(
        f"  {CLEAN_TRAIN_PATH}"
    )

    print(
        f"  {CLEAN_VALIDATION_PATH}"
    )

    print(
        f"  {CLEAN_TEST_PATH}"
    )

    print(
        f"  {REMOVED_PATH}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()