from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_FILE = PROJECT_ROOT / "data" / "manifest.csv"
METADATA_FILE = PROJECT_ROOT / "fma_metadata" / "tracks.csv"

OUTPUT_DIR = PROJECT_ROOT / "data"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MelodyMatch - FMA Medium Split Creator")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\n[1/5] Checking files...")

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_FILE}\n\n"
            "Run build_manifest.py first."
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    print("\n[2/5] Loading manifest...")

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    print(
        f"Manifest tracks: {len(manifest):,}"
    )

    # --------------------------------------------------------
    # Load official FMA metadata
    # --------------------------------------------------------

    print("\n[3/5] Loading official FMA split information...")

    tracks = pd.read_csv(
        METADATA_FILE,
        header=[0, 1],
        index_col=0
    )

    split_data = tracks[
        [
            ("set", "split"),
            ("track", "genre_top")
        ]
    ].copy()

    split_data.columns = [
        "split",
        "metadata_genre"
    ]

    split_data.index.name = "track_id"

    split_data = split_data.reset_index()

    split_data["track_id"] = (
        split_data["track_id"]
        .astype(int)
    )

    print(
        f"Metadata tracks: {len(split_data):,}"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\n[4/5] Matching tracks with official splits...")

    merged = manifest.merge(
        split_data,
        on="track_id",
        how="left",
        suffixes=("", "_metadata")
    )

    # --------------------------------------------------------
    # Validate genre consistency
    # --------------------------------------------------------

    genre_mismatch = (
        merged["genre"]
        != merged["metadata_genre"]
    )

    mismatch_count = genre_mismatch.sum()

    print(
        f"Genre mismatches: {mismatch_count:,}"
    )

    if mismatch_count > 0:

        print(
            "\nWARNING: Genre mismatch detected."
        )

        print(
            merged.loc[
                genre_mismatch,
                [
                    "track_id",
                    "genre",
                    "metadata_genre"
                ]
            ].head(20).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Check missing split assignments
    # --------------------------------------------------------

    missing_split = merged["split"].isna().sum()

    print(
        f"Tracks without split assignment: "
        f"{missing_split:,}"
    )

    if missing_split > 0:

        print(
            "\nWARNING: Some tracks have no official split."
        )

        merged = merged.dropna(
            subset=["split"]
        ).copy()

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    final_manifest = merged[
        [
            "track_id",
            "audio_path",
            "genre",
            "split"
        ]
    ].copy()

    # --------------------------------------------------------
    # Normalize split names
    # --------------------------------------------------------

    final_manifest["split"] = (
        final_manifest["split"]
        .str.lower()
        .str.strip()
    )

    valid_splits = {
        "training",
        "validation",
        "test"
    }

    invalid_splits = set(
        final_manifest["split"].unique()
    ) - valid_splits

    if invalid_splits:

        raise ValueError(
            f"Unexpected split values: "
            f"{invalid_splits}"
        )

    # --------------------------------------------------------
    # Rename training → train
    # --------------------------------------------------------

    final_manifest["split"] = (
        final_manifest["split"]
        .replace({
            "training": "train"
        })
    )

    # --------------------------------------------------------
    # Save combined manifest
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / "manifest_with_splits.csv"

    final_manifest = final_manifest.sort_values(
        "track_id"
    ).reset_index(drop=True)

    final_manifest.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Create separate files
    # --------------------------------------------------------

    print("\nCreating split files...")

    split_files = {
        "train": OUTPUT_DIR / "train.csv",
        "validation": OUTPUT_DIR / "validation.csv",
        "test": OUTPUT_DIR / "test.csv"
    }

    for split_name, output_path in tqdm(
        split_files.items(),
        desc="Saving splits",
        unit="split"
    ):

        split_df = final_manifest[
            final_manifest["split"] == split_name
        ].copy()

        split_df.to_csv(
            output_path,
            index=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Split creation complete")
    print("=" * 70)

    print("\nOverall split distribution:")

    print(
        final_manifest["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Per-split genre distribution
    # --------------------------------------------------------

    print("\nGenre distribution by split:")

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        split_df = final_manifest[
            final_manifest["split"] == split_name
        ]

        print(
            f"\n--- {split_name.upper()} "
            f"({len(split_df):,} tracks) ---"
        )

        counts = (
            split_df["genre"]
            .value_counts()
            .sort_index()
        )

        for genre, count in counts.items():

            percentage = (
                count / len(split_df) * 100
            )

            print(
                f"  {genre:<20} "
                f"{count:>6,} "
                f"({percentage:>5.2f}%)"
            )

    # --------------------------------------------------------
    # Final file information
    # --------------------------------------------------------

    print("\nOutput files:")

    print(
        f"  {output_file}"
    )

    for path in split_files.values():

        print(
            f"  {path}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()