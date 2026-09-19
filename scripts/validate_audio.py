from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest_with_splits.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "audio_validation.csv"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MelodyMatch - FMA Medium Audio Validation")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check manifest
    # --------------------------------------------------------

    print("\n[1/3] Loading manifest...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    print(f"Tracks to validate: {len(df):,}")

    # --------------------------------------------------------
    # 2. Validate audio
    # --------------------------------------------------------

    print("\n[2/3] Validating audio files...")
    print("This may take a while.\n")

    results = []

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Validating audio",
        unit="file"
    ):

        audio_path = PROJECT_ROOT / row.audio_path

        result = {
            "track_id": row.track_id,
            "genre": row.genre,
            "split": row.split,
            "audio_path": row.audio_path,
            "exists": False,
            "valid": False,
            "duration": None,
            "sample_rate": None,
            "channels": None,
            "error": None,
        }

        # File existence
        if not audio_path.exists():
            result["error"] = "file_not_found"
            results.append(result)
            continue

        result["exists"] = True

        try:
            # Load only a tiny amount of audio.
            # This verifies that librosa/audioread/soundfile
            # can actually decode the file without loading
            # the complete 30-second track into memory.
            y, sr = librosa.load(
                audio_path,
                sr=None,
                mono=False,
                duration=1.0
            )

            # Get metadata
            info = librosa.get_duration(
                path=audio_path
            )

            result["valid"] = True
            result["duration"] = float(info)
            result["sample_rate"] = int(sr)

            if y.ndim == 1:
                result["channels"] = 1
            else:
                result["channels"] = y.shape[0]

        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    # --------------------------------------------------------
    # 3. Save results
    # --------------------------------------------------------

    print("\n[3/3] Saving validation results...")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(results_df)
    existing = results_df["exists"].sum()
    valid = results_df["valid"].sum()
    invalid = total - valid

    print("\n" + "=" * 70)
    print("Audio validation complete")
    print("=" * 70)

    print(f"\nTotal tracks:       {total:,}")
    print(f"Files found:        {existing:,}")
    print(f"Valid audio:        {valid:,}")
    print(f"Invalid/unreadable: {invalid:,}")

    if valid > 0:

        valid_df = results_df[results_df["valid"]]

        print("\nAudio statistics:")

        print(
            f"  Duration:"
            f" {valid_df['duration'].min():.2f}s"
            f" - {valid_df['duration'].max():.2f}s"
        )

        print(
            f"  Average duration:"
            f" {valid_df['duration'].mean():.2f}s"
        )

        print(
            f"  Sample rates:"
            f" {sorted(valid_df['sample_rate'].unique().tolist())}"
        )

        print(
            f"  Channels:"
            f" {sorted(valid_df['channels'].unique().tolist())}"
        )

    if invalid > 0:

        print("\nInvalid audio files by error:")

        error_counts = (
            results_df.loc[~results_df["valid"], "error"]
            .value_counts()
        )

        for error, count in error_counts.items():
            print(f"  {error}: {count}")

    print("\nOutput:")
    print(f"  {OUTPUT_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()