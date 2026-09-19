from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIR = PROJECT_ROOT / "fma_medium"
METADATA_FILE = PROJECT_ROOT / "fma_metadata" / "tracks.csv"

OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "manifest.csv"


# ============================================================
# Helpers
# ============================================================

def track_id_to_path(track_id: int) -> Path:
    """
    Convert an FMA track ID into its corresponding audio path.

    FMA stores tracks using the first three digits of the
    zero-padded track ID as the directory name.

    Example:
        12345 -> fma_medium/012/012345.mp3
    """

    track_id_str = f"{int(track_id):06d}"

    folder = track_id_str[:3]
    filename = f"{track_id_str}.mp3"

    return AUDIO_DIR / folder / filename


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MelodyMatch - FMA Medium Manifest Builder")
    print("=" * 70)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    print("\n[1/5] Checking dataset paths...")

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio directory not found:\n{AUDIO_DIR}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Audio directory : {AUDIO_DIR}")
    print(f"Metadata file   : {METADATA_FILE}")

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print("\n[2/5] Loading tracks.csv...")

    tracks = pd.read_csv(
        METADATA_FILE,
        header=[0, 1],
        index_col=0
    )

    print(f"Metadata rows: {len(tracks):,}")

    # --------------------------------------------------------
    # Extract genre labels
    # --------------------------------------------------------

    print("\n[3/5] Extracting genre labels...")

    try:
        genre_series = tracks[("track", "genre_top")]
    except KeyError:
        raise KeyError(
            "Could not find ('track', 'genre_top') in tracks.csv"
        )

    track_ids = tracks.index.astype(int)

    manifest = pd.DataFrame({
        "track_id": track_ids,
        "genre": genre_series.values
    })

    total_metadata_tracks = len(manifest)

    # --------------------------------------------------------
    # Remove tracks without genre
    # --------------------------------------------------------

    manifest = manifest.dropna(
        subset=["genre"]
    ).copy()

    print(
        f"Tracks with genre labels: "
        f"{len(manifest):,} / {total_metadata_tracks:,}"
    )

    # --------------------------------------------------------
    # Map IDs to audio paths
    # --------------------------------------------------------

    print("\n[4/5] Checking audio files...")

    audio_paths = []
    audio_exists = []

    for track_id in tqdm(
        manifest["track_id"],
        total=len(manifest),
        desc="Scanning audio",
        unit="track"
    ):
        path = track_id_to_path(track_id)

        audio_paths.append(path)
        audio_exists.append(path.exists())

    manifest["audio_path"] = audio_paths
    manifest["audio_exists"] = audio_exists

    found = sum(audio_exists)
    missing = len(audio_exists) - found

    print(f"\nAudio files found   : {found:,}")
    print(f"Audio files missing : {missing:,}")

    # --------------------------------------------------------
    # Keep only existing audio
    # --------------------------------------------------------

    manifest = manifest[
        manifest["audio_exists"]
    ].copy()

    # --------------------------------------------------------
    # Convert paths to relative paths
    # --------------------------------------------------------

    manifest["audio_path"] = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in tqdm(
            manifest["audio_path"],
            total=len(manifest),
            desc="Normalizing paths",
            unit="track"
        )
    ]

    manifest = manifest.drop(
        columns=["audio_exists"]
    )

    # --------------------------------------------------------
    # Sort by track ID
    # --------------------------------------------------------

    manifest = manifest.sort_values(
        "track_id"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    print("\n[5/5] Saving manifest...")

    manifest.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Manifest created successfully")
    print("=" * 70)

    print(f"\nOutput file:")
    print(f"  {OUTPUT_FILE}")

    print(f"\nDataset summary:")
    print(f"  Metadata tracks : {total_metadata_tracks:,}")
    print(f"  With genre      : {len(manifest):,}")
    print(f"  Usable audio    : {len(manifest):,}")
    print(f"  Missing audio   : {missing:,}")

    print(
        f"\nNumber of genres: "
        f"{manifest['genre'].nunique()}"
    )

    print("\nGenre distribution:")

    genre_counts = (
        manifest["genre"]
        .value_counts()
        .sort_index()
    )

    for genre, count in genre_counts.items():
        percentage = count / len(manifest) * 100

        print(
            f"  {genre:<20} "
            f"{count:>6,} "
            f"({percentage:>5.2f}%)"
        )

    print("\nFirst five rows:")

    print(
        manifest.head().to_string(
            index=False
        )
    )

    print("\nDone.")


if __name__ == "__main__":
    main()