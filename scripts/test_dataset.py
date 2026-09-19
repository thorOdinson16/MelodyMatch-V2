from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from melodymatch.data.dataset import FMAMelDataset
from melodymatch.data.labels import GENRE_TO_INDEX


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "clean_train.csv"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "mel_cache"
)


# ============================================================
# Dataset
# ============================================================

dataset = FMAMelDataset(
    manifest_path=MANIFEST,
    project_root=PROJECT_ROOT,
    genre_to_index=GENRE_TO_INDEX,
    cache_dir=CACHE_DIR,
)


print("=" * 70)
print("MelodyMatch - Dataset Test")
print("=" * 70)

print(f"\nDataset size: {len(dataset):,}")


# ============================================================
# Test individual sample
# ============================================================

print("\nTesting first sample...")

sample = dataset[0]

print(
    f"Track ID: {sample['track_id']}"
)

print(
    f"Mel shape: {tuple(sample['mel'].shape)}"
)

print(
    f"Label: {sample['label'].item()}"
)


# ============================================================
# DataLoader test
# ============================================================

print("\nTesting DataLoader...")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)


batch = next(iter(loader))

print(
    f"Batch mel shape: "
    f"{tuple(batch['mel'].shape)}"
)

print(
    f"Batch labels shape: "
    f"{tuple(batch['label'].shape)}"
)

print(
    f"Batch track IDs: "
    f"{batch['track_id'].tolist()}"
)


# ============================================================
# Cache test
# ============================================================

print("\nTesting cache...")

print(
    f"Cached files currently: "
    f"{len(list(CACHE_DIR.glob('*.pt'))):,}"
)


print("\nDataset pipeline test complete.")