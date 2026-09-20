import torch
from torch.utils.data import WeightedRandomSampler


def create_balanced_sampler(
    labels,
) -> WeightedRandomSampler:
    """
    Create an inverse-frequency weighted sampler.

    Each training sample receives a weight based on
    the inverse frequency of its class.
    """

    labels = torch.as_tensor(
        labels,
        dtype=torch.long,
    )

    class_counts = torch.bincount(labels)

    class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )