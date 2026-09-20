import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    gamma=0 reduces to standard cross-entropy.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()

        self.gamma = gamma
        self.weight = weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )

        probabilities = torch.exp(-ce_loss)

        focal_loss = (
            (1.0 - probabilities) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


def inverse_frequency_weights(
    class_counts: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Compute smoothed inverse-frequency class weights.

    w_c = (N / N_c)^alpha

    The resulting weights are normalized to mean 1.
    """

    class_counts = class_counts.float()

    total = class_counts.sum()

    weights = (total / class_counts) ** alpha

    weights = weights / weights.mean()

    return weights