from pathlib import Path

import torch
from tqdm import tqdm

from melodymatch.evaluation import compute_metrics


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch: int = 0,
):
    """
    Train a model for one epoch.
    """

    model.train()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    progress = tqdm(
        dataloader,
        desc=f"Train epoch {epoch}",
        unit="batch",
    )

    for batch in progress:

        inputs = batch["mel"].to(
            device,
            non_blocking=True,
        )

        targets = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        running_loss += (
            loss.item() * targets.size(0)
        )

        all_targets.extend(
            targets.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss / len(dataloader.dataset)
    )

    metrics = compute_metrics(
        all_targets,
        all_predictions,
    )

    metrics["loss"] = epoch_loss

    return metrics


@torch.no_grad()
def evaluate_epoch(
    model,
    dataloader,
    criterion,
    device,
    description: str = "Validation",
):
    """
    Evaluate a model without gradient computation.
    """

    model.eval()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    progress = tqdm(
        dataloader,
        desc=description,
        unit="batch",
    )

    for batch in progress:

        inputs = batch["mel"].to(
            device,
            non_blocking=True,
        )

        targets = batch["label"].to(
            device,
            non_blocking=True,
        )

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        running_loss += (
            loss.item() * targets.size(0)
        )

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss / len(dataloader.dataset)
    )

    metrics = compute_metrics(
        all_targets,
        all_predictions,
    )

    metrics["loss"] = epoch_loss

    return metrics


def save_checkpoint(
    model,
    path: str | Path,
    optimizer=None,
    epoch: int | None = None,
    metrics: dict | None = None,
):
    """
    Save a model checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(
        checkpoint,
        path,
    )