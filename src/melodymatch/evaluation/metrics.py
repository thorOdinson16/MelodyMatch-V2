from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


def compute_metrics(y_true, y_pred):
    """
    Compute the primary classification metrics used by MelodyMatch.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }