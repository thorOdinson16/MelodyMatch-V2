import pandas as pd
from sklearn.metrics import confusion_matrix


def generate_confusion_matrix(
    y_true,
    y_pred,
    class_names,
):
    """
    Generate a confusion matrix as a pandas DataFrame.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    return pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )


def normalize_confusion_matrix(confusion_df):
    """
    Normalize each row of a confusion matrix.

    Each row represents the true class, so values represent
    the proportion of samples from that class assigned to
    each predicted class.
    """

    row_sums = confusion_df.sum(axis=1)

    normalized = confusion_df.div(
        row_sums.replace(0, 1),
        axis=0,
    )

    return normalized