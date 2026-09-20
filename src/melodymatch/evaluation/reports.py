import pandas as pd
from sklearn.metrics import classification_report


def generate_classification_report(
    y_true,
    y_pred,
    class_names,
):
    """
    Generate a classification report as a pandas DataFrame.
    """

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).T