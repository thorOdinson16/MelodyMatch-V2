from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR = PROJECT_ROOT / "results" / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name):
    df = pd.read_csv(
        DATA_DIR / f"clean_{name}_features.csv"
    )

    X = df.drop(
        columns=["track_id", "genre", "audio_path", "split"]
    )

    y = df["genre"]

    return X, y


def evaluate(model, X, y, split_name):
    predictions = model.predict(X)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y, predictions
        ),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    print(f"\n{split_name.upper()} RESULTS")
    print("-" * 50)

    for key, value in metrics.items():
        if key != "split":
            print(
                f"{key}: {value:.4f}"
            )

    print("\nPer-class report:")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )

    return metrics


def main():
    print("Loading datasets...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    print(
        f"Train: {X_train.shape}"
    )
    print(
        f"Validation: {X_val.shape}"
    )
    print(
        f"Test: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    print("\nFitting StandardScaler...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
        verbose=1,
    )

    model.fit(X_train, y_train)

    print("\nTraining complete.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_metrics = []

    all_metrics.append(
        evaluate(
            model,
            X_train,
            y_train,
            "train",
        )
    )

    all_metrics.append(
        evaluate(
            model,
            X_val,
            y_val,
            "validation",
        )
    )

    all_metrics.append(
        evaluate(
            model,
            X_test,
            y_test,
            "test",
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_path = (
        RESULTS_DIR
        / "logistic_baseline.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save model + scaler
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR
        / "logistic_baseline.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
        },
        model_path,
    )

    print(
        f"\nMetrics saved to: {metrics_path}"
    )

    print(
        f"Model saved to: {model_path}"
    )


if __name__ == "__main__":
    main()