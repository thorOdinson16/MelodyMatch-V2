from pathlib import Path
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_train():
    df = pd.read_csv(
        DATA_DIR / "clean_train_features.csv"
    )

    X = df.drop(
        columns=[
            "track_id",
            "genre",
            "audio_path",
            "split",
        ]
    )

    y = df["genre"]

    return X, y


def main():
    X, y = load_train()

    # Benchmark on 5,000 training samples.
    X = X.iloc[:5000]
    y = y.iloc[:5000]

    print("Benchmark samples:", len(X))
    print("Features:", X.shape[1])

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
    )

    print("\nTraining RBF SVM benchmark...")

    start = time.perf_counter()

    model.fit(X, y)

    elapsed = time.perf_counter() - start

    print(
        f"\nBenchmark training time: "
        f"{elapsed / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()