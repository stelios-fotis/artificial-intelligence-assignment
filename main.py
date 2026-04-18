from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path(__file__).with_name("processed.cleveland.data.csv")
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def main() -> None:
    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES, na_values="?")
    df = df.dropna(subset=["ca", "thal"])
    df = df.apply(pd.to_numeric)

    X = df.drop(columns=["num"])
    y = (df["num"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # split dataset to 80% for training and 20% for test
        random_state=42,  # make the random split repetable
        stratify=y,  # keep the balance split of outcomes in the sets balanced
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),  # Scale features so columns are on a similar range
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(
                        32,
                        16,
                    ),  # Two hidden layers with 32 and 16 neurons
                    activation="relu",  # Helps the network learn non-linear patterns
                    solver="adam",  # Method used to update the network during training
                    max_iter=1000,  # Maximum training iterations
                    random_state=42,  # Keeps results repeatable between runs
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, test_predictions)

    print(f"Rows after filtering: {len(df)}")
    print(f"Feature columns: {len(X.columns)}")
    print(f"Train accuracy: {accuracy_score(y_train, train_predictions):.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, test_predictions):.3f}")
    print("\nConfusion matrix:")
    print("             Pred 0   Pred 1")
    print(f"Actual 0:    {cm[0, 0]:>6}   {cm[0, 1]:>6}")
    print(f"Actual 1:    {cm[1, 0]:>6}   {cm[1, 1]:>6}")


if __name__ == "__main__":
    main()
