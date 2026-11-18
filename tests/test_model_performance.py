from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_accuracy():
    """Pass only when the saved model's test accuracy is greater than 72%."""
    model = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
    df = pd.read_csv(PROJECT_ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Match the preprocessing and evaluation split in the training notebook.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy > 0.72, (
        f"Model accuracy must be greater than 0.72; actual accuracy: {accuracy:.4f}"
    )
