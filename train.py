from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models/iris_model.pkl")
N_ESTIMATORS = 100
RANDOM_STATE = 42


def train() -> float:
    """Train a RandomForest on Iris, log to MLflow, and save the model."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.set_experiment("iris-classification")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    train()