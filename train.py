"""Train a RandomForestClassifier for GitHub Repository Health & Welcomeness Scoring.

Logs experiments, hyper-parameters, and performance metrics to MLflow, and saves
the serialized model pipeline to disk.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

MODEL_PATH = Path("models/repo_health_model.pkl")
DATA_PATH = Path("data/repos.csv")
N_ESTIMATORS = 100
RANDOM_STATE = 42

FEATURE_NAMES = [
    "days_since_last_push",
    "avg_weekly_commits",
    "issue_close_rate",
    "open_issues_count",
    "stars_count",
    "forks_count",
    "has_readme",
    "has_contributing",
    "has_license",
    "has_code_of_conduct",
    "has_good_first_issues",
]

TIER_NAMES = [
    "Thriving & Welcoming",
    "Healthy & Maintained",
    "Developing / Small Community",
    "Stale / High Risk",
]


def load_dataset() -> pd.DataFrame:
    """Load the repository dataset or generate it if missing."""
    if not DATA_PATH.exists():
        from data_generator import generate_data
        return generate_data(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def train() -> float:
    """Train the model pipeline, log metrics to MLflow, and save the artifact."""
    df = load_dataset()
    X = np.asarray(df[FEATURE_NAMES].values, dtype=np.float64)
    y = np.asarray(df["health_tier"].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                max_depth=12,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))

    mlflow.set_experiment("github-repo-health")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("features", FEATURE_NAMES)
        mlflow.log_param("feature_count", len(FEATURE_NAMES))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # Log feature importances
        rf: RandomForestClassifier = pipeline.named_steps["classifier"]
        for feat_name, imp in zip(FEATURE_NAMES, rf.feature_importances_):
            mlflow.log_metric(f"feat_imp_{feat_name}", float(imp))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    # Attach feature names for downstream inspection
    pipeline.feature_names = FEATURE_NAMES

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    return accuracy


if __name__ == "__main__":
    train()
    print("All done")
