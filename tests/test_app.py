import pytest
from fastapi.testclient import TestClient

from app import app
from train import train


@pytest.fixture(scope="module", autouse=True)
def trained_model():
    train()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_features(client):
    response = client.post(
        "/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "class_name" in data
    assert data["class_name"] in ["setosa", "versicolor", "virginica"]


def test_predict_invalid_feature_count(client):
    response = client.post("/predict", json={"features": [5.1, 3.5]})
    assert response.status_code == 422