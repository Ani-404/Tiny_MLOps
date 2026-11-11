from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app import app
from train import FEATURE_NAMES, train


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
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_with_direct_features(client):
    # Vector: [days_push, commits, close_rate, open_issues, stars, forks, readme, contrib, lic, coc, good_first]
    thriving_vector = [1.5, 12.0, 0.85, 25, 45000, 8000, 1, 1, 1, 1, 1]
    response = client.post("/predict", json={"features": thriving_vector})
    assert response.status_code == 200
    data = response.json()
    assert "health_tier" in data
    assert "health_score" in data
    assert data["health_tier"] in [
        "Thriving & Welcoming",
        "Healthy & Maintained",
        "Developing / Small Community",
        "Stale / High Risk",
    ]
    assert 0.0 <= data["health_score"] <= 100.0
    assert len(data["insights"]) > 0


@patch("app.extract_github_features", new_callable=AsyncMock)
def test_predict_with_repo_url_mocked(mock_extract, client):
    mock_extract.return_value = {
        "repo_name": "pallets/flask",
        "days_since_last_push": 2.0,
        "avg_weekly_commits": 8.5,
        "issue_close_rate": 0.78,
        "open_issues_count": 42,
        "stars_count": 68000,
        "forks_count": 16000,
        "has_readme": 1,
        "has_contributing": 1,
        "has_license": 1,
        "has_code_of_conduct": 1,
        "has_good_first_issues": 1,
    }

    response = client.post(
        "/predict", json={"repo_url": "https://github.com/pallets/flask"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["repo"] == "pallets/flask"
    assert data["health_tier"] in ["Thriving & Welcoming", "Healthy & Maintained"]
    assert data["health_score"] >= 70.0
    assert data["confidence"] > 0.0
    mock_extract.assert_called_once_with("pallets", "flask")


def test_predict_invalid_feature_length(client):
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


def test_predict_empty_request(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_repo_url(client):
    response = client.post("/predict", json={"repo_url": "invalid-url"})
    assert response.status_code == 422
