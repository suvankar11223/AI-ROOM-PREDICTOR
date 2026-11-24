import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

BASE_LISTING = {
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Midtown",
    "latitude": 40.7549,
    "longitude": -73.9840,
    "price": 180,
    "minimum_nights": 2,
    "availability_365": 120,
    "calculated_host_listings_count": 1,
}


def test_health_reports_loaded_models(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_model_catalog_contains_both_scenarios(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert set(response.json()["available_models"]) == {"without_reviews", "with_reviews"}


def test_without_reviews_prediction_returns_all_class_probabilities(client):
    response = client.post(
        "/api/v1/predict",
        json={"mode": "without_reviews", "listing": BASE_LISTING},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predicted_room_type"] in {"Entire home/apt", "Private room", "Shared room"}
    assert set(body["probabilities"]) == {"Entire home/apt", "Private room", "Shared room"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_with_reviews_prediction_accepts_review_features(client):
    listing = {**BASE_LISTING, "number_of_reviews": 12, "reviews_per_month": 1.4}
    response = client.post(
        "/api/v1/predict",
        json={"mode": "with_reviews", "listing": listing},
    )
    assert response.status_code == 200, response.text


def test_with_reviews_accepts_zero_reviews_with_missing_monthly_rate(client):
    listing = {**BASE_LISTING, "number_of_reviews": 0, "reviews_per_month": None}
    response = client.post(
        "/api/v1/predict",
        json={"mode": "with_reviews", "listing": listing},
    )
    assert response.status_code == 200, response.text


def test_with_reviews_requires_review_count(client):
    response = client.post(
        "/api/v1/predict",
        json={"mode": "with_reviews", "listing": BASE_LISTING},
    )
    assert response.status_code == 422


def test_invalid_availability_is_rejected(client):
    listing = {**BASE_LISTING, "availability_365": 500}
    response = client.post(
        "/api/v1/predict",
        json={"mode": "without_reviews", "listing": listing},
    )
    assert response.status_code == 422
