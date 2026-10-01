from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "House Price Prediction API is running"
    assert data["model"] == "Random Forest"


def test_prediction():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_price" in data
    assert "price_unit" in data

    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0


def test_invalid_input():
    payload = {
        "MedInc": "invalid",
        "HouseAge": 20.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422