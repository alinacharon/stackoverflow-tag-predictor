import pytest
from fastapi.testclient import TestClient
from api.main import app 

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_valid_text():
    payload = {"text": "How to use Python dictionaries effectively?"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tags" in data
    assert isinstance(data["tags"], list)
    assert len(data["tags"]) > 0

def test_predict_empty_text():
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # pydantic validation will not miss empty text

def test_predict_invalid_text():
    payload = {"text": "!!!@@@###"}
    response = client.post("/predict", json=payload)
    # return 400 if empty after clearing, or 200 with empty tags
    assert response.status_code in [200, 400]