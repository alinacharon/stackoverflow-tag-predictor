import pytest
from fastapi.testclient import TestClient
from api.main import app
import os
import logging
import asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def client():
    """Create a test client with the test app."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}


def test_predict_valid_text(client):
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info("Starting test_predict_valid_text")

    payload = {"text": "How to use Python dictionaries effectively?"}
    response = client.post("/predict", json=payload)

    print(f"\n=== DEBUG INFO FOR test_predict_valid_text ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Text: {response.text}")
    print(f"===============================================")

    if response.status_code != 200:
        print(f"Expected 200, got {response.status_code}")
        print(f"Response content: {response.text}")

    assert response.status_code == 200
    data = response.json()
    assert "tags" in data
    assert isinstance(data["tags"], list)


def test_predict_empty_text(client):
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_invalid_text(client):
    payload = {"text": "!!!@@@###"}
    response = client.post("/predict", json=payload)

    print(f"\n=== DEBUG INFO FOR test_predict_invalid_text ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Text: {response.text}")
    print(f"===============================================")

    if response.status_code not in [200, 400]:
        print(f"Expected 200 or 400, got {response.status_code}")
        print(f"Response content: {response.text}")

    assert response.status_code in [200, 400]
