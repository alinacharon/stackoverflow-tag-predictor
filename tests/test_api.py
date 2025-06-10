import pytest
import logging
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for local Docker server
DOCKER_API_URL = "http://localhost:3000"


def wait_for_server(url: str, max_retries: int = 5, delay: int = 2):
    """Wait for server to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            logger.info(
                f"Attempt {i+1}/{max_retries}: Server is not ready yet...")
            time.sleep(delay)
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_server():
    """Check server availability before running tests"""
    if not wait_for_server(DOCKER_API_URL):
        pytest.fail(
            "Failed to connect to server. Make sure Docker container is running.")


def test_health_check():
    """Test server health check endpoint"""
    response = requests.get(f"{DOCKER_API_URL}/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}


def test_predict_valid_text():
    """Test prediction for valid text"""
    logger.info("Starting test_predict_valid_text")

    payload = {"text": "How to use Python dictionaries effectively?"}
    response = requests.post(f"{DOCKER_API_URL}/predict", json=payload)

    print(f"\n=== DEBUG INFO FOR test_predict_valid_text ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Text: {response.text}")
    print(f"===============================================")

    assert response.status_code == 200
    data = response.json()
    assert "tags" in data
    assert isinstance(data["tags"], list)


def test_predict_empty_text():
    """Test prediction for empty text"""
    payload = {"text": ""}
    response = requests.post(f"{DOCKER_API_URL}/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_invalid_text():
    """Test prediction for invalid text"""
    payload = {"text": "!!!@@@###"}
    response = requests.post(f"{DOCKER_API_URL}/predict", json=payload)

    print(f"\n=== DEBUG INFO FOR test_predict_invalid_text ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Text: {response.text}")
    print(f"===============================================")

    assert response.status_code in [200, 400]
