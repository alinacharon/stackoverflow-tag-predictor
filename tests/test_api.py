import pytest
import logging
import requests
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration for EC2 instance
API_URL = os.getenv('AWS_API_URL')

# Configure requests session 
session = requests.Session()
retry_strategy = Retry(
    total=3, 
    backoff_factor=1,  
    status_forcelist=[500, 502, 503, 504]  
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


def wait_for_server(url: str, max_retries: int = 5, delay: int = 2):
    """Wait for server to be ready with improved error handling"""
    for i in range(max_retries):
        try:
            # Check the /health endpoint now
            response = session.get(f"{url}/health", timeout=10)
            if response.status_code == 200 and response.json().get("status") == "healthy":
                logger.info("Server is ready and responding")
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Attempt {i+1}/{max_retries}: Server not ready. Error: {str(e)}")
            time.sleep(delay)
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_server():
    """Check server availability before running tests with improved error handling"""
    logger.info(f"Checking server availability at {API_URL}")
    if not wait_for_server(API_URL):
        pytest.fail(
            f"Failed to connect to server at {API_URL}. "
            "Make sure the EC2 instance is running and the API is accessible."
        )
    logger.info("Server is available, proceeding with tests")


def test_health_check():
    """Test server health check endpoint with improved error handling"""
    try:
        response = session.get(f"{API_URL}/health", timeout=10)
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Health check failed: {str(e)}")


def test_predict_valid_text():
    """Test prediction for valid text with improved error handling and logging"""
    logger.info("Starting test_predict_valid_text")
    test_text = "How to use Python dictionaries effectively?"

    try:
        payload = {"text": test_text}
        response = session.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30  # Increased timeout for prediction
        )

        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        logger.info(f"Response body: {response.text}")

        assert response.status_code == 200
        data = response.json()
        assert "tags" in data
        assert isinstance(data["tags"], list)
        assert len(data["tags"]) > 0, "No tags were predicted"

        logger.info(f"Successfully predicted tags: {data['tags']}")

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Prediction request failed: {str(e)}")
    except AssertionError as e:
        pytest.fail(f"Prediction validation failed: {str(e)}")


def test_predict_empty_text():
    """Test prediction for empty text with improved error handling"""
    try:
        payload = {"text": ""}
        response = session.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        assert response.status_code == 422
        assert "detail" in response.json()
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Empty text test failed: {str(e)}")


def test_predict_invalid_text():
    """Test prediction for invalid text with improved error handling"""
    logger.info("Starting test_predict_invalid_text")
    test_text = "!!!@@@###"

    try:
        payload = {"text": test_text}
        response = session.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=60 
        )

        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        logger.info(f"Response body: {response.text}")

        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            assert "tags" in data
            assert isinstance(data["tags"], list)
            logger.info(f"Received tags for invalid text: {data['tags']}")

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Invalid text test failed: {str(e)}")


def test_server_performance():
    """Test server performance with multiple requests"""
    logger.info("Starting performance test")
    test_texts = [
        "How to use Python dictionaries effectively?",
        "What is the best way to learn machine learning?",
        "How to deploy a Docker container?"
    ]

    start_time = time.time()
    successful_requests = 0

    for text in test_texts:
        try:
            payload = {"text": text}
            response = session.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                successful_requests += 1
        except requests.exceptions.RequestException:
            continue

    end_time = time.time()
    total_time = end_time - start_time

    logger.info(f"Performance test completed in {total_time:.2f} seconds")
    logger.info(
        f"Successful requests: {successful_requests}/{len(test_texts)}")

    assert successful_requests > 0, "No successful requests in performance test"
    assert total_time < 90, "Performance test took too long"
