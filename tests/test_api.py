import pytest
import logging
import requests
import time
import os
import subprocess
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local API configuration
API_URL = "http://localhost:3000"

# Configure requests session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


def start_local_server():
    """Start the local API server for testing"""
    try:
        # Calculate the absolute path to the project root
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))

        # Check for models before starting (using absolute paths)
        model_path_abs = os.path.join(project_root, "models/model.pkl")
        mlb_path_abs = os.path.join(project_root, "models/mlb.pkl")
        assert os.path.exists(
            model_path_abs), f"model.pkl not found at {model_path_abs}!"
        assert os.path.exists(
            mlb_path_abs), f"mlb.pkl not found at {mlb_path_abs}!"

        # Configure environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        # Use absolute path for python executable and run uvicorn as a module
        # Set cwd to project_root to ensure relative imports from api/main.py work
        command = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",  # This will now be correctly found via PYTHONPATH
            "--host", "0.0.0.0",
            "--port", "3000"
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root,  # Set CWD to project root for consistent path resolution
            env=env
        )
        return process
    except Exception as e:
        logger.error(f"Failed to start local server: {e}")
        raise


def wait_for_server(url: str, max_retries: int = 20, delay: int = 5):
    """Wait for server to be ready with improved error handling"""
    for i in range(max_retries):
        try:
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
    """Setup and teardown of the local test server"""
    # Start the server
    server_process = start_local_server()

    # Wait for server to be ready
    if not wait_for_server(API_URL):
        stderr = server_process.stderr.read().decode()
        stdout = server_process.stdout.read().decode()
        logger.error(f"Uvicorn stderr: {stderr}")
        logger.error(f"Uvicorn stdout: {stdout}")
        server_process.terminate()
        pytest.fail("Failed to start local server")

    yield

    # Cleanup: stop the server
    server_process.terminate()
    server_process.wait()


def test_health_check():
    """Test the health check endpoint"""
    try:
        response = session.get(f"{API_URL}/health", timeout=10)
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Health check failed: {str(e)}")


def test_predict_valid_text():
    """Test prediction for valid text input"""
    logger.info("Starting test_predict_valid_text")
    test_text = "How to use Python dictionaries effectively?"

    try:
        payload = {"text": test_text}
        response = session.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
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
    """Test prediction for empty text input"""
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
    """Test prediction for invalid text input"""
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
    """Test server performance with multiple concurrent requests"""
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
