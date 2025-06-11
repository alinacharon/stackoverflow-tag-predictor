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
MAX_RETRIES = 3
TIMEOUT = 30  # таймаут в секундах

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
    """Start the FastAPI server locally"""
    try:
        # Запускаем сервер в фоновом режиме
        process = subprocess.Popen(
            ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Даем серверу 5 секунд на запуск
        time.sleep(5)

        # Проверяем, не завершился ли процесс
        if process.poll() is not None:
            stderr = process.stderr.read()
            stdout = process.stdout.read()
            logger.error(
                f"Server failed to start. Exit code: {process.returncode}")
            logger.error(f"Stdout: {stdout}")
            logger.error(f"Stderr: {stderr}")
            raise RuntimeError("Server process terminated unexpectedly")

        return process
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise


def wait_for_server(url: str, max_attempts: int = 20, delay: int = 1) -> bool:
    """Wait for the server to be ready"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.1)
    session.mount('http://', HTTPAdapter(max_retries=retries))

    for attempt in range(max_attempts):
        try:
            response = session.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts}: Server not ready. Error: {str(e)}")
            time.sleep(delay)
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_server():
    """Setup and teardown of the local test server"""
    server_process = None
    try:
        # Start the server
        server_process = start_local_server()

        # Wait for server to be ready
        if not wait_for_server(API_URL):
            stderr = server_process.stderr.read()
            stdout = server_process.stdout.read()
            logger.error(f"Uvicorn stderr: {stderr}")
            logger.error(f"Uvicorn stdout: {stdout}")
            server_process.terminate()
            pytest.fail("Failed to start local server")

        yield

    finally:
        # Cleanup
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()


@pytest.mark.timeout(TIMEOUT)
def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}


@pytest.mark.timeout(TIMEOUT)
def test_predict_valid_text():
    """Test prediction with valid text input"""
    test_text = "How to implement a binary search tree in Python?"
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": test_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert "tags" in data
    assert isinstance(data["tags"], list)


@pytest.mark.timeout(TIMEOUT)
def test_predict_empty_text():
    """Test prediction with empty text input"""
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": ""}
    )
    assert response.status_code == 422


@pytest.mark.timeout(TIMEOUT)
def test_predict_invalid_text():
    """Test prediction with invalid text input"""
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": 123}
    )
    assert response.status_code == 422


@pytest.mark.timeout(TIMEOUT)
def test_server_performance():
    """Test server performance with multiple requests"""
    test_text = "How to implement a binary search tree in Python?"
    num_requests = 10
    successful_requests = 0

    for _ in range(num_requests):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": test_text},
                timeout=5
            )
            if response.status_code == 200:
                successful_requests += 1
        except requests.exceptions.RequestException:
            continue

    assert successful_requests > 0, "No successful requests in performance test"
