import pytest
import logging
import requests
import time
import os
import subprocess
import sys
import signal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local API configuration
API_URL = "http://localhost:3000"
MAX_RETRIES = 20
TIMEOUT = 5  # таймаут в секундах

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
    """Start the local server"""
    try:
        # Получаем абсолютный путь к директории проекта
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        api_dir = os.path.join(project_root, "api")

        logger.info(f"Starting server from directory: {api_dir}")
        logger.info(f"Current working directory: {os.getcwd()}")

        # Запускаем сервер с перенаправлением вывода
        process = subprocess.Popen(
            ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"],
            cwd=api_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Создаем новую группу процессов
        )

        # Даем серверу время на запуск
        time.sleep(5)

        # Проверяем, не завершился ли процесс
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(
                f"Server failed to start. Exit code: {process.returncode}")
            logger.error(f"Stdout: {stdout}")
            logger.error(f"Stderr: {stderr}")
            raise Exception("Server failed to start")

        logger.info("Server process started successfully")
        return process

    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise


def wait_for_server(url: str, max_attempts: int = MAX_RETRIES, delay: int = 1) -> bool:
    """Wait for the server to be ready"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.1)
    session.mount('http://', HTTPAdapter(max_retries=retries))

    for attempt in range(max_attempts):
        try:
            logger.info(
                f"Attempt {attempt + 1}/{max_attempts}: Checking server health...")
            response = session.get(f"{url}/health", timeout=TIMEOUT)
            if response.status_code == 200:
                logger.info("Server is ready!")
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
            if server_process:
                stdout, stderr = server_process.communicate()
                logger.error("Server failed to start within timeout")
                logger.error(f"Server stdout: {stdout}")
                logger.error(f"Server stderr: {stderr}")
            raise Exception("Failed to start local server")

        yield

    finally:
        if server_process:
            try:
                # Отправляем сигнал всей группе процессов
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping server: {str(e)}")
                try:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                except:
                    pass


@pytest.mark.timeout(30)
def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.timeout(30)
def test_predict_valid_text():
    """Test prediction with valid text"""
    text = "How to implement a binary search tree in Python?"
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) > 0


@pytest.mark.timeout(30)
def test_predict_empty_text():
    """Test prediction with empty text"""
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": ""}
    )
    assert response.status_code == 422


@pytest.mark.timeout(30)
def test_predict_invalid_text():
    """Test prediction with invalid text"""
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": 123}
    )
    assert response.status_code == 422


@pytest.mark.timeout(30)
def test_server_performance():
    """Test server performance with multiple requests"""
    text = "How to implement a binary search tree in Python?"
    num_requests = 10
    successful_requests = 0

    for _ in range(num_requests):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": text},
                timeout=5
            )
            if response.status_code == 200:
                successful_requests += 1
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            continue

    assert successful_requests > 0, "No successful requests"
