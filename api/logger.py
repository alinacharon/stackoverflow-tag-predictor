import logging
from logging.handlers import TimedRotatingFileHandler
import os

LOG_DIR = 'logs'
LOG_PATH = os.path.join(LOG_DIR, 'api.log')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter(LOG_FORMAT)

if os.path.exists(LOG_DIR):
    file_handler = TimedRotatingFileHandler(
        LOG_PATH, when='midnight', interval=1, backupCount=7, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
