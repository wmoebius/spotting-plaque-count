from .utils import DIR_ROOT, FILE_CONFIG
from .logging_config import logging
import toml

try:
    config = toml.load(FILE_CONFIG)
    logging.info(f"Loaded config from {FILE_CONFIG}")
except Exception as e:
    logging.error(f"Failed to load config from {FILE_CONFIG}: {e}")
