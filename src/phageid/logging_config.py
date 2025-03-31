import logging
from phageid.utils import DIR_ROOT

# Create a 'logs' folder if it doesn't exist
log_dir = DIR_ROOT / "logs"
if not log_dir.is_dir():
    log_dir.mkdir(parents=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "dev.log"),
        logging.FileHandler(log_dir / "warnings.log"),
        logging.FileHandler(log_dir / "errors.log"),
        logging.StreamHandler(),  # Console output
    ],
)
