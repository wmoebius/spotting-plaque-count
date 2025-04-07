import logging
from pathlib import Path

import toml
from platformdirs import user_config_dir, user_log_dir

PATH_CONFIG = Path(user_config_dir("phageid")) / "config.toml"

if PATH_CONFIG.is_file():
    try:
        config = toml.load(PATH_CONFIG)
    except Exception as e:
        raise e
else:
    default_config = {
        'x_spacing': {'min': 10, 'max': 100, 'current': 20.6},
        'x_offset': {'min': 0, 'max': 100, 'current': 32.5},
        'y_spacing': {'min': 0, 'max': 100, 'current': 21.6},
        'y_offset': {'min': 0, 'max': 100, 'current': 31.0},
        'rows': {'min': 1, 'max': 30, 'current': 12},
        'columns': {'min': 1, 'max': 15, 'current': 7},
        'radius': {'min': 1, 'max': 150, 'current': 10.7},
        'sample_segmentation': {'scaling_factor': 10},
        'logging': {'level': "debug"},
    }

    if not PATH_CONFIG.parent.is_dir():
        PATH_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    with open(PATH_CONFIG, "w") as f:
        toml.dump(default_config, f)

    config = default_config

# Create a 'logs' folder if it doesn't exist
log_dir = Path(user_log_dir("phageid"))
if not log_dir.is_dir():
    log_dir.mkdir(parents=True)

try:
    log_level = getattr(logging, config["logging"]["level"].upper())
except Exception:
    err = "failed to get logging level: {}. Should be one of DEBUG, INFO, WARN, ERROR"
    logging.error(err)
    raise ValueError(err)

# Configure logging
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "dev.log"),
        logging.FileHandler(log_dir / "warnings.log"),
        logging.FileHandler(log_dir / "errors.log"),
        logging.StreamHandler(),  # Console output
    ],
)
