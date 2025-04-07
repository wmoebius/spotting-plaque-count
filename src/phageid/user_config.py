from pathlib import Path

import toml
from platformdirs import user_config_dir

from phageid import logging

PATH_CONFIG = Path(user_config_dir("phageid")) / "config.toml"

print(PATH_CONFIG)
if PATH_CONFIG.is_file():
    try:
        config = toml.load(PATH_CONFIG)
        logging.info(f"Loaded config from {PATH_CONFIG}")
    except Exception as e:
        logging.error(f"Failed to load config from {PATH_CONFIG}: {e}")
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
    }

    if not PATH_CONFIG.parent.is_dir():
        PATH_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(PATH_CONFIG, "w") as f:
            toml.dump(default_config, f)
    except Exception as e:
        err = "failed to create config file at {} due to {}".format(PATH_CONFIG, e)
        logging.error(err)
        raise e

    config = default_config
