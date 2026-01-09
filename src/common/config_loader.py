import yaml
import os
from src.common.logger import get_logger

logger = get_logger(__name__)

def load_yaml_config(path: str) -> dict:
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(path)

    logger.info(f"Loading config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
