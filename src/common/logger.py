import logging
import logging.config
import yaml
from pathlib import Path


def setup_logging(
    default_path="configs/logging.yaml",
    default_level=logging.INFO,
):
    """
    Setup logging configuration.
    Creates log directories if they do not exist.
    """

    config_path = Path(default_path)

    if not config_path.exists():
        logging.basicConfig(level=default_level)
        logging.warning("Logging config file not found. Using basicConfig.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 🔑 Ensure log directories exist
    for handler in config.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
            log_path = Path(filename)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)


def get_logger(name: str):
    return logging.getLogger(name)
