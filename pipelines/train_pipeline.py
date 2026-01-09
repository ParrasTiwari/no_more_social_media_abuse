"""
High-level training pipeline wrapper.
This is what orchestration tools (Airflow/Prefect) will call.
"""

from src.common.logger import setup_logging, get_logger
from src.training.train import train

setup_logging()
logger = get_logger(__name__)


def run():
    logger.info("Running training pipeline")
    train()
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    run()
