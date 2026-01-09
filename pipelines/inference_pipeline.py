"""
Warm-up inference pipeline.
Used for smoke testing and validation.
"""

from src.common.logger import setup_logging, get_logger
from src.inference.model_loader import ModelLoader
from src.inference.predict import Predictor

setup_logging()
logger = get_logger(__name__)


def run():
    logger.info("Running inference pipeline")

    loader = ModelLoader("models/registry.json")
    predictor = Predictor(
        loader.get_model(),
        loader.get_tokenizer()
    )

    sample_text = "you are useless"
    result = predictor.predict(sample_text)

    logger.info(f"Sample inference result: {result}")


if __name__ == "__main__":
    run()
