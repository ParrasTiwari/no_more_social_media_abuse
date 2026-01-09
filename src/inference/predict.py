import time
import numpy as np

from src.common.logger import get_logger
from src.data.preprocessing import text_preprocess

logger = get_logger(__name__)


class Predictor:
    def __init__(self, model, tokenizer, threshold: float = 0.5):
        logger.info("Initializing Predictor")
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold

    def predict(self, text: str) -> dict:
        logger.info("Received inference request")
        start_time = time.time()

        # Preprocess
        processed_text = text_preprocess(text)
        logger.debug(f"Processed text: {processed_text}")

        # Tokenize
        X = self.tokenizer.transform([processed_text])

        # Predict
        prob = float(self.model.predict(X, verbose=0)[0][0])
        label = "abusive" if prob >= self.threshold else "non-abusive"

        latency_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            f"Inference completed | label={label} | prob={prob:.4f} | "
            f"latency={latency_ms}ms"
        )

        return {
            "label": label,
            "probability": prob,
            "latency_ms": latency_ms
        }
