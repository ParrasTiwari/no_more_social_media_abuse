import os
import json
import tensorflow as tf

from src.common.logger import get_logger
from src.features.tokenizer import TextTokenizer

logger = get_logger(__name__)


class ModelLoader:
    def __init__(self, registry_path: str, model_version: str = None):
        """
        If model_version is None → load latest
        """
        logger.info("Initializing ModelLoader")

        if not os.path.exists(registry_path):
            logger.error(f"Registry not found at {registry_path}")
            raise FileNotFoundError(registry_path)

        with open(registry_path, "r") as f:
            self.registry = json.load(f)

        if model_version is None:
            model_version = self.registry.get("latest")

        if model_version not in self.registry["models"]:
            logger.error(f"Model version {model_version} not found in registry")
            raise ValueError(f"Invalid model version: {model_version}")

        self.model_version = model_version
        self.model_path = self.registry["models"][model_version]["path"]

        logger.info(f"Selected model version: {self.model_version}")

        self.model = None
        self.tokenizer = None
        self.metadata = None

        self._load_artifacts()

    def _load_artifacts(self):
        logger.info("Loading model artifacts")

        model_dir = os.path.join(self.model_path, "model")
        tokenizer_path = os.path.join(self.model_path, "tokenizer.pkl")
        metadata_path = os.path.join(self.model_path, "metadata.json")

        self.model = tf.keras.models.load_model(model_dir)
        logger.info("TensorFlow model loaded")

        self.tokenizer = TextTokenizer.load(tokenizer_path)
        logger.info("Tokenizer loaded")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        logger.info("Metadata loaded successfully")

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_metadata(self):
        return self.metadata
