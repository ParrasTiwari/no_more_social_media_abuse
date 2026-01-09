import os
import json
import shutil
import subprocess
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.common.logger import setup_logging, get_logger
from src.common.config_loader import load_yaml_config
from src.data.data_loader import load_raw_data, split_and_save
from src.data.preprocessing import text_preprocess
from src.features.tokenizer import TextTokenizer

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
setup_logging()
logger = get_logger(__name__)


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def build_model(vocab_size: int, cfg: dict) -> tf.keras.Model:
    logger.info("Building BiLSTM model")

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=cfg["model"]["embedding_dim"],
            input_length=cfg["tokenizer"]["max_seq_len"]
        ),
        Bidirectional(
            LSTM(cfg["model"]["lstm_units"], return_sequences=False)
        ),
        Dropout(cfg["model"]["dropout"]),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------
def train():
    logger.info("========== TRAINING PIPELINE STARTED ==========")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_yaml_config("configs/training.yaml")

    model_version = cfg["model"]["version"]
    model_dir = os.path.join(cfg["paths"]["model_dir"], model_version)
    os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and split data
    # ------------------------------------------------------------------
    raw_df = load_raw_data(cfg["paths"]["raw_data_path"])

    split_and_save(
        df=raw_df,
        output_dir=cfg["paths"]["processed_data_dir"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["random_seed"]
    )

    # Load processed data
    logger.info("Loading processed datasets")
    train_df = load_raw_data(
        os.path.join(cfg["paths"]["processed_data_dir"], "train.csv")
    )
    val_df = load_raw_data(
        os.path.join(cfg["paths"]["processed_data_dir"], "val.csv")
    )

    # ------------------------------------------------------------------
    # Preprocess text
    # ------------------------------------------------------------------
    logger.info("Applying text preprocessing")

    X_train_text = train_df["text"].apply(text_preprocess).tolist()
    X_val_text = val_df["text"].apply(text_preprocess).tolist()

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    tokenizer = TextTokenizer(
        max_vocab_size=cfg["tokenizer"]["max_vocab_size"],
        max_seq_len=cfg["tokenizer"]["max_seq_len"],
        oov_token=cfg["tokenizer"]["oov_token"]
    )

    tokenizer.fit(X_train_text)

    X_train = tokenizer.transform(X_train_text)
    X_val = tokenizer.transform(X_val_text)

    vocab_size = min(
        cfg["tokenizer"]["max_vocab_size"],
        len(tokenizer.tokenizer.word_index) + 1
    )

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    model = build_model(vocab_size, cfg)

    logger.info("Starting model training")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=cfg["training"]["early_stopping_patience"],
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    logger.info("Saving trained model")
    model.save(os.path.join(model_dir, "model"))

    tokenizer.save(os.path.join(model_dir, "tokenizer.pkl"))

    # Metrics
    metrics = {
        "train_accuracy": float(max(history.history["accuracy"])),
        "val_accuracy": float(max(history.history["val_accuracy"])),
        "val_loss": float(min(history.history["val_loss"]))
    }

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Immutable config snapshot
    shutil.copy(
        "configs/training.yaml",
        os.path.join(model_dir, "training_config.yaml")
    )

    # Metadata
    metadata = {
        "model_version": model_version,
        "framework": "tensorflow",
        "model_type": "BiLSTM",
        "dataset": os.path.basename(cfg["paths"]["raw_data_path"]),
        "trained_at": datetime.utcnow().isoformat(),
        # "git_commit": get_git_commit(),
        "python_version": os.sys.version
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # ------------------------------------------------------------------
    # Update registry
    # ------------------------------------------------------------------
    registry_path = os.path.join(cfg["paths"]["model_dir"], "registry.json")

    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)

    registry.setdefault("models", {})
    registry["models"][model_version] = {
        "path": model_dir,
        "val_accuracy": metrics["val_accuracy"],
        "created_at": metadata["trained_at"]
    }

    registry["latest"] = model_version

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    logger.info("========== TRAINING PIPELINE COMPLETED ==========")


if __name__ == "__main__":
    train()
