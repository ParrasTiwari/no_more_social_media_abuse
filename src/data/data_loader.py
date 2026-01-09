import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.common.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading raw dataset from {path}")

    if not os.path.exists(path):
        logger.error(f"Dataset not found at {path}")
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df


def split_and_save(
    df: pd.DataFrame,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    seed: int
):
    logger.info("Splitting dataset into train/val/test")

    train_df, temp_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=df["label"]
    )

    val_size = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_size,
        random_state=seed,
        stratify=temp_df["label"]
    )

    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    logger.info(
        f"Split complete | "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
