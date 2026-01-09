import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.common.logger import get_logger

logger = get_logger(__name__)


class TextTokenizer:
    def __init__(self, max_vocab_size: int, max_seq_len: int, oov_token: str):
        logger.info(
            f"Initializing tokenizer | vocab={max_vocab_size}, seq_len={max_seq_len}"
        )
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer(
            num_words=max_vocab_size,
            oov_token=oov_token
        )

    def fit(self, texts):
        logger.info("Fitting tokenizer on training texts")
        self.tokenizer.fit_on_texts(texts)

    def transform(self, texts):
        logger.debug("Tokenizing and padding texts")
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences,
            maxlen=self.max_seq_len,
            padding="post",
            truncating="post"
        )

    def save(self, path: str):
        logger.info(f"Saving tokenizer to {path}")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        logger.info(f"Loading tokenizer from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
