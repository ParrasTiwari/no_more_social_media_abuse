import re
import emoji
from nltk.tokenize import WhitespaceTokenizer
from src.common.logger import get_logger

logger = get_logger(__name__)

FLAGS = re.MULTILINE | re.DOTALL
TOKENIZER = WhitespaceTokenizer()


def emotize(text: str) -> str:
    logger.debug("Applying emoji normalization")
    text = emoji.replace_emoji(text, " emoji ")

    for char, rep in zip(
        "🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ):
        text = re.sub(char, rep, text, flags=FLAGS)

    return text


def text_preprocess(text: str) -> str:
    logger.debug(f"Preprocessing text: {text[:50]}")

    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = emotize(text)
    text = re_sub(r"https?:\/\/\S+|www\.\S+", "url")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "")
    text = re_sub(r"<3", "heart")
    text = re_sub(r"[A-Za-z]+[@#$%^&*()]+[A-Za-z]*", "abuse")

    tokens = TOKENIZER.tokenize(text.lower())
    processed = " ".join(tokens)

    logger.debug(f"Processed text: {processed[:5]}")
    return processed
