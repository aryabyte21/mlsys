import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str) -> set[str]:
    words = normalize_text(text).split()
    return {w for w in words if w not in ENGLISH_STOP_WORDS}
