import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def normalize_text(text: str) -> str:
    """
    Normalize text:
    - lowercase
    - remove special characters
    - remove extra spaces
    """
    # Lowercase
    text = text.lower()
    
    # Remove everything that is not alphabet, number, or whitespace
    text = re.sub(r"[^a-z0-9\s]", "", text)
    
    # Strip by space in between words
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def extract_keywords(text: str) -> set[str]:
    """
    Remove stopwords and return keywords
    """
    words = normalize_text(text).split()
    return {w for w in words if w not in ENGLISH_STOP_WORDS}