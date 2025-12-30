import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load stopwords safely
try:
    stop_words = set(stopwords.words("english"))
except:
    stop_words = set()

# Load WordNet safely
try:
    lemmatizer = WordNetLemmatizer()
    WORDNET_OK = True
except:
    WORDNET_OK = False


def lematization(text):
    """Safe lemmatization (won't crash API if WordNet fails)."""
    if not WORDNET_OK:
        return text
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())


def remove_stopword(text):
    return " ".join(w for w in text.split() if w not in stop_words)


def remove_digit(text):
    return " ".join(w for w in text.split() if not w.isdigit())


def lower_case(text):
    return text.lower()


def removing_punctuations(text):
    # Replace non-alphanumeric (except space) with space
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def removing_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def clean_text(text):
    """Apply all cleaning steps sequentially."""
    text = str(text)
    text = lower_case(text)
    text = removing_urls(text)
    text = remove_digit(text)
    text = removing_punctuations(text)
    text = remove_stopword(text)
    text = lematization(text)
    return text
