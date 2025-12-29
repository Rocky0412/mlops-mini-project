import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import ssl
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------
# FIX SSL ISSUE FOR MAC
# -----------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except Exception:
    pass

# -----------------------------
# LOGGING SETUP
# -----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/data_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Starting data preprocessing script")

# -----------------------------
# DOWNLOAD NLTK DATA SAFELY
# -----------------------------
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    logging.info("Downloaded NLTK dependencies successfully")
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")
    raise


# -----------------------------
# GLOBAL OBJECTS
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------
# TEXT CLEANING FUNCTIONS
# -----------------------------
def lematization(text):
    try:
        return " ".join(lemmatizer.lemmatize(w) for w in text.split())
    except Exception as e:
        logging.error(f"Lemmatization error: {e} | Text: {text}")
        return text


def remove_stopword(text):
    try:
        return " ".join(w for w in text.split() if w not in stop_words)
    except Exception as e:
        logging.error(f"Stopword removal error: {e} | Text: {text}")
        return text


def remove_digit(text):
    try:
        return " ".join(w for w in text.split() if not w.isdigit())
    except Exception as e:
        logging.error(f"Digit removal error: {e} | Text: {text}")
        return text


def lower_case(text):
    try:
        return text.lower()
    except Exception as e:
        logging.error(f"Lowercase conversion error: {e} | Text: {text}")
        return text


def removing_punctuations(text):
    try:
        text = re.sub(r'[!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~]', " ", text)
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        logging.error(f"Punctuation removal error: {e} | Text: {text}")
        return text


def removing_urls(text):
    try:
        return re.sub(r"https?://\S+|www\.\S+", "", text)
    except Exception as e:
        logging.error(f"URL removal error: {e} | Text: {text}")
        return text


def normalize_text(df):
    """Apply sequential cleaning steps with logging."""
    try:
        logging.info("Starting text normalization")

        df["content"] = (
            df["content"]
            .astype(str)
            .apply(lower_case)
            .apply(remove_stopword)
            .apply(remove_digit)
            .apply(removing_punctuations)
            .apply(removing_urls)
            .apply(lematization)
        )

        logging.info("Completed text normalization")
        return df

    except KeyError as e:
        logging.error(f"Missing required column in input dataframe: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during normalization: {e}")
        raise


# -----------------------------
# READ + PROCESS FILE FUNCTION
# -----------------------------
def datapreprocessing(path: str, file: str):
    try:
        file_path = os.path.join(path, file)
        logging.info(f"Processing file: {file_path}")

        df = pd.read_csv(file_path)
        logging.info(f"Loaded file: {file} with {len(df)} rows")

        df = normalize_text(df)
        return df

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Empty file: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while processing {file}: {e}")
        raise


# -----------------------------
# MAIN EXECUTION LOGIC
# -----------------------------
def main():
    try:
        raw_path = "data/raw"
        processed_path = "data/processed"

        os.makedirs(processed_path, exist_ok=True)

        train_data = datapreprocessing(raw_path, "train.csv")
        test_data = datapreprocessing(raw_path, "test.csv")

        train_path = os.path.join(processed_path, "train.csv")
        test_path = os.path.join(processed_path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logging.info(f"Saved processed train to {train_path}")
        logging.info(f"Saved processed test to {test_path}")

        print("Preprocessing completed successfully.")

    except Exception as e:
        logging.error(f"Fatal error in main(): {e}")
        print("Error occurred during preprocessing. Check logs.")
        raise


if __name__ == "__main__":
    main()
