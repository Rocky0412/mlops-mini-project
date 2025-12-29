import numpy as np
import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# LOGGING SETUP
# -----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/feature_engineering.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Feature engineering script started.")


# -----------------------------
# FEATURE ENGINEERING LOGIC
# -----------------------------
def feature_engineering(path: str):
    try:
        logging.info(f"Starting feature engineering on: {path}")

        test_path = os.path.join(path, "test.csv")
        train_path = os.path.join(path, "train.csv")

        # Load files
        logging.info(f"Loading train file: {train_path}")
        train_df = pd.read_csv(train_path)

        logging.info(f"Loading test file: {test_path}")
        test_df = pd.read_csv(test_path)

        # Validate required columns
        if "content" not in train_df.columns or "sentiment" not in train_df.columns:
            raise KeyError("Required columns ('content', 'sentiment') missing in train.csv")

        if "content" not in test_df.columns or "sentiment" not in test_df.columns:
            raise KeyError("Required columns ('content', 'sentiment') missing in test.csv")

        # Vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        logging.info("Initialized CountVectorizer with max_features=1000")

        # TFIDF transformation
        logging.info("Fitting vectorizer on training data...")
        X_train = vectorizer.fit_transform(train_df["content"].fillna("").values)

        logging.info("Transforming test data...")
        X_test = vectorizer.transform(test_df["content"].fillna("").values)

        # Convert to DataFrame
        X_train = pd.DataFrame(X_train.toarray())
        X_test = pd.DataFrame(X_test.toarray())

        # Append labels
        X_train["sentiment"] = train_df["sentiment"].values
        X_test["sentiment"] = test_df["sentiment"].values

        # Save outputs
        stored_path = os.path.join("data", "features")
        os.makedirs(stored_path, exist_ok=True)

        train_out = os.path.join(stored_path, "train_TFIDF.csv")
        test_out = os.path.join(stored_path, "test_TFIDF.csv")

        X_train.to_csv(train_out, index=False)
        X_test.to_csv(test_out, index=False)

        logging.info(f"Saved TFIDF train dataset: {train_out}")
        logging.info(f"Saved TFIDF test dataset: {test_out}")

        print("Feature engineering completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print("Error: One or more input CSV files are missing. Check logs.")
        raise

    except KeyError as e:
        logging.error(f"Missing column error: {e}")
        print("Error: Required column missing in processed data. Check logs.")
        raise

    except Exception as e:
        logging.error(f"Unexpected error in feature engineering: {e}")
        print("Unexpected error occurred. Check logs for more details.")
        raise


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    try:
        path = "./data/processed"
        feature_engineering(path)
    except Exception as e:
        logging.error(f"Fatal error in main(): {e}")
        raise


if __name__ == "__main__":
    main()


    
    