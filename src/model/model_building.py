import numpy as np
import pandas as pd
import os
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import yaml


# ---------------------------------------
# LOGGING SETUP
# ---------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/model_building.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Model building script started.")


def load_parameters():
    """Load model parameters from params.yaml with logging."""
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        logging.info("Loaded parameters from params.yaml successfully")
        return params

    except FileNotFoundError:
        logging.error("params.yaml file not found.")
        raise

    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        raise


def load_training_data(path="./data/features/train_TFIDF.csv"):
    """Load training CSV with defensive checks."""
    try:
        logging.info(f"Loading training data from: {path}")

        df = pd.read_csv(path)
        logging.info(f"Loaded training dataframe with {len(df)} rows")

        if df.shape[1] < 2:
            raise ValueError("Training dataset must contain features and target label.")

        return df

    except FileNotFoundError:
        logging.error(f"Training file not found: {path}")
        raise

    except pd.errors.EmptyDataError:
        logging.error(f"Training file is empty: {path}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error loading training data: {e}")
        raise


def train_model(df, n_estimators, learning_rate):
    """Train Gradient Boosting classifier with error handling."""
    try:
        logging.info("Preparing features and labels")

        X_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values

        logging.info(
            f"Training dataset: X_shape={X_train.shape}, y_length={len(y_train)}"
        )

        logging.info(
            f"Initializing GradientBoostingClassifier (n_estimators={n_estimators}, learning_rate={learning_rate})"
        )

        model = LogisticRegression(C=0.2,penalty='elasticnet',solver='saga')

        model.fit(X_train, y_train)
        logging.info("Model training completed successfully")

        return model

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise


def save_model(model, path="models/emotion_model.pkl"):
    """Save trained model safely with logging."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved successfully at {path}")
        print("Model saved successfully!")

    except Exception as e:
        logging.error(f"Error saving model: {e}")
        print("Error: Could not save model. Check logs.")
        raise


def main():
    try:
        

        df = load_training_data()

        model = train_model(
            df,
            n_estimators=20,
            learning_rate=0.1
        )

        save_model(model)

    except Exception as e:
        logging.error(f"Fatal error in model building pipeline: {e}")
        print("Model building failed. Check logs for details.")
        raise


if __name__ == "__main__":
    main()


