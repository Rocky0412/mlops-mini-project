import os
import requests
import yaml
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
import logging


# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
# --------------------------
# Configure Logging
# --------------------------
logging.basicConfig(
    filename="logs/data_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)



# --------------------------
# Load parameters
# --------------------------



def load_data(url: str) -> None:
    """Download dataset, clean, split, and save to data/raw with logging & exceptions."""

    try:
        logging.info(f"Starting data download from {url}")

        """response = requests.get(url)
        response.raise_for_status()  # Raise error if downloading fails
        logging.info("Data downloaded successfully")

        data_stream = StringIO(response.text)
        df = pd.read_csv(data_stream)

        
        path_url='url_data'
        df.to_csv(os.path.join(path_url,"emotion.csv")) """
        df = pd.read_csv(url)


        
        # --------------------------
        # Data Cleaning
        # --------------------------
        if "tweet_id" in df.columns:
            df.drop(columns=["tweet_id"], inplace=True)
            logging.info("Dropped 'tweet_id' column")

        before_rows = len(df)
        df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
        after_rows = len(df)
        logging.info(f"Filtered sentiment classes: {before_rows} → {after_rows} rows")

        df["sentiment"] = df["sentiment"].map({"happiness": 1, "sadness": 0})
        df = df.infer_objects(copy=False)

        # --------------------------
        # Train-test split
        # --------------------------
        
        train_data, test_data = train_test_split(
            df, test_size=0.3, random_state=42
        )
        logging.info(f"Split data: train={len(train_data)}, test={len(test_data)}")

        # --------------------------
        # Save files
        # --------------------------
        data_path = os.path.join("data", "raw")
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logging.info(f"Saved train.csv to {train_path}")
        logging.info(f"Saved test.csv to {test_path}")

        print("Data ingestion completed successfully.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error while downloading data: {e}")
        print("Error: Could not download the dataset. Check logs.")
        raise

    except FileNotFoundError as e:
        logging.error(f"File handling error: {e}")
        print("Error: File operation failed. Check logs.")
        raise

    except KeyError as e:
        logging.error(f"Missing required column: {e}")
        print("Error: Column missing in dataset. Check logs.")
        raise

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print("Error occurred during data ingestion. Check logs.")
        raise


if __name__ == "__main__":
    DATA_URL = "data/external/tweet_emotions.csv"
    #DATA_URL = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"

    load_data(DATA_URL)



