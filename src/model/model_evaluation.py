import os
import pickle
import yaml
import logging
import pandas as pd
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    roc_auc_score
)
import dagshub # type: ignore
import mlflow # type: ignore
import yaml


dagshub.init(repo_owner='Rocky0412', repo_name='mlops-mini-project', mlflow=True)


# -----------------------------
# LOGGING SETUP
# -----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/model_evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Model evaluation script started.")


def load_model(path="./models/emotion_model.pkl"):
    try:
        logging.info(f"Loading model from: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def load_test_data(path="./data/features/test_TFIDF.csv"):
    try:
        logging.info(f"Loading test data from: {path}")
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("Test dataset must contain features and target label.")
        logging.info(f"Test data loaded with {len(df)} rows")
        return df
    except FileNotFoundError:
        logging.error(f"Test file not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise


def evaluate_model(model, X, y):
    try:
        logging.info("Starting model evaluation")
        y_pred = model.predict(X)

        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="weighted")
        recall = recall_score(y, y_pred, average="weighted")

        # ROC-AUC (binary only)
        if hasattr(model, "predict_proba") and y.nunique() == 2:
            roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        else:
            roc_auc = None
            logging.warning("ROC-AUC not computed (non-binary labels or missing predict_proba)")

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "roc_auc_score": float(roc_auc) if roc_auc is not None else None,
        }

        logging.info(f"Evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise


def save_metrics(metrics, path="./evaluation/metrics.yaml"):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(metrics, f)
        logging.info(f"Metrics saved successfully at {path}")
        print(f"Metrics saved to {path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise


def main():
    try:
        
        mlflow.set_tracking_uri('https://dagshub.com/Rocky0412/mlops-mini-project.mlflow')
        mlflow.set_experiment('DVC-Pipeline-Emotion Detection')
        with mlflow.start_run() as run:
            model = load_model()
            test_df = load_test_data()

            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            metrics = evaluate_model(model, X_test, y_test)
            save_metrics(metrics)

            mlflow.log_metrics(metrics)
            params = model.get_params()
            params={k:v for k,v in params.items()}
            mlflow.log_params(params)
            mlflow.sklearn.log_model(model,"model")
            #Get the model path
            artifact_path = mlflow.get_artifact_uri("model")
            print("Artifact Path:", artifact_path)
            mlflow.log_artifact('evaluation/metrics.yaml')
            mlflow.log_artifact("data/external/tweet_emotions.csv")
            run_id=run.info.run_id
            model_details= {
                'run_id':run_id,
                'path':artifact_path
            }
            with open('./reports/run.yaml','w') as f:
                yaml.dump(model_details,f,indent=4)


    except Exception as e:
        logging.error(f"Fatal error in model evaluation pipeline: {e}")
        print("Model evaluation failed. Check logs for details.")
        raise


if __name__ == "__main__":
    main()


