from flask import Flask, request, jsonify, render_template
import mlflow.pyfunc
import pandas as pd
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from preprocessing_utility import clean_text
import pickle

# Initialize DagsHub & MLflow
dagshub.init(repo_owner='Rocky0412', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Rocky0412/mlops-mini-project.mlflow')

# Load latest model dynamically
client = MlflowClient()
model_name = 'Emotion_detection_LR'
latest_version = client.get_latest_versions(model_name)[-1].version
model_uri = f"models:/{model_name}/{latest_version}"
model = mlflow.pyfunc.load_model(model_uri)

vectorizer= pickle.load(open('model/vectorizer.pkl','rb'))

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    data=clean_text(data['text'])
    #data cleaning
    #vectorizer
    data=vectorizer.transform([data])
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)

