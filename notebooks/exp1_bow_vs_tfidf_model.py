import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,classification_report
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import dagshub
import mlflow

df=pd.read_csv('/Users/rocky/Desktop/mlops_projects/Mini Project/mlops-mini-project/data/external/tweet_emotions.csv')
df.drop(['tweet_id'], axis=1, inplace=True)



# -----------------------------
# DOWNLOAD NLTK DATA
# -----------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# -----------------------------
# GLOBAL OBJECTS
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# TEXT CLEANING FUNCTIONS
# -----------------------------
def lemmatization(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

def remove_stopword(text):
    return " ".join(w for w in text.split() if w not in stop_words)

def remove_digit(text):
    return " ".join(w for w in text.split() if not w.isdigit())

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    text = re.sub(r'[!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~]', " ", text)
    return re.sub(r"\s+", " ", text).strip()

def removing_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def normalize_text(df):
    """Apply sequential text cleaning steps."""
    
    if "content" not in df.columns:
        raise KeyError("Input dataframe must contain a 'content' column.")
    
    df["content"] = (
        df["content"]
        .astype(str)
        .apply(lower_case)
        .apply(removing_urls)
        .apply(removing_punctuations)
        .apply(remove_digit)
        .apply(remove_stopword)
        .apply(lemmatization)
    )

    return df

df=normalize_text(df)

x=df['sentiment'].isin(['happiness','sadness'])
df=df[x]
Y=df['sentiment'].map({'sadness':0,'happiness':1})



dagshub.init(repo_owner='Rocky0412', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Rocky0412/mlops-mini-project.mlflow')
mlflow.set_experiment('BOW vs TF-IDF')

algorithm={
    'LR':LogisticRegression(),
    'DT':DecisionTreeClassifier(),
    'RT':RandomForestClassifier()
}

vectorizers={
    'TF_IDF':TfidfVectorizer(),
    'BOW':CountVectorizer(max_features=1000)
}


with mlflow.start_run(run_name='All Base Line Experment'):
    for algo_name,model in algorithm.items():
        for vec_name,vectorizer in vectorizers.items():
            with mlflow.start_run(run_name=f'{algo_name} with {vec_name}',nested=True):

                X=vectorizer.fit_transform(df['content'])
                y=df['sentiment'].map({'sadness':0,'happiness':1})
                xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)
                mlflow.log_params({
                    'vectorizer':vec_name,
                    'algorithm':algo_name,
                    'test_size':0.2
                })

                if algo_name=='LR':
                    mlflow.log_param('c',model.C)
                else :
                    mlflow.log_params({
                        'max_depth':model.max_depth
                    })
                model.fit(xtrain,ytrain)

                mlflow.log_param('model','Logistic Regression')

                #model Evalution
                y_pred=model.predict(xtest)
                mlflow.log_metric('accuracy',accuracy_score(ytest,y_pred))
                mlflow.log_metric('f1_score',f1_score(ytest,y_pred))
                mlflow.log_metric('precision',precision_score(ytest,y_pred))
                mlflow.log_metric('recall_score',recall_score(ytest,y_pred))

                #model logging
                mlflow.sklearn.log_model(model,'model')
                mlflow.log_artifact(__file__)
                #logging the dataset-> learn
                
                print(f'vectorizer = {vec_name} model = {algo_name}')
                print(classification_report(ytest,y_pred))

        