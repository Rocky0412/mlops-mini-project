import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.linear_model import LogisticRegression

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
mlflow.set_experiment('Logistic Regression and Bow')

param_grid = [
    # L2 with lbfgs
    {
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "C": [0.1, 0.2, 0.3]
    },

    # L1 with liblinear & saga
    {
        "penalty": ["l1"],
        "solver": ["liblinear", "saga"],
        "C": [0.1, 0.2, 0.3]
    },

    # ElasticNet with saga only
    {
        "penalty": ["elasticnet"],
        "solver": ["saga"],
        "l1_ratio": [0.0, 0.5, 1.0],
        "C": [0.1, 0.2, 0.3]
    }
]



vectorizer=CountVectorizer(max_features=1000)
X=vectorizer.fit_transform(df['content'])
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

with mlflow.start_run(description='Grid searchcv') as parent:
    grid_search=GridSearchCV(LogisticRegression(),param_grid=param_grid,
                             cv=5,scoring='f1',n_jobs=-1
                             )
    grid_search.fit(X=xtrain,y=ytrain)
    best_params=grid_search.best_params_
    best_score=grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric('best F1',best_score)

    print(f'Best params : {best_params}')
    print(f'Best F1 score : {best_score}' )
