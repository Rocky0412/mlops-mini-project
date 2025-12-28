import dagshub # type: ignore
import mlflow # type: ignore


dagshub.init(repo_owner='Rocky0412', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Rocky0412/mlops-mini-project.mlflow')



with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
