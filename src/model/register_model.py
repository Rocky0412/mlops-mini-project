import mlflow
import dagshub
import yaml

dagshub.init(repo_owner='Rocky0412', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Rocky0412/mlops-mini-project.mlflow')
 


def Register_Model(model_name,model_info):
    #model_uri=f"runs:/{model_info['run_id']}/Models"
    path=model_info['path']
    reg = mlflow.register_model(model_uri=path, name=model_name)
    print(f"Model registered: version {reg.version}")


with open('reports/run.yaml', 'r') as f:
        model_info = yaml.safe_load(f)

def main():
    Register_Model('Emotion_detection_LR',model_info=model_info)




main()


