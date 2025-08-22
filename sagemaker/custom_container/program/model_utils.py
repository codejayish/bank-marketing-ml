# sagemaker/custom_container/program/model_utils.py
import os
import joblib
import pandas as pd

def save_model(model, model_dir="/opt/ml/model"):
    """Save the trained model to the specified directory"""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_dir: str):
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def input_fn(request_body, content_type="text/csv"):
    if content_type == "text/csv":
        return pd.read_csv(pd.compat.StringIO(request_body), header=None)
    elif content_type == "application/json":
        return pd.DataFrame(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(predictions, accept="application/json"):
    if accept == "application/json":
        return {"predictions": predictions.tolist()}
    elif accept == "text/csv":
        return ",".join(map(str, predictions))
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
