import os
import joblib
import pandas as pd

def model_fn(model_dir):
    """Load model for inference"""
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, request_content_type):
    """Deserialize request"""
    if request_content_type == "text/csv":
        return pd.DataFrame([x.split(",") for x in request_body.split("\n") if x])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    """Serialize prediction"""
    if response_content_type == "text/csv":
        return ",".join(str(x) for x in prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
