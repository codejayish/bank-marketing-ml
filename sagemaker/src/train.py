import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def model_fn(model_dir):
    """Load model for inference."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def main():
    parser = argparse.ArgumentParser()

    # SageMaker passes these automatically
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args = parser.parse_args()

    # Load training data
    data = pd.read_csv(os.path.join(args.train, "train.csv"))
    X = data.drop("y", axis=1)
    y = data["y"]

    # Train logistic regression
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)

    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

if __name__ == "__main__":
    main()
