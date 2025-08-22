# custom_container/program/train.py
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model_utils import save_model 


DATA_PATH = "data/bank-additional.csv"


MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")


print("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep=";")  


df['y'] = df['y'].map({'yes': 1, 'no': 0})


X = df.drop("y", axis=1)
y = df["y"]


X = pd.get_dummies(X, drop_first=True)  

print("Performing train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Evaluating model on validation set...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Saving model to {MODEL_DIR}...")
save_model(model, MODEL_DIR)

print("Training complete. Model saved!")
