# app/app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

from src.schema import CATEGORICAL_COLS, NUMERIC_COLS, CHOICES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", choices=CHOICES)

@app.route("/predict", methods=["POST"])
def predict():
    try: 
        data = {}
    
        for col in CATEGORICAL_COLS:
            data[col] = [request.form.get(col)]
      
        for col in NUMERIC_COLS:
            val = request.form.get(col)
            data[col] = [float(val) if val not in (None, "",) else 0.0]

        df = pd.DataFrame(data)
        pred = model.predict(df)[0]  
        proba = None

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)[0, 1] 
            except Exception:
                proba = None

        msg = f"Prediction: {pred.upper()}"
        if proba is not None:
            msg += f" (P(yes)={proba:.2%})"

        return render_template("index.html", choices=CHOICES, prediction_text=msg)
    except Exception as e:
        return render_template("index.html", choices=CHOICES, prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
