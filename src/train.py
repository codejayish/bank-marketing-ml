# src/train.py
import os
import json
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.model_selection import train_test_split

from joblib import dump

from .schema import TARGET_COL, CATEGORICAL_COLS, NUMERIC_COLS
from .data_utils import load_bank_marketing_df


try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    warnings.warn("xgboost not available; skipping XGBClassifier.")


def build_preprocessor() -> ColumnTransformer:
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_COLS),
            ("num", num_pipe, NUMERIC_COLS),
        ]
    )
    return pre


def train_models(X_train, y_train) -> Dict[str, Pipeline]:
    pre = build_preprocessor()

    models: Dict[str, Pipeline] = {}

    # Logistic Regression
    lr = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    lr.fit(X_train, y_train)
    models["logreg"] = lr

    # Random Forest
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42,
            class_weight="balanced"
        ))
    ])
    rf.fit(X_train, y_train)
    models["rf"] = rf

    # XGBoost (needs numeric labels)
    if HAS_XGB:
        y_train_num = (y_train == "yes").astype(int)

        xgb = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="logloss"
            ))
        ])
        xgb.fit(X_train, y_train_num)
        models["xgb"] = xgb

    return models


def evaluate(model: Pipeline, X, y) -> Dict[str, Any]:
    """
    Evaluates a model's predictions against true labels.
    Handles both string and numeric predictions by standardizing them.
    """
    y_pred = model.predict(X)

   
    if np.issubdtype(y_pred.dtype, np.number):
        y_pred = np.where(y_pred == 1, "yes", "no")

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_yes": float(precision_score(y, y_pred, pos_label="yes")),
        "recall_yes": float(recall_score(y, y_pred, pos_label="yes")),
        "f1_yes": float(f1_score(y, y_pred, pos_label="yes")),
        "report": classification_report(y, y_pred, output_dict=False)
    }



def main():
    print(">>> Loading dataset from UCI...")
    df = load_bank_marketing_df()
    print("Shape:", df.shape)


    missing = set(CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = df[TARGET_COL].astype(str)  

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full
    )
    print(f"Splits -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    print(">>> Training models...")
    models = train_models(X_train, y_train)

    print(">>> Evaluating on validation set...")
    val_scores = {name: evaluate(m, X_val, y_val) for name, m in models.items()}
    for name, sc in val_scores.items():
        print(f"\nModel: {name}")
        print(f"F1(yes)={sc['f1_yes']:.4f}  Acc={sc['accuracy']:.4f}  "
              f"P={sc['precision_yes']:.4f}  R={sc['recall_yes']:.4f}")
        print(sc["report"])

    
    best_name = max(val_scores, key=lambda k: val_scores[k]["f1_yes"])
    best_model = models[best_name]
    print(f"\n>>> Best on validation: {best_name}")

    print(">>> Final evaluation on test set...")
    test_metrics = evaluate(best_model, X_test, y_test)
    print(f"Test F1(yes)={test_metrics['f1_yes']:.4f}  Acc={test_metrics['accuracy']:.4f}")
    print(test_metrics["report"])

    os.makedirs("models", exist_ok=True)
    dump(best_model, "models/model.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump({
            "chosen_model": best_name,
            "val_metrics": val_scores[best_name],
            "test_metrics": test_metrics
        }, f, indent=2)

    print(">>> Saved best model to models/model.pkl")
    print(">>> Metrics snapshot to models/metrics.json")


if __name__ == "__main__":
    main()
