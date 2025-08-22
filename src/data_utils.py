import os
import pandas as pd

def load_bank_marketing_df():
    local_path = "data/bank-additional.csv"
    if os.path.exists(local_path):
        return pd.read_csv(local_path, sep=";")
    else:
        raise FileNotFoundError(
            f"{local_path} not found. Please download the dataset from UCI manually."
        )
