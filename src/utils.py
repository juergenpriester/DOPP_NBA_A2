import numpy as np
import pandas as pd
import os


def check_create_dir(path):
    if os.path.exists(path):
        print(f"Directory {path} already exists.")
    else:
        os.makedirs(path)
        print(f"Directory {path} created.")


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
