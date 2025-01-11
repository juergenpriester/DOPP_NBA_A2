import numpy as np
import os

def check_create_dir(path):
    if os.path.exists(path):
        print(f"Directory {path} already exists.")
    else:
        os.makedirs(path)
        print(f"Directory {path} created.")