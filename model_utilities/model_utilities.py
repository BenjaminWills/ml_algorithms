import pickle
import sys
import os

from datetime import datetime


def bytesto(file_size: float, to: str, bsize: int = 1024) -> float:
    a = {"kb": 1, "mb": 2, "gb": 3, "tb": 4, "pb": 5, "eb": 6}
    r = float(file_size)
    return round(r / (bsize ** a[to]), 2)


def get_save_model_stats(model: object):
    print(
        f"""
    Saving model...
    """
    )


def get_load_model_stats(load_path: str):
    print(
        f"""
    Model size in mb   : {bytesto(os.path.getsize(load_path),'mb')}
    Model modifed on   : {datetime.fromtimestamp(os.path.getmtime(load_path)).strftime("%A, %B %d, %Y %I:%M:%S")}
    Model created on   : {datetime.fromtimestamp(os.path.getctime(load_path)).strftime("%A, %B %d, %Y %I:%M:%S")}
    """
    )


def save_model(model: object, save_path: str):
    get_save_model_stats(model)
    with open(save_path, "wb") as file:
        pickle.dump(model, file)


def load_model(load_path: str):
    get_load_model_stats(load_path)
    with open(load_path, "rb") as file:
        return pickle.load(file)
