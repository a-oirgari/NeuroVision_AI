from functools import lru_cache
from pathlib import Path

import cv2
import mlflow
import mlflow.keras
import numpy as np

from src import config


@lru_cache(maxsize=1)
def get_champion():
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    uri = f"models:/{config.REGISTERED_MODEL}@{config.CHAMPION_ALIAS}"
    try:
        return mlflow.keras.load_model(uri)
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger '{uri}'. Le serveur MLflow est-il lancé sur {config.MLFLOW_URI} ? "
            f"({e})")


def predict_array(img_rgb):
    img = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE)).astype("float32")
    img = img[np.newaxis, ...]

    probs = get_champion().predict(img, verbose=0)[0]
    best = int(np.argmax(probs))
    return config.CLASSES[best], float(probs[best]), dict(zip(config.CLASSES, map(float, probs)))


def predict_image(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    if path.suffix.lower() not in config.VALID_EXT:
        raise ValueError(f"Extension non supportée : {path.suffix}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image illisible ou corrompue.")
    return predict_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


