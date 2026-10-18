from functools import lru_cache
from pathlib import Path

import cv2
import keras
import numpy as np

from src import config


@lru_cache(maxsize=1)
def get_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {config.MODEL_PATH}. Lance d'abord l'entraînement.")
    return keras.models.load_model(config.MODEL_PATH)


def predict_array(img_gray):
    img = cv2.resize(img_gray, (config.IMG_SIZE, config.IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = img[np.newaxis, ..., np.newaxis]

    probs = get_model().predict(img, verbose=0)[0]
    best = int(np.argmax(probs))
    return config.CLASSES[best], float(probs[best]), dict(zip(config.CLASSES, map(float, probs)))


def predict_image(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    if path.suffix.lower() not in config.VALID_EXT:
        raise ValueError(f"Extension non supportée : {path.suffix}")

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image illisible ou corrompue.")
    return predict_array(img)


