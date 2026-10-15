import cv2
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

from src import config

# 1) Chargement brut des images avec OpenCV

def load_images(raw_dir=config.RAW_DIR, img_size=config.IMG_SIZE):

    images, labels = [], []
    skipped = 0

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name

        for file in class_dir.iterdir():
            if file.suffix.lower() not in config.VALID_EXT:
                skipped += 1
                continue
            try:
                img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("image illisible")
                img = cv2.resize(img, (img_size, img_size))
            except Exception as e:
                print(f"[ignorée] {file.name} : {e}")
                skipped += 1
                continue

            images.append(img)
            labels.append(label)

    print(f"{len(images)} images chargées, {skipped} ignorées.")

    images, labels = shuffle(images, labels, random_state=config.SEED)

    X = np.array(images, dtype="uint8")[..., np.newaxis]
    y = np.array(labels)
    return X, y


def load_dataset(use_cache=True):
    if use_cache and config.CACHE_FILE.exists():
        data = np.load(config.CACHE_FILE)
        return data["X"], data["y"]

    X, y = load_images()
    config.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.CACHE_FILE, X=X, y=y)
    return X, y

# 2) Étiquettes, normalisation, split

def encode_labels(y):
    encoder = LabelEncoder().fit(config.CLASSES)
    return encoder.transform(y)

def normalize(image):
    return tf.cast(image, tf.float32) / 255.0

