import cv2
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

from src import config


def load_images_rgb(raw_dir=config.RAW_DIR, img_size=config.IMG_SIZE):
    images, labels = [], []
    skipped = 0

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name


        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() not in config.VALID_EXT:
                skipped += 1
                continue
            try:
                img = cv2.imread(str(file), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("image illisible")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
            except Exception as e:
                print(f"[ignorée] {file.name} : {e}")
                skipped += 1
                continue

            images.append(img)
            labels.append(label)

    print(f"{len(images)} images chargées, {skipped} ignorées.")

    images, labels = shuffle(images, labels, random_state=config.SEED)
    return np.array(images, dtype="uint8"), np.array(labels)


def load_dataset_rgb(use_cache=True):
    if use_cache and config.CACHE_FILE_RGB.exists():
        data = np.load(config.CACHE_FILE_RGB)
        return data["X"], data["y"]

    X, y = load_images_rgb()
    config.CACHE_FILE_RGB.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.CACHE_FILE_RGB, X=X, y=y)
    return X, y


def encode_labels(y):
    return LabelEncoder().fit(config.CLASSES).transform(y)


def get_splits():
    X, y = load_dataset_rgb()
    y_int = encode_labels(y)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_int, test_size=0.30, stratify=y_int, random_state=config.SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=config.SEED)

    n = len(config.CLASSES)
    return (X_train, to_categorical(y_train, n),
            X_val, to_categorical(y_val, n),
            X_test, to_categorical(y_test, n))


def get_class_weights(y_train_onehot):
    y_int = np.argmax(y_train_onehot, axis=1)
    classes = np.unique(y_int)
    weights = compute_class_weight("balanced", classes=classes, y=y_int)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def make_dataset(X, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(X), seed=config.SEED)
    ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32), lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds