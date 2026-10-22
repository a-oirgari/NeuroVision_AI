import keras
import numpy as np
from keras import layers

from src import config


def _augmentation():
    return keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")


def build_scratch_cnn():

    inputs = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(config.CLASSES), activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_from_scratch")

def build_transfer_model():
    base = keras.applications.MobileNetV2(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_base",
    )
    base.trainable = False

    inputs = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS))
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.TL_DROPOUT)(x)
    outputs = layers.Dense(len(config.CLASSES), activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")

def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def unfreeze_for_finetuning(model, unfreeze_from=config.FT_UNFREEZE_FROM):
    base = model.get_layer("mobilenetv2_base")
    base.trainable = True
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base.layers[unfreeze_from:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return model

def count_parameters(model):
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    total = int(model.count_params())
    return {"total_params": total,
            "trainable_params": trainable,
            "non_trainable_params": total - trainable}