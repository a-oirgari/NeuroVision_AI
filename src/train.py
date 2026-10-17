import json
import time

import keras
import matplotlib.pyplot as plt

from src import config
from src.data import get_class_weights, get_splits, make_dataset
from src.model import build_model


def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_title(metric)
        ax.set_xlabel("époque")
        ax.legend()
    fig.savefig(config.REPORTS_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")


def main():
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    keras.utils.set_random_seed(config.SEED)


    X_train, y_train, X_val, y_val, X_test, y_test = get_splits()
    print("train:", X_train.shape, "| val:", X_val.shape, "| test:", X_test.shape)
    class_weights = get_class_weights(y_train)
    print("class_weight:", class_weights)

    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds = make_dataset(X_val, y_val)


    model = build_model()
    model.summary()
    try:
        keras.utils.plot_model(model, to_file=str(config.REPORTS_DIR / "model.png"),
                               show_shapes=True)
    except Exception as e:
        print("plot_model ignoré :", e)


    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.BackupAndRestore(
            backup_dir=str(config.BACKUP_DIR)),
    ]


    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    duration = time.time() - start
    print(f"Durée d'entraînement : {duration / 60:.1f} min")

    plot_curves(history)
    with open(config.REPORTS_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)


if __name__ == "__main__":
    main()