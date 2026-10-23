import time

import keras
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import f1_score

from src import config
from src.model_tl import count_parameters


class MlflowEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_title(metric)
        ax.set_xlabel("époque")
        ax.legend()
    return fig


def train_and_log(model, run_name, train_ds, val_ds, y_val, class_weights,
                  epochs, learning_rate, patience=5, extra_params=None):

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = config.MODELS_DIR / f"{run_name}.keras"
    backup = config.MODELS_DIR / f"backup_{run_name}"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt),
            monitor="val_loss",
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        ),
        keras.callbacks.BackupAndRestore(
            backup_dir=str(backup)
        ),
        MlflowEpochLogger(),
    ]

    summary_lines = []
    model.summary(print_fn=summary_lines.append)

    with mlflow.start_run(run_name=run_name):
        # hyperparamètres
        mlflow.log_params({
            "model": model.name, "learning_rate": learning_rate, "epochs_max": epochs,
            "batch_size": config.BATCH_SIZE, "img_size": config.IMG_SIZE,
            "optimizer": "Adam", "loss": "categorical_crossentropy",
            **count_parameters(model), **(extra_params or {}),
        })
        mlflow.log_text("\n".join(summary_lines), "model_summary.txt")

        # entraînement chronométré
        start = time.time()
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                            class_weight=class_weights, callbacks=callbacks)
        duration_min = (time.time() - start) / 60
        print(f"Durée d'entraînement : {duration_min:.1f} min")

        # métriques du MEILLEUR modèle sur la validation
        best_model = keras.models.load_model(ckpt)
        val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(best_model.predict(val_ds, verbose=0), axis=1)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Validation -> loss {val_loss:.4f} | acc {val_acc:.4f} | F1 macro {val_f1:.4f}")

        mlflow.log_metrics({"duration_min": duration_min, "best_val_loss": val_loss,
                            "best_val_accuracy": val_acc, "best_val_f1_macro": val_f1})
        mlflow.log_param("epochs_run", len(history.history["loss"]))

        # artefacts : courbes + modèle
        mlflow.log_figure(plot_curves(history), "learning_curves.png")
        info = mlflow.keras.log_model(best_model, name="model")
        mlflow.set_tag("model_uri", info.model_uri)       # retrouvé par tracking.py

    return best_model, history