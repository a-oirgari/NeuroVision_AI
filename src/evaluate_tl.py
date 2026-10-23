import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src import config


def evaluate_on_test(model, X_test, y_test, test_ds, run_id=None):
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    print(classification_report(y_true, y_pred, target_names=config.CLASSES, digits=3))


    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion (test)")

    # Faux négatifs graves : une vraie tumeur prédite "notumor"
    tumor_ids = [config.CLASSES.index(c) for c in ("glioma", "meningioma", "pituitary")]
    notumor_id = config.CLASSES.index("notumor")
    is_tumor = np.isin(y_true, tumor_ids)
    missed = int(np.sum(is_tumor & (y_pred == notumor_id)))
    missed_rate = missed / int(is_tumor.sum())
    print(f"Tumeurs prédites 'notumor' (faux négatifs graves) : "
          f"{missed}/{int(is_tumor.sum())} ({missed_rate:.2%})")

    # Quelques erreurs
    wrong = np.where(y_true != y_pred)[0][:8]
    fig_err = None
    if len(wrong):
        fig_err, axes = plt.subplots(2, 4, figsize=(12, 6))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), wrong):
            ax.imshow(X_test[i].astype("uint8"))
            ax.set_title(f"réel: {config.CLASSES[y_true[i]]}\nprédit: {config.CLASSES[y_pred[i]]}",
                         fontsize=9)


    recalls = cm.diagonal() / cm.sum(axis=1)

    metrics = {
        "test_accuracy": float(np.mean(y_true == y_pred)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "test_tumor_missed_rate": float(missed_rate),
        **{
            f"test_recall_{c}": float(r)
            for c, r in zip(config.CLASSES, recalls)
        },
    }

    print(metrics)

