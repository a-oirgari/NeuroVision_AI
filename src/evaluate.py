import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.data import get_splits, make_dataset


def main():
    *_, X_test, y_test = get_splits()
    model = keras.models.load_model(config.MODEL_PATH)

    test_ds = make_dataset(X_test, y_test, training=False)
    loss, acc = model.evaluate(test_ds)
    print(f"Test -> loss: {loss:.4f} | accuracy: {acc:.4f}")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(test_ds), axis=1)


    print(classification_report(y_true, y_pred, target_names=config.CLASSES))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    # Exemples corrects / incorrects
    for name, mask in [("correct", y_true == y_pred), ("incorrect", y_true != y_pred)]:
        idx = np.where(mask)[0][:8]
        if len(idx) == 0:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for ax, i in zip(axes.ravel(), idx):
            ax.imshow(X_test[i].squeeze(), cmap="gray")
            ax.set_title(f"réel: {config.CLASSES[y_true[i]]}\nprédit: {config.CLASSES[y_pred[i]]}",
                         fontsize=9)
        for ax in axes.ravel():
            ax.axis("off")
        fig.savefig(config.REPORTS_DIR / f"examples_{name}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()