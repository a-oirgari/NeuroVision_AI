from pathlib import Path

# Chemins (relatifs à la racine du projet)
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_FILE = ROOT / "data" / "processed" / "dataset.npz"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "best_model.keras"
BACKUP_DIR = MODELS_DIR / "backup"

# Données
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # ordre alphabétique
VALID_EXT = {".jpeg", ".jpg", ".bmp", ".png"}
IMG_SIZE = 224          # taille fixe des images (224x224)

# Entraînement
SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3