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
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
VALID_EXT = {".jpeg", ".jpg", ".bmp", ".png"}
IMG_SIZE = 224

# Entraînement
SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3

# Partie 2 : MobileNetV2 + MLflow
CHANNELS = 3
CACHE_FILE_RGB = ROOT / "data" / "processed" / "dataset_rgb.npz"

SCRATCH_EPOCHS = 30
TL_EPOCHS, TL_LR, TL_DROPOUT = 15, 1e-3, 0.3
FT_EPOCHS, FT_LR = 20, 1e-4
FT_UNFREEZE_FROM = 100

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "neurovision-part2"
REGISTERED_MODEL = "neurovision-classifier"
CHAMPION_ALIAS = "champion"