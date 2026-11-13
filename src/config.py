# Configuration centralisée du projet : chemins, constantes et matériel de calcul.
# Chaque notebook importe ce fichier pour éviter de redéfinir les mêmes valeurs.

import os
from pathlib import Path

import torch

# ---- Chemins ----
# Racine du projet : remonte d'un niveau depuis src/
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
DFIRE_DIR = DATA_DIR / "D-Fire"                  # Dataset brut (non versionné)
PROCESSED_DIR = DATA_DIR / "processed"           # Fichiers produits par les notebooks
FEATURES_DIR = PROCESSED_DIR / "features"        # Vecteurs de features (.npy)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = PROCESSED_DIR / "metadata.csv"   # data_exploration : une ligne par image
BOXES_PATH = PROCESSED_DIR / "boxes.csv"         # data_exploration : une ligne par boîte
SPLITS_PATH = PROCESSED_DIR / "splits.csv"       # preprocessing_feature_extraction : split figé
FEATURES_PATH = FEATURES_DIR / "resnet50_features.npy"          # matrice (n_images, 2048)
FEATURES_META_PATH = FEATURES_DIR / "resnet50_metadata.csv"     # même ordre que la matrice
WEAK_LABELS_PATH = PROCESSED_DIR / "weak_labels.csv"            # unsupervised_analysis : labels faibles
CLUSTERING_COMPARISON_PATH = PROCESSED_DIR / "clustering_comparison.csv"
RESULTS_DIR = ROOT_DIR / "results"               # Figures sauvegardées
RESULTS_DIR.mkdir(exist_ok=True)

# Cache d'images prétraitées (image_cache) : uint8 (n_images, 3, 224, 224), ordre = IMAGE_CACHE_INDEX_PATH
CACHE_DIR = PROCESSED_DIR / "cache"
IMAGE_CACHE_PATH = CACHE_DIR / "images_224.npy"
IMAGE_CACHE_INDEX_PATH = CACHE_DIR / "images_224_index.csv"
MODELS_DIR = ROOT_DIR / "models"                 # Poids des modèles entraînés (non versionnés)
MODEL_COMPARISON_PATH = PROCESSED_DIR / "model_comparison.csv"

# ---- Suivi d'expériences (MLflow) ----
# Base SQLite (backend par défaut de MLflow 3) et fichiers des runs, à la racine du projet
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS_DIR = ROOT_DIR / "mlruns"

# ---- Dataset ----
CLASS_NAMES = {0: "smoke", 1: "fire"}            # Convention D-Fire (vérifiée dans data_exploration)
CATEGORY_ORDER = ["smoke_only", "fire_only", "fire_and_smoke", "none"]
TASKS = ["has_smoke", "has_fire"]                # Deux tâches binaires (multi-label)

# ---- Reproductibilité ----
RANDOM_STATE = 42                                # Même graine dans tous les notebooks

# ---- Semi-supervisé ----
# Part du train qui garde ses vrais labels (le reste est traité comme non labellisé).
# P7 : 100 images labellisées sur 1 506 (~6,6 %).
LABELED_FRACTION = 0.05

# ---- Preprocessing (ResNet50 ImageNet) ----
IMG_SIZE = 224                                   # Taille d'entrée de ResNet50
IMAGENET_MEAN = [0.485, 0.456, 0.406]            # Statistiques ImageNet (doc torchvision)
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---- Matériel de calcul : GPU obligatoire ----
# On refuse de basculer silencieusement sur le CPU (calculs beaucoup plus lents).
# Exception : FFD_REQUIRE_GPU=0 (CI GitHub Actions et tests unitaires, machines sans GPU).
REQUIRE_GPU = os.environ.get("FFD_REQUIRE_GPU", "1") != "0"
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "Aucun GPU CUDA détecté. Vérifier que le kernel du notebook utilise le .venv du projet "
        "(torch installé depuis l'index CUDA, cf. pyproject.toml)."
    )
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Chargement des images (DataLoader) ----
# Le goulot est le décodage des JPEG sur CPU, pas le GPU.
# Mesures sur 2 560 images (RTX 5070 Laptop, ResNet50, batch 64) :
#   décodage complet : 86 img/s (0 worker) | 102 (4) | 57 (8) | 39 (12)
#   décodage réduit  : 171 img/s (0 worker) | 107 (4)
# -> décodage JPEG réduit + chargement dans le processus principal.
BATCH_SIZE = 64
NUM_WORKERS = 0                                  # workers plus lents sur cette machine (cf. mesures)
PIN_MEMORY = DEVICE.type == "cuda"               # transfert CPU -> GPU plus rapide
PERSISTENT_WORKERS = NUM_WORKERS > 0             # utile seulement si NUM_WORKERS > 0

# Décodage JPEG directement à taille réduite (Image.draft, JPEG uniquement) :
# 2x plus rapide, features quasi identiques (cosinus moyen 0,998 vs décodage complet).
# Appliqué à toutes les images de tous les splits -> preprocessing identique.
FAST_JPEG_DECODE = True


def describe_device():
    """Affiche le matériel utilisé (à appeler en début de notebook)."""
    print(f"PyTorch {torch.__version__} | appareil : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU : {props.name} | mémoire : {props.total_memory / 1024**3:.1f} Go")
    print(f"DataLoader : batch {BATCH_SIZE}, {NUM_WORKERS} workers, pin_memory={PIN_MEMORY}, "
          f"décodage JPEG réduit={FAST_JPEG_DECODE}")
