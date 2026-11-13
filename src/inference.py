# Inférence sur une image : prétraitement identique à l'entraînement, modèle champion du Model Registry,
# seuils de décision et catégorie parmi les 4 (rien, fumée, feu, feu et fumée).

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import FAST_JPEG_DECODE, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MLFLOW_TRACKING_URI

CHAMPION_MODEL = "forest_fire_convnext_A"        # nom dans le Model Registry
CHAMPION_ALIAS = "champion"

# Libellés affichés pour les 4 catégories
CATEGORY_LABELS = {
    "none": "Ni feu ni fumée",
    "smoke_only": "Fumée seule",
    "fire_only": "Feu seul",
    "fire_and_smoke": "Feu et fumée",
}

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),     # image entière, sans recadrage (comme à l'entraînement)
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(source):
    """Fichier image (chemin ou objet fichier) -> tenseur normalisé (1, 3, 224, 224) + image RGB pour l'affichage."""
    img = Image.open(source)
    if FAST_JPEG_DECODE:
        img.draft("RGB", (IMG_SIZE, IMG_SIZE))   # même décodage JPEG réduit que le cache d'entraînement
    img = img.convert("RGB")                     # PNG / RGBA ramenés à 3 canaux
    return PREPROCESS(img).unsqueeze(0), img


def to_category(smoke, fire):
    """Catégorie à partir des deux décisions binaires (fumée, feu)."""
    if smoke and fire:
        return "fire_and_smoke"
    if fire:
        return "fire_only"
    if smoke:
        return "smoke_only"
    return "none"


@torch.no_grad()
def predict_image(model, tensor, thresholds):
    """Probabilités, décisions par tâche, catégorie et alerte pour une image prétraitée.

    `model` renvoie des probabilités (enveloppe ProbabilityModel du registre).
    """
    device = next(model.parameters()).device
    probs = model(tensor.to(device)).float().cpu().numpy()[0]
    smoke, fire = (probs >= np.asarray(thresholds)).tolist()
    return {
        "smoke_score": float(probs[0]),
        "fire_score": float(probs[1]),
        "smoke": bool(smoke),
        "fire": bool(fire),
        "category": to_category(smoke, fire),
        "alert": bool(smoke or fire),
    }


def load_champion(name=CHAMPION_MODEL, alias=CHAMPION_ALIAS, device="cpu"):
    """Charge le modèle champion depuis le Model Registry MLflow, avec ses seuils (tags de la version)."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(name, alias)
    thresholds = [float(version.tags["threshold_smoke"]), float(version.tags["threshold_fire"])]
    model = mlflow.pytorch.load_model(f"models:/{name}@{alias}", map_location=device).to(device).eval()
    info = {"name": name, "version": version.version, "run_id": version.run_id,
            "val_ap_mean": version.tags.get("val_ap_mean"), "test_alert_f2": version.tags.get("test_alert_f2")}
    return model, thresholds, info
