# Inférence sur une image : prétraitement identique à l'entraînement, modèle champion (Model Registry MLflow
# ou GitHub Release), seuils de décision et catégorie parmi les 4 (rien, fumée, feu, feu et fumée).

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import FAST_JPEG_DECODE, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MLFLOW_TRACKING_URI, ROOT_DIR

CHAMPION_MODEL = "forest_fire_convnext_A"        # nom dans le Model Registry
CHAMPION_ALIAS = "champion"
REGISTRY_PATH = ROOT_DIR / "mlflow.db"           # registre local, créé par semi_supervised_models
# Poids publiés en GitHub Release + architecture et seuils (pour utiliser l'app sans relancer les notebooks)
CHAMPION_META_PATH = Path(__file__).with_name("champion_model.json")

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


def load_from_registry(name=CHAMPION_MODEL, alias=CHAMPION_ALIAS, device="cpu"):
    """Charge le modèle champion depuis le Model Registry MLflow, avec ses seuils (tags de la version)."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(name, alias)
    thresholds = [float(version.tags["threshold_smoke"]), float(version.tags["threshold_fire"])]
    model = mlflow.pytorch.load_model(f"models:/{name}@{alias}", map_location=device).to(device).eval()
    info = {"name": name, "version": version.version, "run_id": version.run_id, "source": "MLflow Model Registry",
            "val_ap_mean": version.tags.get("val_ap_mean"), "test_alert_f2": version.tags.get("test_alert_f2")}
    return model, thresholds, info


def load_from_release(device="cpu"):
    """Télécharge les poids du champion (GitHub Release, empreinte SHA256 vérifiée) et reconstruit le réseau."""
    from src.cnn import ProbabilityModel, build_model

    meta = json.loads(CHAMPION_META_PATH.read_text(encoding="utf-8"))
    # Mis en cache dans <torch hub>/checkpoints sous file_name : téléchargé une seule fois.
    # check_hash : le préfixe SHA256 contenu dans file_name doit correspondre au contenu (doc torch.hub)
    state_dict = torch.hub.load_state_dict_from_url(meta["url"], map_location=device, check_hash=True,
                                                    file_name=meta["file_name"], weights_only=True)
    net = build_model(hidden_dim=meta["hidden_dim"], dropout=meta["dropout"], pretrained=False)
    net.load_state_dict(state_dict)                  # strict : architecture identique à l'entraînement
    model = ProbabilityModel(net).to(device).eval()
    thresholds = [meta["threshold_smoke"], meta["threshold_fire"]]
    info = {"name": meta["name"], "version": meta["registry_version"], "run_id": meta["run_id"],
            "source": "GitHub Release", "val_ap_mean": meta["val_ap_mean"], "test_alert_f2": meta["test_alert_f2"]}
    return model, thresholds, info


def load_champion(device="cpu", source="auto"):
    """Modèle champion, seuils et infos. `auto` : registre MLflow local s'il existe, sinon GitHub Release."""
    if source == "auto":
        source = "registry" if REGISTRY_PATH.exists() else "release"
    return load_from_registry(device=device) if source == "registry" else load_from_release(device=device)
