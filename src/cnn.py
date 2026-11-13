# CNN ConvNeXt-Tiny (dernier stage dégelé + tête MLP), chargement des lots depuis le cache d'images,
# augmentation et entraînement avec arrêt anticipé.

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import v2
from tqdm import tqdm  # barres de progression en mode texte (conservées dans le notebook)

from src.config import DEVICE, IMAGENET_MEAN, IMAGENET_STD
from src.metrics import evaluate_predictions

USE_AMP = DEVICE.type == "cuda"            # précision mixte float16 uniquement sur GPU
_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

# Augmentation adaptée au feu : pas de flip vertical (ciel en bas), pas de teinte (couleur des flammes),
# luminosité modérée (images fire_only sombres), pas de recadrage (feux en bord d'image).
AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=10),
    v2.ColorJitter(brightness=0.15),
    v2.GaussianNoise(sigma=0.03),
])


@dataclass
class ImageSet:
    """Sous-ensemble d'images : positions dans le cache, labels (n, 2), images à augmenter."""
    positions: np.ndarray
    labels: np.ndarray
    augment: np.ndarray = None

    def __post_init__(self):
        if self.augment is None:
            self.augment = np.zeros(len(self.positions), dtype=bool)

    def __len__(self):
        return len(self.positions)


def build_model(hidden_dim=512, dropout=0.2, pretrained=True, n_outputs=2):
    """ConvNeXt-Tiny ImageNet : tout gelé sauf le dernier stage (features[7]) et la tête."""
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights, progress=False)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[7].parameters():         # dernier stage (équivalent de layer4)
        param.requires_grad = True
    for param in model.classifier[0].parameters():       # LayerNorm pré-entraînée de la tête
        param.requires_grad = True
    # Couche finale 768 -> 1000 classes ImageNet remplacée par une tête MLP à 2 sorties (fumée, feu)
    model.classifier[2] = nn.Sequential(
        nn.Linear(768, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_outputs)
    )
    return model


class ProbabilityModel(nn.Module):
    """Enveloppe pour le déploiement : image normalisée (n, 3, 224, 224) -> probabilités (n, 2) [fumée, feu]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def param_groups(model, lr_backbone, lr_head, weight_decay):
    """Deux taux d'apprentissage : faible pour le dernier stage, plus élevé pour la tête."""
    return [
        {"params": list(model.features[7].parameters()), "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": list(model.classifier.parameters()), "lr": lr_head, "weight_decay": weight_decay},
    ]


def augment_batch(x01):
    """Augmentation image par image (sur un lot, v2 tirerait un seul aléa pour toutes les images)."""
    return torch.stack([AUGMENT(img) for img in x01])


def load_batch(images, positions, augment_flags=None, device=DEVICE):
    """Lot uint8 du cache -> float normalisé ImageNet sur le device (augmentation éventuelle avant)."""
    batch = torch.from_numpy(np.ascontiguousarray(images[positions]))
    x = batch.to(device, non_blocking=True).float().div_(255)
    if augment_flags is not None and augment_flags.any():
        idx = torch.as_tensor(np.flatnonzero(augment_flags), device=device)
        x[idx] = augment_batch(x[idx])
    return (x - _MEAN.to(device)) / _STD.to(device)


def pos_weight_from(labels):
    """Poids des positifs par tâche = négatifs / positifs (doc BCEWithLogitsLoss)."""
    positives = labels.sum(axis=0)
    return (len(labels) - positives) / np.maximum(positives, 1)


def run_epoch(model, optimizer, scaler, loss_fn, images, data, batch_size, rng, progress=False):
    """Une époque d'entraînement ; renvoie la perte moyenne. `progress` : barre des lots."""
    model.train()
    order = rng.permutation(len(data))
    total = 0.0
    starts = range(0, len(order), batch_size)
    for start in tqdm(starts, desc="  lots", leave=False, disable=not progress):
        idx = order[start:start + batch_size]
        x = load_batch(images, data.positions[idx], data.augment[idx])
        y = torch.as_tensor(data.labels[idx], dtype=torch.float32, device=DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):   # forward + perte en float16
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(idx)
    return total / len(order)


@torch.no_grad()
def predict(model, images, positions, batch_size=128):
    """Probabilités (n, 2) par sigmoïde des sorties."""
    model.to(DEVICE).eval()
    probs = []
    for start in range(0, len(positions), batch_size):
        x = load_batch(images, positions[start:start + batch_size])
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def fit(model, images, train, val, lr_backbone, lr_head, weight_decay, max_epochs, patience,
        batch_size=64, seed=42, on_epoch=None, progress=False, desc="entraînement"):
    """Entraîne avec arrêt anticipé sur l'AP moyenne (fumée + feu) de val, indépendante du seuil.

    Restaure les poids de la meilleure époque. `on_epoch(epoch, score)` permet le pruning Optuna.
    `progress` affiche une barre des époques (perte, AP val) et une barre des lots de l'époque en cours.
    Renvoie (meilleur score, historique).
    """
    rng = np.random.default_rng(seed)
    model.to(DEVICE)                                       # avant de créer l'optimiseur
    optimizer = torch.optim.AdamW(param_groups(model, lr_backbone, lr_head, weight_decay))
    pos_weight = torch.as_tensor(pos_weight_from(train.labels), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

    best_score, best_state, wait, history = -1.0, None, 0, []
    epochs = tqdm(range(max_epochs), desc=desc, unit="époque", disable=not progress)
    for epoch in epochs:
        loss = run_epoch(model, optimizer, scaler, loss_fn, images, train, batch_size, rng, progress)
        probs = predict(model, images, val.positions)
        score = evaluate_predictions(val.labels, probs, thresholds=[0.5, 0.5])["ap_mean"]   # AP : seuil sans effet
        history.append({"epoch": epoch, "train_loss": loss, "val_ap_mean": score})
        epochs.set_postfix(perte=f"{loss:.4f}", ap_val=f"{score:.4f}", meilleure=f"{max(score, best_score):.4f}")
        if on_epoch is not None:
            on_epoch(epoch, score)
        if score > best_score:
            best_score, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    epochs.close()                                         # barre terminée même en cas d'arrêt anticipé
    model.load_state_dict(best_state)
    return best_score, history
