# Métriques d'évaluation des modèles multi-label (fumée, feu) et de l'alerte (fumée OU feu).

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import TASKS

BETA = 2                                   # F2 : le rappel compte 2 fois plus que la précision
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def tune_thresholds(y_true, probs, grid=THRESHOLD_GRID, beta=BETA):
    """Seuil de décision par tâche qui maximise le F-beta (à calculer sur val uniquement)."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    thresholds = []
    for j in range(y_true.shape[1]):
        scores = [fbeta_score(y_true[:, j], probs[:, j] >= t, beta=beta, zero_division=0) for t in grid]
        thresholds.append(float(grid[int(np.argmax(scores))]))
    return np.array(thresholds)


def binary_metrics(y_true, prob, pred, beta=BETA):
    """F-beta, rappel, précision, ROC-AUC et AP pour une tâche binaire."""
    both_classes = len(np.unique(y_true)) == 2         # AUC / AP indéfinis avec une seule classe
    return {
        "f2": fbeta_score(y_true, pred, beta=beta, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if both_classes else np.nan,
        "ap": average_precision_score(y_true, prob) if both_classes else np.nan,
    }


def evaluate_predictions(y_true, probs, thresholds, tasks=TASKS):
    """Métriques par tâche, F2 moyen, et métriques de l'alerte (fumée OU feu).

    L'alerte est positive si au moins une tâche est prédite positive ; son score est la plus
    grande des probabilités (pour ROC-AUC et AP).
    """
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    preds = (probs >= np.asarray(thresholds)).astype(int)
    out = {}
    for j, task in enumerate(tasks):
        name = task.replace("has_", "")
        for metric, value in binary_metrics(y_true[:, j], probs[:, j], preds[:, j]).items():
            out[f"{name}_{metric}"] = value
    out["f2_mean"] = float(np.mean([out[f"{t.replace('has_', '')}_f2"] for t in tasks]))
    # AP moyenne : indépendante du seuil (critère d'arrêt anticipé et d'Optuna)
    aps = [out[f"{t.replace('has_', '')}_ap"] for t in tasks]
    out["ap_mean"] = float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else np.nan
    alert = binary_metrics(y_true.max(axis=1), probs.max(axis=1), preds.max(axis=1))
    out.update({f"alert_{metric}": value for metric, value in alert.items()})
    return out


def bootstrap_ci(y_true, probs, thresholds, metric="f2_mean", n_boot=1000, seed=42, alpha=0.05):
    """Intervalle de confiance bootstrap (percentile) d'une métrique, seuils fixés."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = [evaluate_predictions(y_true[idx], probs[idx], thresholds)[metric]
              for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))
