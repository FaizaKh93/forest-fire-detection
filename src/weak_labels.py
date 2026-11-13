# Attribution des labels faibles par vote majoritaire dans chaque cluster (mapping cluster -> label).
# Code sorti du notebook unsupervised_analysis pour pouvoir être testé.

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.config import TASKS


def assign_weak_labels(clusters, lab_mask, y_lab):
    """Labels faibles (n, n_taches) par vote majoritaire du cluster ; -1 quand il n'y a pas de vote.

    - clusters : identifiant de cluster de chaque image (-1 = bruit)
    - lab_mask : booléen, True pour les images labellisées
    - y_lab    : vrais labels binaires des images labellisées, forme (n_labellisées, n_taches)

    Égalité de vote -> positif (précaution). Pour les images labellisées, leur propre label est exclu
    du vote (leave-one-out). Pas de vote si bruit ou si le cluster ne contient aucune autre image labellisée.
    """
    clusters = np.asarray(clusters)
    y_lab = np.asarray(y_lab)
    lab_idx = np.flatnonzero(lab_mask)
    cols = list(range(y_lab.shape[1]))

    # Somme des labels et nombre d'images labellisées du cluster de chaque image
    # (copy=True : tableaux modifiables, pandas 3 renvoie des vues en lecture seule)
    votes = pd.DataFrame(y_lab, columns=cols).assign(cluster=clusters[lab_idx])
    sums = votes.groupby("cluster")[cols].sum().reindex(clusters).fillna(0).to_numpy(copy=True)
    counts = votes.groupby("cluster").size().reindex(clusters).fillna(0).to_numpy(copy=True)

    sums[lab_idx] -= y_lab                          # leave-one-out
    counts[lab_idx] -= 1
    has_vote = (counts > 0) & (clusters != -1)      # -1 = bruit
    weak = (sums / np.maximum(counts, 1)[:, None] >= 0.5).astype(int)
    weak[~has_vote] = -1                            # pas de label faible
    return weak, has_vote


def score_tasks(y_true, y_pred, tasks=TASKS):
    """Précision, rappel et F1 par tâche, plus le F1 moyen."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for j, task in enumerate(tasks):
        p, r, f, _ = precision_recall_fscore_support(y_true[:, j], y_pred[:, j], average="binary", zero_division=0)
        out.update({f"{task}_precision": p, f"{task}_recall": r, f"{task}_f1": f})
    out["f1_mean"] = float(np.mean([out[f"{t}_f1"] for t in tasks]))
    return out
