# Tests du vote majoritaire (mapping cluster -> label faible) sur des données synthétiques.
import numpy as np
import pytest

from src.weak_labels import assign_weak_labels, score_tasks


def make_case(clusters, labeled):
    """Construit (clusters, lab_mask, y_lab) ; `labeled` = {index: [fumée, feu]}."""
    clusters = np.array(clusters)
    lab_mask = np.zeros(len(clusters), dtype=bool)
    lab_mask[list(labeled)] = True
    y_lab = np.array([labeled[i] for i in sorted(labeled)])
    return clusters, lab_mask, y_lab


def test_majority_vote_for_unlabeled_image():
    # Cluster 0 : 2 images labellisées « fumée », 1 « rien » -> l'image non labellisée reçoit « fumée »
    clusters, lab_mask, y_lab = make_case([0, 0, 0, 0], {0: [1, 0], 1: [1, 0], 2: [0, 0]})
    weak, has_vote = assign_weak_labels(clusters, lab_mask, y_lab)
    assert has_vote[3]
    assert weak[3].tolist() == [1, 0]


def test_tie_gives_positive_label():
    # Égalité 1 contre 1 sur chaque tâche -> positif (règle de précaution)
    clusters, lab_mask, y_lab = make_case([0, 0, 0], {0: [1, 1], 1: [0, 0]})
    weak, _ = assign_weak_labels(clusters, lab_mask, y_lab)
    assert weak[2].tolist() == [1, 1]


def test_leave_one_out_excludes_own_label():
    # Image 0 (fumée) votée par 1 et 2 (rien) -> 0 ; image 1 votée par 0 et 2 -> égalité -> fumée
    clusters, lab_mask, y_lab = make_case([0, 0, 0], {0: [1, 0], 1: [0, 0], 2: [0, 0]})
    weak, has_vote = assign_weak_labels(clusters, lab_mask, y_lab)
    assert has_vote.all()
    assert weak[0].tolist() == [0, 0]
    assert weak[1].tolist() == [1, 0]


def test_single_labeled_image_has_no_vote():
    # Seule image labellisée de son cluster : sans son propre vote, il ne reste personne
    clusters, lab_mask, y_lab = make_case([0, 1, 1], {0: [1, 1], 1: [0, 0]})
    weak, has_vote = assign_weak_labels(clusters, lab_mask, y_lab)
    assert not has_vote[0]
    assert weak[0].tolist() == [-1, -1]


def test_noise_points_get_no_label():
    # Le bruit (-1) ne forme pas un cluster : aucun vote, même avec des images labellisées en bruit
    clusters, lab_mask, y_lab = make_case([-1, -1, -1], {0: [1, 0], 1: [1, 0]})
    weak, has_vote = assign_weak_labels(clusters, lab_mask, y_lab)
    assert not has_vote.any()
    assert (weak == -1).all()


def test_cluster_without_labeled_image_gets_no_label():
    clusters, lab_mask, y_lab = make_case([0, 0, 1, 1], {0: [0, 1], 1: [0, 1]})
    weak, has_vote = assign_weak_labels(clusters, lab_mask, y_lab)
    assert not has_vote[2:].any()
    assert (weak[2:] == -1).all()


def test_output_shape_and_inputs_not_modified():
    clusters, lab_mask, y_lab = make_case([0, 0, 1, 1, -1], {0: [1, 0], 2: [0, 1]})
    clusters_before, y_lab_before = clusters.copy(), y_lab.copy()
    weak, has_vote = assign_weak_labels(clusters, lab_mask, y_lab)
    assert weak.shape == (5, 2)
    assert has_vote.shape == (5,)
    np.testing.assert_array_equal(clusters, clusters_before)
    np.testing.assert_array_equal(y_lab, y_lab_before)


def test_score_tasks_perfect_and_empty_predictions():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    perfect = score_tasks(y_true, y_true)
    assert perfect["f1_mean"] == pytest.approx(1.0)

    # Aucune prédiction positive : F1 = 0 sans erreur (zero_division=0)
    empty = score_tasks(y_true, np.zeros_like(y_true))
    assert empty["has_smoke_f1"] == 0.0
    assert empty["has_fire_f1"] == 0.0
