# Tests des métriques multi-label (fumée, feu) et de l'alerte.
import numpy as np
import pytest

from src.metrics import bootstrap_ci, evaluate_predictions, tune_thresholds


def test_tune_thresholds_separates_perfectly_separable_scores():
    # Positifs à 0.8, négatifs à 0.3 : tout seuil dans ]0.3, 0.8] donne F2 = 1
    y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    probs = np.where(y == 1, 0.8, 0.3)
    thresholds = tune_thresholds(y, probs)
    assert np.all((thresholds > 0.3) & (thresholds <= 0.8))
    assert evaluate_predictions(y, probs, thresholds)["f2_mean"] == pytest.approx(1.0)


def test_alert_is_positive_when_any_task_is_positive():
    y = np.array([[1, 0], [0, 1], [0, 0]])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.1]])
    metrics = evaluate_predictions(y, probs, thresholds=[0.5, 0.5])
    assert metrics["alert_recall"] == 1.0
    assert metrics["alert_precision"] == 1.0


def test_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(200, 2))
    probs = rng.random((200, 2))
    metrics = evaluate_predictions(y, probs, thresholds=[0.5, 0.5])
    for key in ["smoke_f2", "fire_f2", "smoke_ap", "fire_roc_auc", "f2_mean", "alert_f2", "alert_ap"]:
        assert 0.0 <= metrics[key] <= 1.0


def test_always_yes_baseline_values():
    # « Toujours oui » : F2 = 5p / (4p + 1) et AP = prévalence p (score constant, aucune information)
    y = np.array([[1, 1], [1, 0], [0, 0], [0, 0]])          # prévalences 0.5 et 0.25
    metrics = evaluate_predictions(y, np.ones_like(y, dtype=float), thresholds=[0.5, 0.5])
    assert metrics["smoke_f2"] == pytest.approx(5 * 0.5 / (4 * 0.5 + 1))
    assert metrics["fire_f2"] == pytest.approx(5 * 0.25 / (4 * 0.25 + 1))
    assert metrics["ap_mean"] == pytest.approx((0.5 + 0.25) / 2)


def test_auc_is_nan_when_only_one_class():
    y = np.array([[1, 0], [1, 0]])                     # feu toujours négatif
    metrics = evaluate_predictions(y, np.array([[0.9, 0.2], [0.8, 0.1]]), thresholds=[0.5, 0.5])
    assert np.isnan(metrics["fire_roc_auc"])


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=(300, 2))
    probs = np.clip(y * 0.6 + rng.random((300, 2)) * 0.5, 0, 1)
    point = evaluate_predictions(y, probs, [0.5, 0.5])["f2_mean"]
    low, high = bootstrap_ci(y, probs, [0.5, 0.5], n_boot=200)
    assert low <= point <= high
