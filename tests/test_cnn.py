# Tests du CNN ConvNeXt-Tiny (sans téléchargement des poids : pretrained=False) et de l'entraînement.
import numpy as np
import torch

from src.cnn import (
    ImageSet,
    ProbabilityModel,
    augment_batch,
    build_model,
    fit,
    load_batch,
    pos_weight_from,
    predict,
)


def test_only_last_stage_and_head_are_trainable():
    model = build_model(hidden_dim=64, pretrained=False)
    trainable = {name.split(".")[0] + "." + name.split(".")[1] for name, p in model.named_parameters()
                 if p.requires_grad}
    assert trainable == {"features.7", "classifier.0", "classifier.2"}


def test_model_outputs_two_logits():
    model = build_model(hidden_dim=64, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 2)


def test_probability_model_outputs_probabilities():
    model = build_model(hidden_dim=16, pretrained=False).eval()
    wrapped = ProbabilityModel(model).eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        probs = wrapped(x)
        assert torch.allclose(probs, torch.sigmoid(model(x)))
    assert probs.shape == (3, 2) and bool(((probs >= 0) & (probs <= 1)).all())


def test_augmentation_is_drawn_per_image():
    # 16 images identiques : avec un aléa par image, les sorties doivent différer entre elles
    x = torch.rand(1, 3, 32, 32).repeat(16, 1, 1, 1)
    out = augment_batch(x)
    assert out.shape == x.shape
    assert not all(torch.equal(out[0], out[i]) for i in range(1, 16))


def test_load_batch_normalizes_and_keeps_shape():
    images = np.full((4, 3, 8, 8), 255, dtype=np.uint8)
    x = load_batch(images, np.array([0, 2]), device=torch.device("cpu"))
    assert x.shape == (2, 3, 8, 8)
    # (1 - mean) / std pour le canal rouge
    assert torch.allclose(x[0, 0], torch.full((8, 8), (1 - 0.485) / 0.229), atol=1e-5)


def test_pos_weight_is_negatives_over_positives():
    labels = np.array([[1, 0], [0, 0], [0, 0], [1, 1]])
    np.testing.assert_allclose(pos_weight_from(labels), [1.0, 3.0])


def test_fit_runs_and_restores_best_state():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(12, 3, 32, 32), dtype=np.uint8)
    labels = rng.integers(0, 2, size=(12, 2))
    labels[0], labels[1] = [1, 1], [0, 0]                 # au moins un positif et un négatif par tâche
    train = ImageSet(np.arange(8), labels[:8], augment=np.array([True, False] * 4))
    val = ImageSet(np.arange(8, 12), np.vstack([[1, 1], [0, 0], labels[10:12]]))
    model = build_model(hidden_dim=16, pretrained=False)
    best, history = fit(model, images, train, val, lr_backbone=1e-4, lr_head=1e-3, weight_decay=1e-4,
                        max_epochs=2, patience=1, batch_size=4)
    assert 1 <= len(history) <= 2
    assert 0.0 <= best <= 1.0
    assert best == max(h["val_ap_mean"] for h in history)   # arrêt anticipé sur l'AP moyenne de val
    probs = predict(model, images, val.positions)
    assert probs.shape == (4, 2) and np.all((probs >= 0) & (probs <= 1))
