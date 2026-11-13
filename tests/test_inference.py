# Tests de l'inférence (prétraitement, catégories, prédiction) et du catalogue d'exemples de l'application.
import json
import re

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.cnn import ProbabilityModel, build_model
from src.config import ROOT_DIR
from src.inference import (
    CATEGORY_LABELS,
    CHAMPION_META_PATH,
    load_champion,
    predict_image,
    preprocess_image,
    to_category,
)

EXAMPLES_DIR = ROOT_DIR / "app" / "examples"


@pytest.mark.parametrize("mode, ext", [("RGB", "jpg"), ("RGBA", "png")])
def test_preprocess_gives_normalized_3x224x224(tmp_path, mode, ext):
    channels = 4 if mode == "RGBA" else 3
    array = np.random.default_rng(0).integers(0, 255, size=(300, 500, channels), dtype=np.uint8)
    path = tmp_path / f"img.{ext}"
    Image.fromarray(array).save(path)
    tensor, display = preprocess_image(path)
    assert tensor.shape == (1, 3, 224, 224)
    assert display.mode == "RGB"
    assert tensor.min() < 0 < tensor.max()          # valeurs centrées par la normalisation ImageNet


@pytest.mark.parametrize("smoke, fire, expected", [
    (False, False, "none"), (True, False, "smoke_only"), (False, True, "fire_only"), (True, True, "fire_and_smoke"),
])
def test_to_category(smoke, fire, expected):
    assert to_category(smoke, fire) == expected
    assert expected in CATEGORY_LABELS


def test_predict_image_applies_thresholds():
    model = ProbabilityModel(build_model(hidden_dim=16, pretrained=False)).eval()
    tensor = torch.randn(1, 3, 64, 64)
    always = predict_image(model, tensor, thresholds=[0.0, 0.0])     # seuils nuls : tout est détecté
    never = predict_image(model, tensor, thresholds=[1.01, 1.01])    # seuils > 1 : rien n'est détecté
    assert always["category"] == "fire_and_smoke" and always["alert"]
    assert never["category"] == "none" and not never["alert"]
    assert 0.0 <= always["smoke_score"] <= 1.0 and 0.0 <= always["fire_score"] <= 1.0


def test_champion_meta_is_consistent():
    meta = json.loads(CHAMPION_META_PATH.read_text(encoding="utf-8"))
    # Convention torch.hub (check_hash) : nom du fichier en cache = nom-<début du SHA256>.ext
    hash_prefix = re.search(r"-([a-f0-9]{8,})\.", meta["file_name"]).group(1)
    assert meta["sha256"].startswith(hash_prefix)
    assert 0 < meta["threshold_smoke"] < 1 and 0 < meta["threshold_fire"] < 1


def test_load_from_release_rebuilds_model(monkeypatch):
    meta = json.loads(CHAMPION_META_PATH.read_text(encoding="utf-8"))
    weights = build_model(hidden_dim=meta["hidden_dim"], pretrained=False).state_dict()
    # Téléchargement simulé (pas de réseau en test) : poids d'un réseau de même architecture
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda url, **kwargs: weights)
    model, thresholds, info = load_champion(source="release")
    assert thresholds == [meta["threshold_smoke"], meta["threshold_fire"]]
    assert info["source"] == "GitHub Release"
    with torch.no_grad():
        probs = model(torch.randn(2, 3, 224, 224))
    assert probs.shape == (2, 2) and ((probs >= 0) & (probs <= 1)).all()


def test_example_catalog_covers_four_categories():
    catalog = pd.read_csv(EXAMPLES_DIR / "examples.csv")
    assert len(catalog) == 6
    assert set(catalog["category"]) == set(CATEGORY_LABELS)
    assert all((EXAMPLES_DIR / f).exists() for f in catalog["file"])
    assert "PublicDataset" not in set(catalog["source"])           # sources CC0 uniquement
