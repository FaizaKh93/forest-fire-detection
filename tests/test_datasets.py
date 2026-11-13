# Tests du Dataset PyTorch sur de petites images créées à la volée (aucune donnée réelle nécessaire).
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torchvision import transforms

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from src.datasets import DFireDataset

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@pytest.fixture
def images(tmp_path):
    """Trois cas rencontrés dans D-Fire : JPEG paysage, PNG RGBA, JPEG portrait."""
    rng = np.random.default_rng(0)
    specs = {"landscape.jpg": ("RGB", (640, 360)), "rgba.png": ("RGBA", (100, 80)), "portrait.jpg": ("RGB", (360, 640))}
    rows = []
    for name, (mode, size) in specs.items():
        channels = 4 if mode == "RGBA" else 3
        array = rng.integers(0, 255, size=(size[1], size[0], channels), dtype=np.uint8)
        path = tmp_path / name
        Image.fromarray(array).save(path)                       # mode déduit : 3 canaux RGB, 4 RGBA
        # Chemin absolu : ROOT_DIR / chemin_absolu renvoie le chemin absolu
        rows.append({"image_id": path.stem, "path": str(path)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fast_decode", [True, False])
def test_every_image_becomes_a_3x224x224_tensor(images, fast_decode):
    dataset = DFireDataset(images, transform=PREPROCESS, fast_decode=fast_decode)
    assert len(dataset) == 3
    for idx in range(len(dataset)):
        tensor, image_id = dataset[idx]
        assert tuple(tensor.shape) == (3, IMG_SIZE, IMG_SIZE)   # RGBA ramené à 3 canaux
        assert image_id == images.loc[idx, "image_id"]


def test_without_transform_returns_rgb_pil_image(images):
    dataset = DFireDataset(images, transform=None)
    img, _ = dataset[1]                                          # PNG RGBA
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
