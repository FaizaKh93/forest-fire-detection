# Datasets PyTorch du projet.
# Défini dans un module (et non dans le notebook) : sous Windows, les workers du
# DataLoader sont lancés avec spawn et doivent pouvoir importer cette classe.

from PIL import Image
from torch.utils.data import Dataset

from src.config import FAST_JPEG_DECODE, IMG_SIZE, ROOT_DIR


class DFireDataset(Dataset):
    """Renvoie (image prétraitée, image_id) pour chaque ligne du DataFrame."""

    def __init__(self, dataframe, transform=None, fast_decode=FAST_JPEG_DECODE):
        # Seules les colonnes utiles sont gardées (objet léger à transmettre aux workers)
        self.paths = dataframe["path"].tolist()
        self.ids = dataframe["image_id"].tolist()
        self.transform = transform
        self.fast_decode = fast_decode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(ROOT_DIR / self.paths[idx])
        if self.fast_decode:
            # JPEG décodé à la plus petite échelle >= IMG_SIZE (sans effet sur les PNG)
            img.draft("RGB", (IMG_SIZE, IMG_SIZE))
        # convert("RGB") : 3 canaux garantis (PNG et RGBA inclus)
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.ids[idx]
