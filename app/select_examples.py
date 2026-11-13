# Constitue le catalogue d'images d'exemple de l'application (6 images du test « propre »).
# Usage (depuis la racine du projet) : uv run python app/select_examples.py

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RANDOM_STATE, ROOT_DIR, SPLITS_PATH

EXAMPLES_DIR = ROOT_DIR / "app" / "examples"
MAX_FILE_KB = 400                          # images légères pour le dépôt git
# Nombre d'exemples par catégorie et par source (sources CC0 uniquement : PublicDataset a ses propres licences)
PLAN = [
    ("none", "WEB", 1),
    ("smoke_only", "AoF", 1),
    ("smoke_only", "WEB", 1),
    ("fire_only", "WEB", 1),
    ("fire_and_smoke", "AoF", 1),
    ("fire_and_smoke", "WEB", 1),
]


def main():
    splits = pd.read_csv(SPLITS_PATH)
    # Test « propre » : images sans quasi-doublon dans le train (démo honnête)
    pool = splits[(splits["final_split"] == "test") & splits["test_clean"].eq(True)].copy()
    pool["size_kb"] = [(ROOT_DIR / p).stat().st_size / 1024 for p in pool["path"]]
    pool = pool[pool["size_kb"] <= MAX_FILE_KB]

    rows = []
    for category, source, n in PLAN:
        candidates = pool[(pool["category"] == category) & (pool["source"] == source)]
        rows.append(candidates.sample(n, random_state=RANDOM_STATE))
    examples = pd.concat(rows)

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in EXAMPLES_DIR.glob("*.jpg"):
        old.unlink()                           # catalogue régénéré à l'identique
    for row in examples.itertuples():
        shutil.copy(ROOT_DIR / row.path, EXAMPLES_DIR / f"{row.image_id}.jpg")

    catalog = examples[["image_id", "source", "category", "has_smoke", "has_fire"]].assign(
        file=examples["image_id"] + ".jpg")
    catalog.to_csv(EXAMPLES_DIR / "examples.csv", index=False)
    print(catalog.to_string(index=False))


if __name__ == "__main__":
    main()
