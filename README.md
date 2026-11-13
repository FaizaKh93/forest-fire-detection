# Détection précoce de feu et de fumée – D-Fire

Ce projet classe des images en 4 catégories : **ni feu ni fumée**, **fumée seule**, **feu seul**, **feu et fumée**. Il s'appuie sur le dataset [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset), qui compte 21 527 images sous licence CC0 1.0.

Le projet compare un apprentissage **supervisé** et **semi-supervisé**, en simulant le cas où seulement 5 % des images sont labellisées. La chaîne de traitement est la suivante :
- extraction de features avec un ResNet50 pré-entraîné ;
- réduction de dimension (PCA puis UMAP) et clustering K-Means pour produire des labels faibles ;
- entraînement de 3 modèles ConvNeXt-Tiny, suivis dans MLflow ;
- une application Streamlit de démonstration.

## Résultats

Résultats sur le jeu de test complet (4 306 images). Le F2 favorise le rappel : rater un feu coûte plus cher qu'une fausse alerte.
- **F2 de l'alerte** : alerte « fumée ou feu ».
- **F2 moyen** : moyenne des tâches fumée et feu, avec un intervalle de confiance bootstrap à 95 %.
- **AP moyenne** : moyenne des AP des deux tâches.

| Modèle | Données d'entraînement | F2 de l'alerte | F2 moyen [IC 95 %] | AP moyenne |
|---|---|---|---|---|
| **A – supervisé (champion)** | 688 images labellisées (5 % du train) | **0,910** | **0,903** [0,895 – 0,911] | **0,945** |
| B – semi-supervisé | 688 labellisées + labels faibles, puis ajustement sur les 688 | 0,858 | 0,841 [0,830 – 0,852] | 0,879 |
| C – semi-supervisé + augmentation | comme B, avec augmentation des images labellisées | 0,857 | 0,852 [0,840 – 0,862] | 0,894 |
| Référence « toujours oui » | – | 0,852 | 0,730 | 0,371 |

La référence « toujours oui » obtient déjà un F2 de l'alerte de 0,852, car 53 % des images du test contiennent du feu ou de la fumée et le F2 privilégie le rappel. L'AP, indépendante du seuil, départage donc mieux les modèles. Les détails et les limites (seuils bas, scores non calibrés) sont dans la conclusion de `notebooks/semi_supervised_models.ipynb`.

## Structure du projet

```
forest-fire-detection/
├── .github/workflows/ci.yml   # CI GitHub Actions : ruff + pytest
├── app/
│   ├── streamlit_app.py       # application de démonstration
│   ├── select_examples.py     # constitue le catalogue d'exemples
│   └── examples/              # 6 images du jeu de test + examples.csv
├── data/                      # non versionné
│   ├── D-Fire/                # dataset brut (à télécharger)
│   └── processed/             # fichiers produits par les notebooks
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing_feature_extraction.ipynb
│   ├── unsupervised_analysis.ipynb
│   ├── image_cache.ipynb
│   └── semi_supervised_models.ipynb
├── results/                   # figures versionnées (courbes, UMAP, Optuna…)
├── src/
│   ├── config.py              # chemins, constantes, choix du GPU
│   ├── datasets.py            # chargement des images
│   ├── weak_labels.py         # labels faibles par vote majoritaire dans les clusters
│   ├── cnn.py                 # ConvNeXt-Tiny, augmentation, entraînement
│   ├── metrics.py             # F2, AP, seuils, intervalles bootstrap
│   ├── tracking.py            # MLflow : tags de reproductibilité, Model Registry
│   ├── inference.py           # prédiction d'une image avec le modèle champion
│   └── champion_model.json    # lien GitHub Release, SHA256, architecture et seuils du champion
├── tests/                     # tests unitaires et test de l'application (pytest)
├── models/                    # poids des modèles A, B, C (non versionné, produit par les notebooks)
├── mlflow.db, mlruns/         # suivi MLflow local (non versionné)
├── pyproject.toml, uv.lock    # dépendances (uv)
└── .python-version            # Python 3.11
```

## Installation

**Prérequis :**
- [Git](https://git-scm.com/) ;
- [uv](https://docs.astral.sh/uv/getting-started/installation/) ;
- un **GPU NVIDIA** compatible CUDA 13 pour les notebooks d'entraînement. Il n'est pas nécessaire pour l'application ni pour les tests.

```bash
git clone https://github.com/FaizaKh93/forest-fire-detection.git
cd forest-fire-detection
uv sync
```

`uv sync` installe Python 3.11 si besoin, crée l'environnement `.venv/` et installe les versions exactes de `uv.lock`. Sous Windows et Linux, PyTorch est installé avec CUDA 13 ; cette version fonctionne aussi sans GPU, sur CPU. Sous macOS, c'est la version standard de PyPI qui est installée.

### Données (seulement pour exécuter les notebooks)

1. Télécharger D-Fire depuis le [dépôt officiel](https://github.com/gaia-solutions-on-demand/DFireDataset), section *Download*.
2. Le placer dans `data/D-Fire/` avec cette arborescence :

```
data/D-Fire/
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Tester l'application sans relancer les notebooks

Après le clonage et `uv sync`, l'application fonctionne directement, sans dataset ni GPU :

```bash
uv run streamlit run app/streamlit_app.py
```

Au premier lancement, les poids du modèle champion (environ 110 Mo) sont téléchargés depuis la [GitHub Release v1.0.0](https://github.com/FaizaKh93/forest-fire-detection/releases/tag/v1.0.0). Le fichier est mis en cache local (`~/.cache/torch/hub/checkpoints/`) et son empreinte SHA256 est vérifiée. Les seuils et l'architecture viennent de `src/champion_model.json`.

## Reproduire tout le projet

### 1. Exécuter les notebooks, dans l'ordre

| Notebook | Traitements | Sorties |
|---|---|---|
| `data_exploration` | • Lecture des images et des annotations YOLO de D-Fire<br>• Répartition des 4 catégories et des sources<br>• Analyse de la taille des images et des boîtes englobantes | • `data/processed/metadata.csv` (une ligne par image)<br>• `data/processed/boxes.csv` (une ligne par boîte) |
| `preprocessing_feature_extraction` | • Split figé : train 13 773 / val 3 448 / test 4 306 images<br>• Validation (20 % du train) stratifiée et groupée par quasi-doublons (dHash), contre les fuites de données<br>• **Test propre** : 2 192 images sans quasi-doublon dans le train<br>• Extraction des features ResNet50 ImageNet (2 048 dimensions) | • `data/processed/splits.csv`<br>• `data/processed/features/resnet50_features.npy` + `resnet50_metadata.csv` |
| `unsupervised_analysis` | • Simulation de 5 % de données labellisées (688 images)<br>• Réduction de dimension : StandardScaler, PCA à 950 composantes (90 % de variance), UMAP en 10 dimensions<br>• Comparaison de K-Means, DBSCAN et HDBSCAN : K-Means K = 64 retenu<br>• Labels faibles par vote majoritaire des images labellisées de chaque cluster<br>• Suivi des résultats dans MLflow | • `data/processed/weak_labels.csv`<br>• `data/processed/clustering_comparison.csv`<br>• Figures dans `results/` (variance PCA, UMAP, clusters, comparaison) |
| `image_cache` | • Prétraitement unique de toutes les images en 224×224 (redimensionnement sans recadrage)<br>• Stockage en uint8 pour un entraînement GPU beaucoup plus rapide | • `data/processed/cache/images_224.npy` (environ 3 Go) + index |
| `semi_supervised_models` | • Réglage des hyperparamètres avec Optuna (20 essais, AP moyenne sur val)<br>• Entraînement de 3 modèles ConvNeXt-Tiny :<br>&nbsp;&nbsp;– **A** sur les 688 images labellisées<br>&nbsp;&nbsp;– **B** sur les labels faibles, puis ajustement sur les 688<br>&nbsp;&nbsp;– **C** comme B, avec augmentation<br>• Seuils fixés par F2 sur val<br>• Évaluation unique sur le test complet et le test propre (intervalles bootstrap, référence « toujours oui »)<br>• Enregistrement dans le Model Registry MLflow, alias `champion` pour A | • `models/model_{A,B,C}.pt`<br>• `data/processed/model_comparison.csv`<br>• Figures dans `results/` (courbes d'apprentissage, précision-rappel, comparaison, Optuna)<br>• Runs et modèles dans MLflow |

Le noyau Jupyter à sélectionner est celui de `.venv`. Le dernier notebook enregistre les modèles dans MLflow et donne l'alias `champion` au modèle A.

### 2. Lancer l'application Streamlit

```bash
uv run streamlit run app/streamlit_app.py
```

L'application s'ouvre sur http://localhost:8501. Si le registre MLflow local (`mlflow.db`) existe, elle charge le champion depuis ce registre. Sinon, elle le télécharge depuis la GitHub Release. Elle tourne sur CPU, en environ 20 ms par image.

Deux modes sont proposés :
- **Exemples** : 6 images du jeu de test, jamais vues à l'entraînement. Le bouton *Analyser* affiche l'alerte, la catégorie prédite face à la vraie catégorie, et les scores fumée et feu face à leurs seuils.
- **Import** : analyse de n'importe quelle image JPEG ou PNG.

### 3. Consulter le suivi MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

L'interface s'ouvre sur http://localhost:5000. Elle contient les runs, les métriques, les hyperparamètres Optuna et le Model Registry (`forest_fire_convnext_A`, `_B`, `_C`).

### 4. Tests et qualité du code

```bash
uv run ruff check .                                         # style et erreurs
uv run pytest                                               # tests unitaires
uv run pytest --cov=src --cov=app --cov-report=html         # rapport de couverture dans htmlcov/
```

Les tests tournent sans GPU. Le test de l'application est ignoré si `mlflow.db` est absent.

## Licence des données

D-Fire est distribué sous licence CC0 1.0. Les images de la source *PublicDataset* relèvent de licences tierces. Elles sont donc exclues du catalogue d'exemples de l'application.
