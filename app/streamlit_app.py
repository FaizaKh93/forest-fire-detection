# Application Streamlit de démonstration : détection de feu et de fumée sur une image.
# Lancement (depuis la racine du projet) : uv run streamlit run app/streamlit_app.py

import os
import sys
from pathlib import Path

# L'inférence tourne aussi sur CPU : on lève l'obligation de GPU définie dans src/config.py
os.environ.setdefault("FFD_REQUIRE_GPU", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # rend le package src/ importable

import pandas as pd
import streamlit as st
from mlflow.exceptions import MlflowException

from src.inference import CATEGORY_LABELS, load_champion, predict_image, preprocess_image

EXAMPLES_DIR = ROOT / "app" / "examples"
SOURCE_EXAMPLE = "Choisir une image d'exemple (jeu de test)"
SOURCE_UPLOAD = "Importer ma propre image"

st.set_page_config(page_title="Détection feu et fumée", layout="wide")


@st.cache_resource(show_spinner="Chargement du modèle champion…")
def get_model():
    """Modèle champion du Model Registry MLflow, chargé une seule fois pour toute la session."""
    return load_champion(device="cpu")


@st.cache_data
def get_catalog():
    return pd.read_csv(EXAMPLES_DIR / "examples.csv")


st.title("Détection précoce de feu et de fumée")
st.caption("ConvNeXt-Tiny entraîné sur D-Fire – modèle « champion » du Model Registry MLflow")

try:
    model, thresholds, info = get_model()
except (MlflowException, OSError, KeyError) as exc:   # registre absent, fichiers manquants ou tags absents
    st.error("Impossible de charger le modèle champion depuis MLflow. Exécuter d'abord "
             f"`notebooks/semi_supervised_models.ipynb` (registre local `mlflow.db`).\n\nDétail : {exc}")
    st.stop()

def show_result(image_file, truth=None):
    """Prédit l'image puis affiche l'alerte, la catégorie et les scores face aux seuils."""
    tensor, _ = preprocess_image(image_file)
    result = predict_image(model, tensor, thresholds)
    label = CATEGORY_LABELS[result["category"]]
    if result["alert"]:
        st.error(f"**Alerte** – {label}")
    else:
        st.success(f"**Pas d'alerte** – {label}")
    if truth is not None:                         # vraie catégorie connue pour les exemples du test
        correct = "correcte" if truth == result["category"] else "incorrecte"
        st.write(f"Vraie catégorie : **{CATEGORY_LABELS[truth]}** (prédiction {correct})")

    st.write("**Scores du modèle** (alerte si le score dépasse le seuil) :")
    for name, key, threshold in [("Fumée", "smoke", thresholds[0]), ("Feu", "fire", thresholds[1])]:
        score = result[f"{key}_score"]
        st.progress(min(max(score, 0.0), 1.0), text=f"{name} : score {score:.3f} – seuil {threshold:.2f} → "
                                f"{'détecté' if result[key] else 'non détecté'}")


# ---- Choix de l'image ----
source = st.radio("Image à analyser", [SOURCE_EXAMPLE, SOURCE_UPLOAD], horizontal=True)

if source == SOURCE_EXAMPLE:
    analysed = st.session_state.setdefault("analysed", set())   # exemples déjà analysés (gardés entre les reruns)
    st.write("Six images du jeu de test, jamais vues pendant l'entraînement :")
    for row in get_catalog().itertuples():
        # Une ligne par image : l'image et son bouton à gauche, les scores à droite après le clic
        with st.container(border=True):
            img_col, res_col = st.columns([1, 1], vertical_alignment="center")
            with img_col:
                st.image(str(EXAMPLES_DIR / row.file), width="stretch")
                if st.button(f"Analyser {row.image_id}", key=row.image_id, width="stretch"):
                    analysed.add(row.image_id)
            with res_col:
                if row.image_id in analysed:
                    show_result(EXAMPLES_DIR / row.file, truth=row.category)
                else:
                    st.caption("Cliquer sur « Analyser » pour afficher les scores.")
else:
    uploaded = st.file_uploader("Image JPEG ou PNG", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Importer une image pour lancer l'analyse.")
    else:
        img_col, res_col = st.columns([1, 1], vertical_alignment="center")
        with img_col:
            st.image(uploaded, width="stretch")
        with res_col:
            show_result(uploaded)

with st.expander("À propos du modèle et de ses limites"):
    st.markdown(
        f"- Modèle : `{info['name']}` version {info['version']} (alias « champion »), "
        f"AP moyenne sur val {info['val_ap_mean']}, F2 de l'alerte sur le test {info['test_alert_f2']}.\n"
        "- Seuils choisis pour privilégier le rappel : rater un feu coûte plus cher qu'une fausse alerte.\n"
        "- Les scores ne sont **pas des probabilités calibrées** (modèle sur-confiant).\n"
        "- Sur le test, environ **25 %** des images sans feu ni fumée déclenchent une fausse alerte "
        "(nuages, brume, ciel couvert confondus avec de la fumée).\n"
        "- Outil d'aide à la décision : une alerte doit être vérifiée par un opérateur."
    )
