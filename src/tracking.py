# Configuration du suivi d'expériences MLflow, commune à tous les notebooks.
# Base SQLite et fichiers des runs à la racine du projet, quel que soit le dossier de lancement.

import hashlib
import platform
import subprocess

import mlflow

from src.config import MLFLOW_ARTIFACTS_DIR, MLFLOW_TRACKING_URI, ROOT_DIR


def setup_mlflow(experiment_name):
    """Active l'expérience `experiment_name` (créée si besoin) et renvoie son identifiant."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.end_run()            # ferme un run resté ouvert (ex. notebook interrompu puis relancé)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Emplacement explicite des fichiers (sinon relatif au dossier courant)
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    print(f"MLflow : expérience « {experiment_name} » | base : {MLFLOW_TRACKING_URI}")
    return experiment_id


def reproducibility_tags():
    """Commit git, empreinte de uv.lock et versions clés : de quoi refaire le run à l'identique."""
    import torch
    import torchvision

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT_DIR, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "inconnu"

    lock = ROOT_DIR / "uv.lock"
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(git("status", "--porcelain"))),       # modifications non commitées ?
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()[:16] if lock.exists() else "absent",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def log_and_register_model(model, registered_name, input_example, thresholds, tags=None):
    """Enregistre un modèle PyTorch dans le run actif et dans le Model Registry.

    - Le modèle enregistré renvoie des **probabilités** (enveloppe `ProbabilityModel`).
    - Signature (forme et type des entrées / sorties) déduite d'un exemple réel.
    - `code_paths` embarque `src/` pour pouvoir recharger le modèle ailleurs.
    - Les seuils de décision et les tags sont ajoutés à la version enregistrée.
    Renvoie (ModelInfo, numéro de version).
    """
    import torch
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    from src.cnn import ProbabilityModel

    wrapped = ProbabilityModel(model).cpu().eval()
    with torch.no_grad():
        output = wrapped(torch.from_numpy(input_example)).numpy()
    signature = infer_signature(input_example, output)

    # Format "pickle" : modèle PyTorch complet (taille de lot libre, CPU ou GPU) ; le format "pt2" par défaut
    # fige le graphe sur la forme de l'exemple (lot de 1) et sur le CPU.
    info = mlflow.pytorch.log_model(
        wrapped, name="model", signature=signature, input_example=input_example,
        registered_model_name=registered_name, code_paths=[str(ROOT_DIR / "src")],
        serialization_format="pickle",
    )
    version = str(info.registered_model_version)
    client = MlflowClient()
    version_tags = {"threshold_smoke": f"{thresholds[0]:.2f}", "threshold_fire": f"{thresholds[1]:.2f}",
                    "output": "probabilités [fumée, feu]", **(tags or {})}
    for key, value in version_tags.items():
        client.set_model_version_tag(registered_name, version, key, value)
    return info, version
