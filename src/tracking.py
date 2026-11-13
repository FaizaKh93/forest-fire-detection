# Configuration du suivi d'expériences MLflow, commune à tous les notebooks.
# Base SQLite et fichiers des runs à la racine du projet, quel que soit le dossier de lancement.

import mlflow

from src.config import MLFLOW_ARTIFACTS_DIR, MLFLOW_TRACKING_URI


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
