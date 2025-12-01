# src/optuna_yolo.py

import argparse
import os
from pathlib import Path

import mlflow
from mlflow import log_metric, log_param
import optuna
from ultralytics import YOLO

# On essaie de désactiver l'intégration MLflow interne d'Ultralytics
try:
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"mlflow": False})
except Exception:
    # Si l'API change ou n'existe pas, on ignore simplement
    pass


# Nom d'expérience MLflow / Optuna
EXPERIMENT_NAME = "cv_yolo_tiny_optuna"


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande pour l'étude Optuna."""
    parser = argparse.ArgumentParser(
        description="Optimisation d'hyperparamètres pour YOLO tiny avec Optuna + MLflow."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Nombre d'essais (trials) dans l'étude Optuna.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/tiny_coco.yaml",
        help="Fichier data YAML pour YOLO (dataset).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Poids YOLOv8 à utiliser comme base.",
    )
    parser.add_argument(
        "--exp-prefix",
        type=str,
        default="optuna_yolo",
        help="Préfixe pour les noms d'expériences MLflow / runs YOLO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Configuration de MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Clés de métriques renvoyées par Ultralytics (avec parenthèses)
    MAP50_KEY = "metrics/mAP50(B)"
    MAP5095_KEY = "metrics/mAP50-95(B)"
    PREC_KEY = "metrics/precision(B)"
    RECALL_KEY = "metrics/recall(B)"

    def objective(trial: optuna.Trial) -> float:
        """Fonction objectif Optuna.

        À chaque appel :
        - propose une config (epochs, imgsz),
        - lance un entraînement YOLO,
        - loggue la config + les métriques dans MLflow,
        - renvoie une métrique (à maximiser).
        """

        # SÉCURITÉ : si un run MLflow est encore actif (YOLO ou autre),
        # on le ferme avant d'en démarrer un nouveau.
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()

        # Hyperparamètres explorés par Optuna
        epochs = trial.suggest_int("epochs", 2, 5)
        imgsz = trial.suggest_categorical("imgsz", [320, 416])

        exp_name = f"{args.exp_prefix}_trial{trial.number}_e{epochs}_img{imgsz}"

        # Nouveau run MLflow pour ce trial
        with mlflow.start_run(run_name=exp_name):
            # Logs des hyperparamètres
            log_param("epochs", epochs)
            log_param("imgsz", imgsz)
            log_param("data_yaml", args.data)
            log_param("model", args.model)
            log_param("optuna_trial_number", trial.number)

            # Tag pour filtrer facilement les runs liés à cette étude Optuna
            mlflow.set_tag("optuna_study", EXPERIMENT_NAME)

            # 2) Entraînement YOLO (comme dans train_cv.py)
            model = YOLO(args.model)
            results = model.train(
                data=args.data,
                epochs=epochs,
                imgsz=imgsz,
                project="runs/train",
                name=exp_name,
            )

            # 3) Récupération des métriques Ultralytics
            try:
                # Ultralytics >= 8.3 : objet avec .results_dict
                metrics = results.results_dict
            except Exception:
                metrics = {}

            # Métrique objective à maximiser (par défaut zéro si rien trouvé)
            objective_metric = 0.0

            # mAP50 (métrique principale)
            if MAP50_KEY in metrics:
                objective_metric = float(metrics[MAP50_KEY])
                # Nom "propre" pour MLflow (sans parenthèses)
                log_metric("metrics/mAP50_B", objective_metric)

            # mAP50-95 (info complémentaire)
            if MAP5095_KEY in metrics:
                log_metric("metrics/mAP50_95_B", float(metrics[MAP5095_KEY]))

            # Précision
            if PREC_KEY in metrics:
                log_metric("metrics/precision_B", float(metrics[PREC_KEY]))

            # Rappel
            if RECALL_KEY in metrics:
                log_metric("metrics/recall_B", float(metrics[RECALL_KEY]))

            # Si aucune métrique exploitable n'est disponible, log minimal
            if not metrics:
                log_metric("training_finished", 1.0)

            # 4) Dossier de sortie YOLO
            run_dir = Path("runs/train") / exp_name
            mlflow.log_param("yolo_run_dir", str(run_dir))

        # On cherche à MAXIMISER la métrique objective_metric
        return objective_metric

    # 5) Création et exécution de l'étude Optuna
    study = optuna.create_study(
        direction="maximize",
        study_name=EXPERIMENT_NAME,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("========== Étude Optuna terminée ==========")
    print(f"Meilleure valeur (mAP50 approximatif) : {study.best_value:.4f}")
    print("Meilleurs hyperparamètres trouvés :")
    for k, v in study.best_params.items():
        print(f"  - {k} = {v}")


if __name__ == "__main__":
    main()
