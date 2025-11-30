# src/optuna_yolo.py

import argparse
import os
from pathlib import Path

import mlflow
from mlflow import log_metric, log_param
import optuna
from ultralytics import YOLO

EXPERIMENT_NAME = "cv_yolo_tiny_optuna"


def parse_args() -> argparse.Namespace:
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

    def objective(trial: optuna.Trial) -> float:
        """Fonction objectif Optuna.

        À chaque appel :
        - propose une config (epochs, imgsz),
        - lance un entraînement YOLO,
        - logge la config + les métriques dans MLflow,
        - renvoie une métrique (à maximiser).
        """
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

            # Tag pour retrouver facilement les runs liés à Optuna
            mlflow.set_tag("optuna_study", EXPERIMENT_NAME)

            # Entraînement YOLO (comme dans train_cv.py)
            model = YOLO(args.model)
            results = model.train(
                data=args.data,
                epochs=epochs,
                imgsz=imgsz,
                project="runs/train",
                name=exp_name,
            )

            # Récupération des métriques Ultralytics
            try:
                metrics = results.results_dict  # ultralytics >= 8.3
            except Exception:
                metrics = {}

            # On privilégie mAP50 comme métrique d'optimisation si disponible
            objective_metric = 0.0

            if "metrics/mAP50(B)" in metrics:
                objective_metric = float(metrics["metrics/mAP50(B)"])
                log_metric("metrics/mAP50(B)", objective_metric)

            # Si mAP50-95 est dispo, on la logge aussi (info)
            if "metrics/mAP50-95(B)" in metrics:
                log_metric("metrics/mAP50-95(B)", float(metrics["metrics/mAP50-95(B)"]))

            # Metrics optionnelles : précision / rappel
            for key in ["metrics/precision(B)", "metrics/recall(B)"]:
                if key in metrics:
                    log_metric(key, float(metrics[key]))

            # Si aucune métrique exploitable n'est disponible, on logge au moins un flag
            if not metrics:
                log_metric("training_finished", 1.0)

            # Dossier de sortie YOLO
            run_dir = Path("runs/train") / exp_name
            mlflow.log_param("yolo_run_dir", str(run_dir))

        # On cherche à MAXIMISER la métrique objective_metric
        return objective_metric

    # 2) Création et exécution de l'étude Optuna
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
