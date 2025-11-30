@echo off
set MLFLOW_TRACKING_URI=http://localhost:5000

echo [run_optuna] Lancement de l'étude Optuna (n_trials=5)...
python -m src.optuna_yolo --n-trials 5 --exp-prefix optuna_yolo
echo [run_optuna] Étude Optuna terminée.
