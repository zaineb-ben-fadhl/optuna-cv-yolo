$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

Write-Host "[run_optuna] Lancement de l'étude Optuna (n_trials=5)..."
python -m src.optuna_yolo --n-trials 5 --exp-prefix optuna_yolo
Write-Host "[run_optuna] Étude Optuna terminée."
