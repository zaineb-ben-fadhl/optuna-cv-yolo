
# optuna-cv-yolo (YOLO tiny + MLflow + Optuna)

Projet MLOps de détection d'objets (YOLO tiny) avec :

- **DVC** pour le versioning de données (tiny COCO "person")
- **MLflow + AWS S3** pour le tracking d'expériences et des artefacts
- **Scripts Python** pour :
  - un entraînement **baseline** YOLO tiny (`src/train_cv.py`)
  - une petite **grille d’expériences** (`scripts/run_grid.*`)
  - une **optimisation d’hyperparamètres avec Optuna** (`src/optuna_yolo.py`, `scripts/run_optuna.*`)
- Contexte :
  - TP4 : script `train_cv.py` + MLflow
  - TP6 : ajout d’**Optuna** pour optimiser les hyperparamètres (epochs, imgsz, …)

---

## 1. Démarrer l'infrastructure (MLflow + AWS S3)

Puis lancer l’infrastructure Docker :

```bash
docker compose up -d
```

Services principaux :

* **MLflow UI** : [http://localhost:5000](http://localhost:5000)
* (Éventuellement) **ZenML Server** : [http://localhost:8080](http://localhost:8080) (non utilisé dans ce TP)

Les artefacts MLflow (modèles, fichiers, métriques, etc.) sont stockés dans un bucket **AWS S3**
(configuré dans `docker-compose.yml`).

---

## 2. Environnement Python local

Toutes les commandes d’entraînement (baseline, grille, Optuna) s’exécutent dans
votre **environnement Python local** (venv).

Création de l’environnement :

```bash
python3 -m venv .venv
source .venv/bin/activate          # Adapter sous Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Le fichier `requirements.txt` inclut notamment :

* `mlflow`
* `ultralytics`
* `dvc[s3]`
* `optuna`

---

## 3. Dataset tiny COCO (person) + DVC

Le projet utilise un dataset **réduit** dérivé de COCO (personnes uniquement).

### Génération locale

```bash
python tools/make_tiny_person_from_coco128.py
```

### Versioning avec DVC

Selon les TP précédents, soit :

```bash
dvc status
```

ou, si nécessaire :

```bash
dvc init
dvc add data/tiny_coco -R
git add data/tiny_coco.dvc .gitignore .dvc/ .gitattributes
git commit -m "Track dataset tiny_coco with DVC"
```

Le dossier `data/tiny_coco` est ensuite utilisé pour :

* l’entraînement YOLO (script `train_cv.py`)
* les expériences Optuna (`optuna_yolo.py`)

---

## 4. Configuration MLflow

Avant de lancer les scripts, assurez-vous que `MLFLOW_TRACKING_URI`
pointe vers l’instance locale (dans Docker) :

### Linux / macOS / Git Bash

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

### PowerShell

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

### CMD

```bat
set MLFLOW_TRACKING_URI=http://localhost:5000
```

Les runs seront visibles dans l’UI MLflow et les artefacts seront stockés sur **AWS S3**.

---

## 5. Entraînement baseline (`src/train_cv.py`)

Script principal d’entraînement YOLO tiny (hérité du TP4).

Exemple de commande :

```bash
python -m src.train_cv --epochs 3 --imgsz 320 --exp-name yolo_baseline_optuna
```

Ce script :

* entraîne un modèle YOLOv8 tiny (`ultralytics.YOLO`)
* loggue dans MLflow :

  * les paramètres (epochs, imgsz, etc.)
  * les métriques (par ex. `metrics/mAP50(B)`, `metrics/mAP50-95(B)`)
  * le chemin du dossier YOLO (`runs/train/...`)
* stocke les artefacts dans **S3** (via la config MLflow)

---

## 6. Grille d’expériences (scripts `run_grid.*`)

Pour rappeler l’approche « grille » (TP4), le dépôt fournit des scripts :

* `scripts/run_grid.sh` (Linux / macOS / Git Bash)
* `scripts/run_grid.ps1` (PowerShell)
* `scripts/run_grid.cmd` (CMD)

Ils lancent plusieurs runs avec des variations d’hyperparamètres simples (par ex. `epochs`, `imgsz`).

Exemples de lancement :

```bash
# Linux / macOS
bash scripts/run_grid.sh

# PowerShell
.\scripts\run_grid.ps1

# CMD
scripts\run_grid.cmd
```

Dans l’UI MLflow, vous verrez plusieurs runs du type :

* `yolo_e3_320`
* `yolo_e5_416`
* etc.

---

## 7. Optimisation des hyperparamètres avec Optuna

Le cœur du TP6 est le script :

* `src/optuna_yolo.py`

Il définit une étude Optuna qui :

* choisit des hyperparamètres (par ex. `epochs`, `imgsz`)
* lance un entraînement YOLO tiny pour chaque **trial**
* loggue chaque trial dans MLflow (un run MLflow par trial)
* renvoie une métrique à **maximiser** (par ex. `metrics/mAP50(B)`)

### Scripts de lancement

Suivant votre OS, utilisez :

```bash
# Linux / macOS
bash scripts/run_optuna.sh

# PowerShell
.\scripts\run_optuna.ps1

# CMD
scripts\run_optuna.cmd
```

Les paramètres par défaut (dans les scripts) :

* `--n-trials 5` (5 essais)
* `--exp-prefix optuna_yolo` (préfixe dans les noms de runs)

En fin d’étude, le script affiche :

* la meilleure valeur de la métrique (ex. meilleur mAP50)
* les meilleurs hyperparamètres trouvés par Optuna

---

## 8. Analyse dans MLflow

Dans l’UI MLflow ([http://localhost:5000](http://localhost:5000)), vous pouvez :

1. **Filtrer** les runs :

   * par expérience (`cv_yolo_tiny` / `cv_yolo_tiny_optuna` selon config)
   * par tag (`optuna_study`, etc.)
2. **Comparer** :

   * les runs issus de la **grille** (`run_grid`)
   * les runs issus d’**Optuna** (`run_optuna`)
3. Observer :

   * l’impact de `epochs` et `imgsz` sur `metrics/mAP50(B)`
   * les compromis performance / temps d’entraînement

---

## 9. (Optionnel) ZenML / ZenML Server

Le dépôt contient encore :

* des fichiers de configuration ZenML (`src/zenml_steps/`, `src/zenml_pipelines/`, etc.)
* un conteneur **ZenML Server** dans `docker-compose.yml`

Ils proviennent du TP5 (pipelines ZenML + MLflow + S3).

> Pour ce TP6 (**Optuna**), vous n’avez pas besoin de :
>
> * vous connecter à ZenML Server
> * gérer les stacks ZenML
> * lancer des pipelines ZenML

Toute l’activité se fait via :

* `train_cv.py`
* `run_grid.*`
* `optuna_yolo.py`
* MLflow + S3

---

## 10. (Annexe) Configuration initiale côté ZenML Server

> Cette section est **réservée à l’admin**.

1. Créer 2 buckets AWS S3 :

* `mlflow-artifacts`
* `zenml-artifacts`

2. Entrer dans le conteneur ZenML Server :

```bash
docker exec -it zenml-server bash
```

3. Configurer MLflow comme tracker :

```bash
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri=http://mlflow:5000 \
    --tracking_token="dummy-token"
```

4. Créer le secret AWS :

```bash
zenml secret create aws_s3_secret \
    --aws_access_key_id="$AWS_ACCESS_KEY_ID" \
    --aws_secret_access_key="$AWS_SECRET_ACCESS_KEY" \
    --aws_session_token="$AWS_SESSION_TOKEN"
```

5. Artifact store S3 :

```bash
zenml artifact-store register s3_artifacts \
    --flavor=s3 \
    --path='s3://VOTRE_BUCKET/zenml-artifacts' \
    --authentication_secret=aws_s3_secret
```

6. Orchestrateur local :

```bash
zenml orchestrator register local_orch --flavor=local
```

7. Stack complète :

```bash
zenml stack register mlflow_stack \
    -o local_orch -a s3_artifacts -e mlflow_tracker
```

8. Définir la stack par défaut :

```bash
zenml stack set mlflow_stack
```