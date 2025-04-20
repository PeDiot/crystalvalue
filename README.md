# `crystalvalue`

## Install 

```
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

## Exploration 

See [`notebook`](notebook.ipynb).

## AutoML 

1. Get secret key for service account on GCP
2. Build image
    ```
    docker build -t crystalvalue-train .
    ```
3. Run container for training
    ```
    docker run -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json -v $(pwd)/secrets/gcp_credentials.json:/app/credentials.json crystalvalue-train
    ```

## Optuna

Optimization params are in [`config`](config.yaml).

### Run

```
source venv/bin/activate
pip install -r requirements_dev.txt
python3 optimize.py
```

### Monitor

```
optuna-dashboard sqlite:///optuna_study.db
```