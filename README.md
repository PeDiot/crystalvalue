# `crystalvalue`

## Training 

1. Get secret key for service account on GCP
2. Build image
    ```
    docker build -t crystalvalue-train .
    ```
3. Run container for training
    ```
    docker run -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json -v $(pwd)/secrets/gcp_credentials.json:/app/credentials.json crystalvalue-train
    ```

## Exploration 

See [`notebook`](notebook.ipynb).