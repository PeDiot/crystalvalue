`crystalvalue`

## Run 

```
docker build -t crystalvalue-train .
docker run -d --name crystalvalue-train -e GCP_CREDENTIALS="$(cat secrets/gcp_credentials.json)" crystalvalue-train
```