from src import crystalvalue
from google.oauth2 import service_account


CREDENTIALS_PATH = "gcp_credentials.json"

GCP_PROJECT_ID = "pltv-457408"
GCP_LOCATION = "europe-west4"
GCP_DATASET_ID = "crystalvalue"
GCP_TABLE_ID = "data"

CUSTOMER_ID_COLUMN = "customer_id"
DATE_COLUMN = "date"
VALUE_COLUMN = "value"
IGNORE_COLUMNS = ["order_number", "days_to_next_order"]

LOOKBACK_DAYS = 0
LOOKAHEAD_DAYS = 365


def main(feature_engineering: bool = False):
    credentials = service_account.Credentials.from_service_account_file(
        filename=CREDENTIALS_PATH
    )

    pipeline = crystalvalue.CrystalValue(
        project_id=GCP_PROJECT_ID,
        dataset_id=GCP_DATASET_ID,
        credentials=credentials,
        customer_id_column=CUSTOMER_ID_COLUMN,
        date_column=DATE_COLUMN,
        value_column=VALUE_COLUMN,
        days_lookback=LOOKBACK_DAYS,
        days_lookahead=LOOKAHEAD_DAYS,
        ignore_columns=IGNORE_COLUMNS,
        location=GCP_LOCATION,
        write_parameters=True,
    )

    if feature_engineering:
        pipeline.feature_engineer(
            transaction_table_name=GCP_TABLE_ID,
            query_type="train_query",
        )
        print("Feature engineering done")

    pipeline.train_automl_model()
    print("Training done")

    pipeline.deploy_model()
    print("Deploying done")


if __name__ == "__main__":
    main()