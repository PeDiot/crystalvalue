import os, json
from google.oauth2 import service_account
from src import crystalvalue


GCP_PROJECT_ID = "pltv-457408"
GCP_LOCATION = "europe-west9"
GCP_DATASET_ID = "crystalvalue"
GCP_TABLE_ID_TRAIN = "train"
GCP_TABLE_ID_PREDICT = "predict"

CUSTOMER_ID_COLUMN = "customer_id"
DATE_COLUMN = "date"
VALUE_COLUMN = "value"
IGNORE_COLUMNS = ["order_number", "days_to_next_order"]

LOOKBACK_DAYS = 365
LOOKAHEAD_DAYS = 365

TEST_YEAR = 2024
TEST_MONTH = 3


def main(): 
    credentials_dict = json.loads(os.getenv("GCP_CREDENTIALS"))
    credentials_dict["private_key"] = credentials_dict["private_key"].replace("\\n", "\n")

    credentials = service_account.Credentials.from_service_account_info(
        info=credentials_dict
    )

    pipeline = crystalvalue.CrystalValue(
        project_id=GCP_PROJECT_ID,
        dataset_id=GCP_DATASET_ID,
        credentials=credentials,
        training_table_name=GCP_TABLE_ID_TRAIN,
        predict_table_name=GCP_TABLE_ID_PREDICT,
        customer_id_column=CUSTOMER_ID_COLUMN,
        date_column=DATE_COLUMN,
        value_column=VALUE_COLUMN,
        days_lookback=LOOKBACK_DAYS,
        days_lookahead=LOOKAHEAD_DAYS,
        ignore_columns=IGNORE_COLUMNS,
        location=GCP_LOCATION,
        write_parameters=True,
    )  

    print("Feature engineering...")
    crystalvalue_train_data = pipeline.feature_engineer(
        transaction_table_name=GCP_TABLE_ID_TRAIN,
    )
    print("Feature engineering done")

    print("Training...")
    model_object = pipeline.train_automl_model()
    print("Training done")

    print("Deploying...")
    model_object = pipeline.deploy_model()
    print("Deploying done")


if __name__ == "__main__":
    main()