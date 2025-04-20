from src import crystalvalue


GCP_PROJECT_ID = "pltv-457408"
GCP_LOCATION = "europe-west9"
GCP_DATASET_ID = "crystalvalue"
GCP_TABLE_ID_TRAIN = "train"

CUSTOMER_ID_COLUMN = "customer_id"
DATE_COLUMN = "date"
VALUE_COLUMN = "value"
IGNORE_COLUMNS = ["order_number", "days_to_next_order"]

LOOKBACK_DAYS = 365
LOOKAHEAD_DAYS = 365

TEST_YEAR = 2024
TEST_MONTH = 3


def main(feature_engineering: bool = False):
    pipeline = crystalvalue.CrystalValue(
        project_id=GCP_PROJECT_ID,
        dataset_id=GCP_DATASET_ID,
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
            transaction_table_name=GCP_TABLE_ID_TRAIN,
        )
        print("Feature engineering done")

    pipeline.train_automl_model()
    print("Training done")

    pipeline.deploy_model()
    print("Deploying done")


if __name__ == "__main__":
    main()
