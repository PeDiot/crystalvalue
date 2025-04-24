import logging

from google.cloud import aiplatform, bigquery
from google.oauth2 import service_account


CREDENTIALS_PATH = "gcp_credentials.json"

GCP_PROJECT_ID = "pltv-457408"
GCP_LOCATION = "europe-west4"
GCP_DATASET_ID = "crystalvalue_20250424_104512"
GCP_TABLE_ID = "crystalvalue_train_data"

CUSTOMER_ID_COLUMN = "customer_id"
DATE_COLUMN = "date"
VALUE_COLUMN = "value"
IGNORE_COLUMNS = ["order_number", "days_to_next_order", "shipping_address_dept"]

LOOKBACK_DAYS = 0
LOOKAHEAD_DAYS = 365

_NON_FEATURES = [
    "customer_id",
    "window_date",
    "lookback_start",
    "lookahead_start",
    "lookahead_stop",
    "future_value",
    "future_value_classification",
    "predefined_split_column",
]


def create_automl_dataset() -> aiplatform.datasets.tabular_dataset.TabularDataset:
    logging.info(
        "Creating Vertex AI Dataset with display name %r", GCP_DATASET_ID
    )
    bigquery_uri = f"bq://{GCP_PROJECT_ID}.{GCP_DATASET_ID}.{GCP_TABLE_ID}"

    aiplatform.init(
        project=GCP_PROJECT_ID, 
        location=GCP_LOCATION, 
        credentials=credentials
    )
    dataset = aiplatform.TabularDataset.create(
        display_name=GCP_DATASET_ID, bq_source=bigquery_uri
    )

    dataset.wait()

    return dataset


def train_automl_model(
    aiplatform_dataset: aiplatform.TabularDataset,
    predefined_split_column_name: str = "predefined_split_column",
    target_column: str = "future_value",
    optimization_objective: str = "minimize-rmse",
    optimization_prediction_type: str = "regression",
    budget_milli_node_hours: int = 1000,
) -> aiplatform.models.Model:
    logging.info(
        "Creating Vertex AI AutoML model with display name %r", GCP_DATASET_ID
    )

    transformations = [
        {"auto": {"column_name": f"{feature}"}}
        for feature in aiplatform_dataset.column_names
        if feature not in _NON_FEATURES
    ]

    aiplatform.init(
        project=GCP_PROJECT_ID, 
        location=GCP_LOCATION, 
        credentials=credentials
    )

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=GCP_DATASET_ID,
        optimization_prediction_type=optimization_prediction_type,
        optimization_objective=optimization_objective,
        column_transformations=transformations,
    )

    model = job.run(
        dataset=aiplatform_dataset,
        target_column=target_column,
        budget_milli_node_hours=budget_milli_node_hours,
        model_display_name=GCP_DATASET_ID,
        predefined_split_column_name=predefined_split_column_name,
    )

    model.wait()
    logging.info("Created AI Platform Model with display name %r", model.display_name)
    return model


def deploy_model(model_id: str, machine_type: str = "n1-standard-2") -> aiplatform.Model:
    bq_client = bigquery.Client(credentials=credentials)

    aiplatform.init(project=bq_client.project, location=GCP_LOCATION, credentials=credentials)
    model = aiplatform.Model(model_name=model_id)
    model.deploy(machine_type=machine_type)
    model.wait()
    
    logging.info("Deployed model with display name %r", model.display_name)

    return model


def main():
    global credentials
    credentials = service_account.Credentials.from_service_account_file(
        filename=CREDENTIALS_PATH
    )

    dataset = create_automl_dataset()
    model = train_automl_model(aiplatform_dataset=dataset)
    deploy_model(model.name)


if __name__ == "__main__":
    main()