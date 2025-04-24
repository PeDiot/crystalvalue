import logging
from typing import Any, Optional

from google.cloud import aiplatform, bigquery
from google.oauth2 import service_account


CREDENTIALS_PATH = "gcp_credentials.json"

GCP_PROJECT_ID = "pltv-457408"
GCP_LOCATION = "europe-west4"
GCP_DATASET_ID = "crystalvalue_20250424_104512"
GCP_TABLE_ID = "data"

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


def create_automl_dataset(
    project_id: str,
    dataset_id: str,
    table_name: str = "training_data",
    dataset_display_name: str = "crystalvalue_dataset",
    location: str = "europe-west4",
    credentials: Optional[Any] = None
) -> aiplatform.datasets.tabular_dataset.TabularDataset:
    logging.info(
        "Creating Vertex AI Dataset with display name %r", dataset_display_name
    )
    bigquery_uri = f"bq://{project_id}.{dataset_id}.{table_name}"

    aiplatform.init(
        project=project_id, 
        location=location, 
        credentials=credentials
    )
    dataset = aiplatform.TabularDataset.create(
        display_name=dataset_display_name, bq_source=bigquery_uri
    )

    dataset.wait()
    return dataset


def train_automl_model(
    project_id: str,
    aiplatform_dataset: aiplatform.TabularDataset,
    model_display_name: str = "crystalvalue_model",
    predefined_split_column_name: str = "predefined_split_column",
    target_column: str = "future_value",
    optimization_objective: str = "minimize-rmse",
    optimization_prediction_type: str = "regression",
    budget_milli_node_hours: int = 1000,
    location: str = "europe-west4",
    credentials: Optional[Any] = None
) -> aiplatform.models.Model:
    logging.info(
        "Creating Vertex AI AutoML model with display name %r", model_display_name
    )

    transformations = [
        {"auto": {"column_name": f"{feature}"}}
        for feature in aiplatform_dataset.column_names
        if feature not in _NON_FEATURES
    ]

    aiplatform.init(
        project=project_id, 
        location=location, 
        credentials=credentials
    )

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=model_display_name,
        optimization_prediction_type=optimization_prediction_type,
        optimization_objective=optimization_objective,
        column_transformations=transformations,
    )

    model = job.run(
        dataset=aiplatform_dataset,
        target_column=target_column,
        budget_milli_node_hours=budget_milli_node_hours,
        model_display_name=model_display_name,
        predefined_split_column_name=predefined_split_column_name,
    )

    model.wait()
    logging.info("Created AI Platform Model with display name %r", model.display_name)
    return model


def deploy_model(
    bigquery_client: bigquery.Client,
    model_id: str,
    machine_type: str = "n1-standard-2",
    location: str = "europe-west4",
    credentials: Optional[Any] = None
) -> aiplatform.Model:
    aiplatform.init(project=bigquery_client.project, location=location, credentials=credentials)
    model = aiplatform.Model(model_name=model_id)
    model.deploy(machine_type=machine_type)
    model.wait()
    
    logging.info("Deployed model with display name %r", model.display_name)

    return model


def main():
    credentials = service_account.Credentials.from_service_account_file(
        filename=CREDENTIALS_PATH
    )

    bq_client = bigquery.Client(credentials=credentials)

    dataset = create_automl_dataset(
        project_id=GCP_PROJECT_ID,
        dataset_id=GCP_DATASET_ID,
        table_name=GCP_TABLE_ID,
        location=GCP_LOCATION,
        credentials=credentials
    )

    model = train_automl_model(
        project_id=GCP_PROJECT_ID,
        aiplatform_dataset=dataset,
        model_display_name="crystalvalue_model",
        location=GCP_LOCATION,
        credentials=credentials
    )

    deploy_model(
        bigquery_client=bq_client,
        model_id=model.name,
        location=GCP_LOCATION,
        credentials=credentials
    )


if __name__ == "__main__":
    main()