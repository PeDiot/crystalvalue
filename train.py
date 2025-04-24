from typing import Literal
from dataclasses import dataclass

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


@dataclass
class TrainingConfig:
    mode: Literal["regression", "classification"]
    predefined_split_column_name: str = "predefined_split_column"
    budget_milli_node_hours: int = 1000

    def __post_init__(self):
        if self.mode == "regression":
            self.target_column = "future_value"
            self.optimization_objective = "minimize-rmse"
            self.optimization_prediction_type = "regression"
        else:
            self.target_column = "future_value_classification"
            self.optimization_objective = "maximize-au-roc"
            self.optimization_prediction_type = "classification"

    @property
    def model_display_name(self) -> str:
        return f"{GCP_DATASET_ID}_{self.mode}"


def create_automl_dataset() -> aiplatform.datasets.tabular_dataset.TabularDataset:
    logging.info(
        "Creating Vertex AI Dataset with display name %r", GCP_DATASET_ID
    )
    bigquery_uri = f"bq://{GCP_PROJECT_ID}.{GCP_DATASET_ID}.{GCP_TABLE_ID}"

    dataset = aiplatform.TabularDataset.create(
        display_name=GCP_DATASET_ID, bq_source=bigquery_uri
    )

    dataset.wait()

    return dataset


def load_automl_dataset() -> aiplatform.datasets.tabular_dataset.TabularDataset:
    logging.info("Attempting to load existing Vertex AI Dataset with display name %r", GCP_DATASET_ID)
    
    datasets = aiplatform.TabularDataset.list()
    for dataset in datasets:
        if dataset.display_name == GCP_DATASET_ID:
            logging.info("Found existing dataset with display name %r", GCP_DATASET_ID)
            return dataset
    
    logging.info("No existing dataset found with display name %r", GCP_DATASET_ID)

    return None


def train_automl_model(
    aiplatform_dataset: aiplatform.TabularDataset,
    config: TrainingConfig,
) -> aiplatform.models.Model:
    logging.info(
        "Creating Vertex AI AutoML model with display name %r", GCP_DATASET_ID
    )

    transformations = [
        {"auto": {"column_name": f"{feature}"}}
        for feature in aiplatform_dataset.column_names
        if feature not in _NON_FEATURES
    ]

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=GCP_DATASET_ID,
        optimization_prediction_type=config.optimization_prediction_type,
        optimization_objective=config.optimization_objective,
        column_transformations=transformations,
    )

    model = job.run(
        dataset=aiplatform_dataset,
        target_column=config.target_column,
        budget_milli_node_hours=config.budget_milli_node_hours,
        model_display_name=config.model_display_name,
        predefined_split_column_name=config.predefined_split_column_name,
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


def main(mode: Literal["regression", "classification"]):
    global credentials
    credentials = service_account.Credentials.from_service_account_file(
        filename=CREDENTIALS_PATH
    )

    aiplatform.init(
        project=GCP_PROJECT_ID, 
        location=GCP_LOCATION, 
        credentials=credentials
    )

    dataset = load_automl_dataset()
    if dataset is None:
        dataset = create_automl_dataset()
    
    config = TrainingConfig(mode=mode)
    print(config)

    model = train_automl_model(
        aiplatform_dataset=dataset, config=config
    )
    deploy_model(model.name)


if __name__ == "__main__":
    main(mode="classification")