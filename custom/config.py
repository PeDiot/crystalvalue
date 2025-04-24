from typing import List
from dataclasses import dataclass, field

import yaml
from pathlib import Path


VALID_FEATURE_SELECTION_METHODS = ["variance", "mutual_info", "f_regression"]


@dataclass
class Config:
    project_id: str
    dataset_id: str
    table_id: str
    target_column: str
    train_val_test_split_column: str
    category_columns: List[str] = field(default_factory=lambda: [])
    excluded_columns: List[str] = field(default_factory=lambda: [])
    n_samples: int = -1
    stratified: bool = False
    n_trials: int = 100
    timeout: int = 3600
    random_state: int = 42
    storage_name: str = "sqlite:///optuna_study.db"
    save_dir: str = "./models"
    feature_selection: bool = False
    min_features: int = 5
    max_features: int = 50
    feature_selection_method: str = "variance"

    def __post_init__(self):
        if self.target_column not in ["future_value", "future_value_classification"]:
            raise ValueError("Invalid target column")

        if self.target_column == "future_value_classification":
            self.excluded_columns.append("future_value")
            self.training_mode = "classification"

        if self.target_column == "future_value":
            self.excluded_columns.append("future_value_classification")
            self.training_mode = "regression"
        if self.feature_selection:
            assert (
                self.feature_selection_method in VALID_FEATURE_SELECTION_METHODS
            ), "Invalid feature selection method"

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "Config":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found at {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)
