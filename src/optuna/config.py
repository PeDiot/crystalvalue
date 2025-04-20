from typing import List
import yaml
from dataclasses import dataclass, field
from pathlib import Path


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
    n_trials: int = 100
    timeout: int = 3600
    random_state: int = 42
    storage_name: str = "sqlite:///optuna_study.db"
    save_dir: str = "./models"

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "Config":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found at {yaml_path}")
            
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)