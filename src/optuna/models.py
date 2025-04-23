from abc import ABC, abstractmethod
from typing import Any, Optional


import optuna, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from .config import Config


class BaseModel(ABC):
    def __init__(self, config: Config, trial: Optional[optuna.Trial] = None):
        self.config = config
        self.trial = trial
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self) -> Any:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestModel(BaseModel):
    def _create_model(self) -> RandomForestRegressor:
        if self.trial is None:
            return RandomForestRegressor(random_state=self.config.random_state)
        
        return RandomForestRegressor(
            n_estimators=self.trial.suggest_int("n_estimators", 100, 1000),
            max_depth=self.trial.suggest_int("max_depth", 3, 20),
            min_samples_split=self.trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=self.trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=self.trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            random_state=self.config.random_state,
        )


class XGBoostModel(BaseModel):
    def _create_model(self) -> XGBRegressor:
        if self.trial is None:
            return XGBRegressor(random_state=self.config.random_state)
        
        return XGBRegressor(
            n_estimators=self.trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=self.trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth=self.trial.suggest_int("max_depth", 3, 10),
            min_child_weight=self.trial.suggest_int("min_child_weight", 1, 10),
            subsample=self.trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=self.trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=self.config.random_state,
        )


class LightGBMModel(BaseModel):
    def _create_model(self) -> LGBMRegressor:
        if self.trial is None:
            return LGBMRegressor(random_state=self.config.random_state)
        
        return LGBMRegressor(
            n_estimators=self.trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=self.trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth=self.trial.suggest_int("max_depth", 3, 10),
            num_leaves=self.trial.suggest_int("num_leaves", 20, 100),
            min_child_samples=self.trial.suggest_int("min_child_samples", 1, 20),
            subsample=self.trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=self.trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=self.config.random_state,
        )


class LinearRegressionModel(BaseModel):
    def _create_model(self) -> LinearRegression:
        if self.trial is None:
            return LinearRegression()

        return LinearRegression(
            fit_intercept=self.trial.suggest_categorical("fit_intercept", [True, False]),
            n_jobs=-1
        )


class ModelFactory:
    @staticmethod
    def create_model(
        model_name: str, 
        config: Config, 
        trial: Optional[optuna.Trial] = None, 
    ) -> BaseModel:
        model_classes = {
            "random_forest": RandomForestModel,
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel,
            "linear_regression": LinearRegressionModel,
        }

        if model_name not in model_classes:
            raise ValueError(f"Unknown model type: {model_name}")

        return model_classes[model_name](config, trial)