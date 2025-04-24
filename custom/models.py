from abc import ABC, abstractmethod
from typing import Any, Optional


import optuna, numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

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


class BaseClassificationModel(BaseModel):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


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


class RandomForestClassifierModel(BaseClassificationModel):
    def _create_model(self) -> RandomForestClassifier:
        if self.trial is None:
            return RandomForestClassifier(random_state=self.config.random_state)
        
        return RandomForestClassifier(
            n_estimators=self.trial.suggest_int("n_estimators", 100, 1000),
            max_depth=self.trial.suggest_int("max_depth", 3, 20),
            min_samples_split=self.trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=self.trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=self.trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            random_state=self.config.random_state,
        )


class XGBoostClassifierModel(BaseClassificationModel):
    def _create_model(self) -> XGBClassifier:
        if self.trial is None:
            return XGBClassifier(random_state=self.config.random_state)
        
        return XGBClassifier(
            n_estimators=self.trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=self.trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth=self.trial.suggest_int("max_depth", 3, 10),
            min_child_weight=self.trial.suggest_int("min_child_weight", 1, 10),
            subsample=self.trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=self.trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=self.config.random_state,
        )


class LightGBMClassifierModel(BaseClassificationModel):
    def _create_model(self) -> LGBMClassifier:
        if self.trial is None:
            return LGBMClassifier(random_state=self.config.random_state)
        
        return LGBMClassifier(
            n_estimators=self.trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=self.trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth=self.trial.suggest_int("max_depth", 3, 10),
            num_leaves=self.trial.suggest_int("num_leaves", 20, 100),
            min_child_samples=self.trial.suggest_int("min_child_samples", 1, 20),
            subsample=self.trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=self.trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=self.config.random_state,
        )


class LogisticRegressionModel(BaseClassificationModel):
    def _create_model(self) -> LogisticRegression:
        if self.trial is None:
            return LogisticRegression(random_state=self.config.random_state)

        return LogisticRegression(
            C=self.trial.suggest_float("C", 0.1, 10.0),
            penalty=self.trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]),
            solver=self.trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
            max_iter=self.trial.suggest_int("max_iter", 100, 1000),
            random_state=self.config.random_state,
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
            "random_forest_classifier": RandomForestClassifierModel,
            "xgboost_classifier": XGBoostClassifierModel,
            "lightgbm_classifier": LightGBMClassifierModel,
            "logistic_regression": LogisticRegressionModel,
        }

        if model_name not in model_classes:
            raise ValueError(f"Unknown model type: {model_name}")

        return model_classes[model_name](config, trial)