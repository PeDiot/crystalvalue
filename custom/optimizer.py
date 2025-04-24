from typing import Tuple
from abc import ABC, abstractmethod

import optuna, logging, numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

from .models import ModelFactory
from .data import DataLoader
from .feature_selection import create_feature_selector

logger = logging.getLogger(__name__)


class BaseModelOptimizer(ABC):
    study_type = ""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.config = data_loader.config
        self.study_name = (
            f"{self.study_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        (
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
        ) = self.data_loader.load_training_data()

        self.feature_names = self.data_loader.feature_names
        self.feature_selector = None

    def optimize(self) -> optuna.Trial:
        study = optuna.create_study(
            direction=self._get_optimization_direction(),
            storage=self.config.storage_name,
            study_name=self.study_name,
            load_if_exists=True,
        )

        study.optimize(
            self._objective, n_trials=self.config.n_trials, timeout=self.config.timeout
        )

        logger.info("Best trial:")
        logger.info(f"  Value: {study.best_trial.value:.4f}")
        logger.info("  Params: ")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

        return study.best_trial

    @abstractmethod
    def _get_optimization_direction(self) -> str:
        """Return the optimization direction ('minimize' or 'maximize')"""
        pass

    @abstractmethod
    def _calculate_metrics(self, y_true, y_pred) -> tuple:
        pass

    def _objective(self, trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical(
            "model", ["random_forest", "xgboost", "lightgbm"]
        )

        self._get_feature_selector(trial)

        if self.feature_selector is not None:
            X_train_selected = self.feature_selector.fit_transform(
                self.X_train, self.y_train
            )
            X_val_selected = self.feature_selector.transform(self.X_val)

            if hasattr(self.feature_selector, "get_support"):
                selected_features = [
                    self.feature_names[i]
                    for i in range(len(self.feature_names))
                    if self.feature_selector.get_support()[i]
                ]
                logger.info(f"Selected features: {selected_features}")
        else:
            X_train_selected = self.X_train
            X_val_selected = self.X_val

        model = ModelFactory.create_model(
            model_name=model_name, config=self.config, trial=trial
        )
        model.fit(X_train_selected, self.y_train)

        y_pred = model.predict(X_val_selected)
        metrics = self._calculate_metrics(self.y_val, y_pred)

        logger.info(f"Trial {trial.number}: Model={model_name}, Metrics={metrics}")

        return metrics[0]

    def _get_feature_selector(self, trial: optuna.Trial):
        if not self.config.feature_selection:
            self.feature_selector = None
            return

        n_features = trial.suggest_int(
            "n_features",
            self.config.min_features,
            min(self.config.max_features, len(self.feature_names)),
        )

        self.feature_selector = create_feature_selector(
            method=self.config.feature_selection_method, n_features=n_features
        )


class RegressionModelOptimizer(BaseModelOptimizer):
    study_type = "reg"

    def _get_optimization_direction(self) -> str:
        return "minimize"

    def _calculate_metrics(self, y_true, y_pred) -> Tuple[float, float]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return rmse, r2


class ClassificationModelOptimizer(BaseModelOptimizer):
    study_type = "clf"

    def _get_optimization_direction(self) -> str:
        return "maximize"

    def _calculate_metrics(self, y_true, y_pred) -> Tuple[float, float]:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        return f1, accuracy
