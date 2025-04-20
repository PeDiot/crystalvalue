import optuna, logging, numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os

from .config import Config
from .models import ModelFactory
from .data import DataLoader

logger = logging.getLogger(__name__)


class ModelOptimizer:
    def __init__(self, data_loader: DataLoader, config: Config):
        self.data_loader = data_loader
        self.config = config
        
        (
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
        ) = self.data_loader.load_data()

    def objective(self, trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical(
            "model", ["random_forest", "xgboost", "lightgbm"]
        )

        model = ModelFactory.create_model(model_name, trial, self.config)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_val)

        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        r2 = r2_score(self.y_val, y_pred)

        logger.info(
            f"Trial {trial.number}: Model={model_name}, RMSE={rmse:.4f}, R2={r2:.4f}"
        )

        return rmse

    def optimize(self) -> optuna.Trial:
        study = optuna.create_study(
            direction="minimize",
            storage=self.config.storage_name,
            study_name="model_optimization",
            load_if_exists=True
        )
        
        study.optimize(
            self.objective, n_trials=self.config.n_trials, timeout=self.config.timeout
        )

        logger.info("Best trial:")
        logger.info(f"  Value: {study.best_trial.value:.4f}")
        logger.info("  Params: ")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

        return study.best_trial