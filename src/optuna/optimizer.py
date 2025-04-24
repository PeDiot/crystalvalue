import optuna, logging, numpy as np
from datetime import datetime

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold

from .models import ModelFactory
from .data import DataLoader

logger = logging.getLogger(__name__)


class ModelOptimizer:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.config = data_loader.config
        self.study_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        (
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
        ) = self.data_loader.load_training_data()
        
        self.feature_names = self.data_loader.training_data.drop(
            columns=[self.config.target_column, self.config.train_val_test_split_column]
        ).columns.tolist()

    def optimize(self) -> optuna.Trial:
        study = optuna.create_study(
            direction="minimize",
            storage=self.config.storage_name,
            study_name=self.study_name,
            load_if_exists=True
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
    
    def _objective(self, trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical(
            "model", ["random_forest", "xgboost", "lightgbm"]
        )

        feature_selector = self._get_feature_selector(trial)
        if feature_selector is not None:
            X_train_selected = feature_selector.fit_transform(self.X_train, self.y_train)
            X_val_selected = feature_selector.transform(self.X_val)
            
            if hasattr(feature_selector, 'get_support'):
                selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                                  if feature_selector.get_support()[i]]
                logger.info(f"Selected features: {selected_features}")
        else:
            X_train_selected = self.X_train
            X_val_selected = self.X_val

        model = ModelFactory.create_model(
            model_name=model_name,
            config=self.config,
            trial=trial
        )
        model.fit(X_train_selected, self.y_train)

        y_pred = model.predict(X_val_selected)

        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        r2 = r2_score(self.y_val, y_pred)

        logger.info(
            f"Trial {trial.number}: Model={model_name}, RMSE={rmse:.4f}, R2={r2:.4f}"
        )

        return rmse
    
    def _get_feature_selector(self, trial: optuna.Trial) -> SelectKBest:
        if not self.config.feature_selection:
            return None
            
        n_features = trial.suggest_int(
            "n_features", 
            self.config.min_features, 
            min(self.config.max_features, len(self.feature_names))
        )
        
        if self.config.feature_selection_method == "variance":
            selector = VarianceThreshold()
        elif self.config.feature_selection_method == "mutual_info":
            selector = SelectKBest(mutual_info_regression, k=n_features)
        else:
            selector = SelectKBest(f_regression, k=n_features)
            
        return selector