#!/usr/bin/env python3

import logging

from src.optuna.config import Config
from src.optuna.data import DataLoader
from src.optuna.optimizer import ModelOptimizer
from src.optuna.models import ModelFactory
from src.optuna.utils import ModelSaver


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        config = Config.from_yaml()
        logger.info("Initializing data loader...")
        data_loader = DataLoader(config=config)

        logger.info("Starting model optimization...")
        optimizer = ModelOptimizer(data_loader=data_loader, config=config)
        best_trial = optimizer.optimize()

        logger.info("Creating and saving the best model...")
        best_model = ModelFactory.create_model(best_trial.params["model"], best_trial, config)
        best_model.fit(optimizer.X_train, optimizer.y_train)

        ModelSaver.save_model(
            model=best_model,
            preprocessor=data_loader.preprocessor,
            model_name=best_trial.params["model"], 
            save_dir=config.save_dir
        )

        logger.info("Optimization completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred during optimization: {str(e)}")
        raise


if __name__ == "__main__":
    main()