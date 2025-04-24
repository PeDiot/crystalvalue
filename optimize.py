#!/usr/bin/env python3

import logging

from google.oauth2 import service_account

from custom.config import Config
from custom.data import DataLoader
from custom.optimizer import ModelOptimizer
from custom.models import ModelFactory
from custom.utils import ModelSaver


CREDENTIALS_PATH = "secrets/gcp_credentials.json"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    credentials = service_account.Credentials.from_service_account_file(
        filename=CREDENTIALS_PATH
    )
    
    try:
        config = Config.from_yaml()
        logger.info("Initializing data loader...")
        loader = DataLoader(config=config, gcp_credentials=credentials)

        logger.info("Starting model optimization...")
        optimizer = ModelOptimizer(data_loader=loader)
        best_trial = optimizer.optimize()

        logger.info("Creating and saving the best model...")
        best_model = ModelFactory.create_model(
            model_name=best_trial.params["model"],
            trial=best_trial,
            config=config
        )
        best_model.fit(optimizer.X_train, optimizer.y_train)

        ModelSaver.save_model(
            model=best_model,
            preprocessor=loader.preprocessor,
            feature_selector=optimizer.feature_selector,
            model_name=best_trial.params['model'], 
            save_dir=f"{config.save_dir}/{optimizer.study_name}"
        )

        logger.info("Optimization completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred during optimization: {str(e)}")
        raise


if __name__ == "__main__":
    main()