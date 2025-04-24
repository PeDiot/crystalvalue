from typing import Tuple, Optional

import joblib, logging, os
from sklearn.compose import ColumnTransformer

from .models import BaseModel
from .feature_selection import FeatureSelector
logger = logging.getLogger(__name__)


class ModelSaver:
    @staticmethod
    def save_model(
        model: BaseModel, 
        preprocessor: ColumnTransformer, 
        model_name: str, 
        feature_selector: Optional[FeatureSelector] = None,
        save_dir: str = "./"
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"{model_name}.joblib")
        preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
        
        joblib.dump(model.model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        if feature_selector is not None:
            feature_selector_path = os.path.join(save_dir, "feature_selector.joblib")
            joblib.dump(feature_selector, feature_selector_path)

        logger.info(f"Saved model and preprocessor to disk in directory: {save_dir}")

    @staticmethod
    def load_model(model_name: str, save_dir: str = "./") -> Tuple[BaseModel, ColumnTransformer]:
        model_path = os.path.join(save_dir, f"{model_name}.joblib")
        preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Model or preprocessor files not found in {save_dir}")
            
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        logger.info(f"Loaded model and preprocessor from directory: {save_dir}")
        return model, preprocessor