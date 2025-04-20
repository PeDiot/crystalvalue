import joblib, logging, os
from sklearn.compose import ColumnTransformer

from .models import BaseModel

logger = logging.getLogger(__name__)


class ModelSaver:
    @staticmethod
    def save_model(model: BaseModel, preprocessor: ColumnTransformer, model_name: str, save_dir: str = "./") -> None:
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"best_{model_name}_model.joblib")
        preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
        
        joblib.dump(model.model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Saved model and preprocessor to disk in directory: {save_dir}")