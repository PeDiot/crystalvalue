from typing import Tuple

import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = None
        
    def load_data(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = self._run_query()

        X, y, train_val_split = self._select_features(df)
        X_train, X_val, y_train, y_val = self._train_val_split(X, y, train_val_split)
        X_train, X_val = self._preprocess(X_train, X_val)

        return X_train, X_val, y_train, y_val
    
    def _train_val_split(self, X: pd.DataFrame, y: pd.Series, train_val_split: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_mask = train_val_split == "TRAIN"
        val_mask = train_val_split == "VALIDATE"

        X_train = X[train_mask]
        X_val = X[val_mask]
        y_train = y[train_mask]
        y_val = y[val_mask] 

        return X_train, X_val, y_train, y_val
    
    def _preprocess(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
        
        num_transformer = ("num", StandardScaler(), numeric_features)
        cat_transformer = ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        
        self.preprocessor = ColumnTransformer(
            transformers=[num_transformer, cat_transformer]
        )
        
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_val_scaled = self.preprocessor.transform(X_val)

        return X_train_scaled, X_val_scaled
    
    def _select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        train_val_split = df[self.config.train_val_test_split_column]
        
        to_drop_columns = [self.config.target_column, self.config.train_val_test_split_column]
        X = df.drop(columns=to_drop_columns)
        y = df[self.config.target_column]

        return X, y, train_val_split    
    
    def _run_query(self) -> pd.DataFrame:
        query = self._query()
        
        return pd.read_gbq(
            query=query,
            project_id=self.config.project_id,
            use_bqstorage_api=True, 
            progress_bar_type="tqdm"
        )
    
    def _query(self) -> str:        
        excluded_columns_str = ", ".join([f"{col}" for col in self.config.excluded_columns])

        query = f"""
        SELECT * EXCEPT({excluded_columns_str}, future_value), future_value
        FROM {self.config.dataset_id}.{self.config.table_id}
        WHERE {self.config.train_val_test_split_column} IN ('TRAIN', 'VALIDATE')
        """

        if self.config.n_samples > 0:
            query += f" LIMIT {self.config.n_samples}"

        return query