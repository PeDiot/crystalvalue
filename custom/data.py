from typing import Tuple, Optional, Any, List

import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .config import Config


class DataLoader:
    def __init__(
        self, 
        config: Config, 
        preprocessor: Optional[ColumnTransformer] = None, 
        gcp_credentials: Optional[Any] = None
    ):
        self.config = config
        self.preprocessor = preprocessor
        self.gcp_credentials = gcp_credentials

        self._training_data = None
        self._test_data = None
        self._feature_names = None

    @property
    def training_data(self) -> pd.DataFrame:
        return self._training_data
    
    @property
    def test_data(self) -> pd.DataFrame:
        return self._test_data
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
        
    def load_training_data(
        self, only_positive_samples: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = self._run_query(is_training=True, only_positive_samples=only_positive_samples)
        self._training_data = df

        X, y, train_val_split = self._select_features(df)
        X_train, X_val, y_train, y_val = self._train_val_split(X, y, train_val_split)
        X_train, X_val = self._preprocess(X_train, X_val)

        return X_train, X_val, y_train, y_val
    
    def load_test_data(
        self
    ) -> Tuple[np.ndarray, np.ndarray]:
        df = self._run_query(is_training=False)
        self._test_data = df

        X, y, _ = self._select_features(df)        
        X = self.preprocessor.transform(X)
        
        return X, y
    
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
        cat_transformer = ("cat", OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        
        self.preprocessor = ColumnTransformer(
            transformers=[num_transformer, cat_transformer]
        )
        
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_val_scaled = self.preprocessor.transform(X_val)

        return X_train_scaled, X_val_scaled
    
    def _select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        train_val_split = df[self.config.train_val_test_split_column]
        
        to_drop_columns = [self.config.target_column, self.config.train_val_test_split_column]
        self._feature_names = [col for col in df.columns if col not in to_drop_columns]

        X = df.drop(columns=to_drop_columns)
        y = df[self.config.target_column]

        return X, y, train_val_split    
    
    def _run_query(self, is_training: bool, only_positive_samples: bool = False) -> pd.DataFrame:
        query = self._query(is_training, only_positive_samples)
        
        return pd.read_gbq(
            query=query,
            project_id=self.config.project_id,
            use_bqstorage_api=True, 
            progress_bar_type="tqdm", 
            credentials=self.gcp_credentials
        )
    
    def _query(self, is_training: bool, only_positive_samples: bool = False) -> str:        
        excluded_columns_str = ", ".join([f"{col}" for col in self.config.excluded_columns])

        if is_training:
            where_clause = f"WHERE {self.config.train_val_test_split_column} IN ('TRAIN', 'VALIDATE')"
        else:
            where_clause = f"WHERE {self.config.train_val_test_split_column} = 'TEST'"

        if only_positive_samples:
            where_clause += f" AND future_value_classification = 1"

        query = f"""
        SELECT * EXCEPT({excluded_columns_str}, {self.config.target_column}), {self.config.target_column}
        FROM {self.config.dataset_id}.{self.config.table_id}
        {where_clause}
        """

        if is_training and self.config.n_samples > 0:
            query += f" LIMIT {self.config.n_samples}"

        return query