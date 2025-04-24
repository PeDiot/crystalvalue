from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    VarianceThreshold
)

class FeatureSelector:    
    def __init__(self, n_features: Optional[int] = None):
        self.n_features = n_features
        self.selector = None
        self.selected_features = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'FeatureSelector':
        raise NotImplementedError
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.selector is None:
            raise ValueError("Feature selector must be fitted before transforming data")
        return self.selector.transform(X)
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        return self.fit(X, y).transform(X)
    
    def get_selected_features(self, feature_names: List[str]) -> List[str]:
        if self.selector is None:
            raise ValueError("Feature selector must be fitted before getting selected features")
        if not hasattr(self.selector, 'get_support'):
            raise ValueError("This selector does not support getting selected features")
        return [feature_names[i] for i in range(len(feature_names)) 
                if self.selector.get_support()[i]]


class VarianceFeatureSelector(FeatureSelector):    
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'VarianceFeatureSelector':
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(X)
        return self


class FRegressionFeatureSelector(FeatureSelector):    
    def __init__(self, n_features: int):
        super().__init__(n_features)
        if n_features is None:
            raise ValueError("n_features must be specified for F-regression selector")
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'FRegressionFeatureSelector':
        self.selector = SelectKBest(f_regression, k=self.n_features)
        self.selector.fit(X, y)
        return self


class MutualInfoFeatureSelector(FeatureSelector):    
    def __init__(self, n_features: int):
        super().__init__(n_features)
        if n_features is None:
            raise ValueError("n_features must be specified for mutual information selector")
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'MutualInfoFeatureSelector':
        self.selector = SelectKBest(mutual_info_regression, k=self.n_features)
        self.selector.fit(X, y)
        return self


def create_feature_selector(
    method: str,
    n_features: Optional[int] = None,
    threshold: float = 0.0
) -> FeatureSelector:
    if method == "variance":
        return VarianceFeatureSelector(threshold=threshold)
    if method == "f_regression":
        return FRegressionFeatureSelector(n_features=n_features)
    if method == "mutual_info":
        return MutualInfoFeatureSelector(n_features=n_features)
    raise ValueError(f"Unknown feature selection method: {method}") 