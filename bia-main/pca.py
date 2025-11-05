# Principal Component Analysis (PCA) for robust Dimensionality Reduction (Traditional Method).

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Union
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class PCASelector:


    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.pca_ = None
        self.explained_variance_ratio_ = None


    def fit(self, X: np.ndarray, y: np.ndarray = None):
        self.pca_ = PCA(n_components=self.variance_threshold, random_state=42)
        self.pca_.fit(X)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        return self
    

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise ValueError("PCA must be fitted first.")
        return self.pca_.transform(X)
    

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


    def get_num_components(self) -> int:
        if self.pca_ is None:
            return 0
        return self.pca_.n_components_
    

    def get_total_explained_variance(self) -> float:
        if self.explained_variance_ratio_ is None:
            return 0.0
        return float(np.sum(self.explained_variance_ratio_))