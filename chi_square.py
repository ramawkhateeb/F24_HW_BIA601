# Chi-Square Test (Filter Method): Statistically evaluates feature importance independently of the final model.

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class ChiSquareSelector:
    def __init__(self, k_features: int = 10):
        self.k_features = k_features
        self.selector_ = None
        self.scaler_ = None
        self.selected_features_names_ = None


    def fit(self, X: pd.DataFrame, y: pd.Series):
        # CRITICAL: Scale data to the [0, 1] range to satisfy the non-negativity requirement of the chi2 test.
        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Apply SelectKBest using chi2 score function to rank features.
        self.selector_ = SelectKBest(score_func=chi2, k=min(self.k_features, X.shape[1]))
        self.selector_.fit(X_scaled, y)
        
        # Identify selected feature names using the fitted mask.
        selected_mask = self.selector_.get_support()
        self.selected_features_names_ = list(X.columns[selected_mask])
        
        return self
    

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selector_ is None:
            raise ValueError("fit() must be called first.")
        
        if not self.selected_features_names_:
            return pd.DataFrame()
        
        # Apply the same scaling learned during the fit step.
        X_scaled = self.scaler_.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled[self.selected_features_names_]
   
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


    def get_selected_features_names(self) -> List[str]:
        return self.selected_features_names_