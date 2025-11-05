# L1 Regularization (Lasso Effect): Embedded Feature Selection for Classification. 

import pandas as pd
import numpy as np
import sys 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(file_path):
    # Safely loads the dataset.
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def apply_lasso_feature_selection(data_frame, target_column_name):
    # Separate Features (X) and Target (y)
    X = data_frame.drop(columns=[target_column_name, 'Name', 'Ticket', 'Cabin', 'PassengerId'], errors='ignore')
    y = data_frame[target_column_name]
    
    # Impute missing values (median for numeric, mode for categorical).
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
        elif X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0])
    
    X = pd.get_dummies(X, drop_first=True) 

    # Split Data FIRST to avoid data leakage.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize data: Fit only on the training set.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply L1 Logistic Regression (the Lasso selector).
    log_reg_l1 = LogisticRegression(
        penalty='l1', 
        C=0.1,  # C controls regularization strength
        solver='liblinear', 
        random_state=42, 
        max_iter=10000
        )
    log_reg_l1.fit(X_train_scaled, y_train)

    # Feature Selection Logic
    if log_reg_l1.coef_.ndim > 1:
        # Multiclass: Use mean of absolute coefficients for feature importance.
         coef_means = np.mean(np.abs(log_reg_l1.coef_), axis=0)
    else:
        # Binary: Use absolute coefficients directly.
        coef_means = np.abs(log_reg_l1.coef_)

    coefficients = pd.Series(coef_means, index=X_train.columns)
    
    # Select features that were not fully zeroed out.
    threshold = 1e-6 
    selected_features_names = list(coefficients[coefficients > threshold].index)
    
    # Retrain and Evaluate Final Model on the selected subset.
    if not selected_features_names:
        print("Warning: L1 Regularization selected zero features.")
        return [], 0.0

    # Map the selected feature names back to the scaled data matrices.
    X_train_final = X_train_scaled[:, coefficients.index.get_indexer(selected_features_names)]
    X_test_final = X_test_scaled[:, coefficients.index.get_indexer(selected_features_names)]

    final_model = LogisticRegression(max_iter=1000).fit(X_train_final, y_train)
    
    y_pred = final_model.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)

    return selected_features_names, accuracy