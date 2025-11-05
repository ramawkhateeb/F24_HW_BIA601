# Feature Comparison Script , This script executes and reports the performance of three classic feature methods for comparison.

import sys
import pandas as pd
from lasso_expert import apply_lasso_feature_selection
from chi_square import ChiSquareSelector
from pca import PCASelector # Ensure PCASelector class is imported correctly
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_and_split(data_frame, target_column_name):

    # Data Cleaning: Remove non-feature columns (Name, Ticket, ..etc)
    X = data_frame.drop(columns=[target_column_name, 'Name', 'Ticket', 'Cabin', 'PassengerId'], errors='ignore')
    y = data_frame[target_column_name]
    
    # 1. Simple Imputation
    for col in X.columns:          
          if X[col].dtype in ['int64', 'float64']:
              X.loc[:, col] = X[col].fillna(X[col].median())             
          elif X[col].dtype == 'object':
               X.loc[:, col] = X[col].fillna(X[col].mode()[0])

    # 2. One-Hot Encoding for categorical features.
    X = pd.get_dummies(X, drop_first=True) 
    
    # 3. Split Data into Train and Test Sets
    return train_test_split(X, y, test_size=0.3, random_state=42)



# Apply PCA (Dimensionality Reduction)
def apply_pca_comparison(X_train, X_test, y_train, y_test):  

    # PCA requires scaled data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use PCASelector (95% variance retained).
    pca_selector = PCASelector(variance_threshold=0.95)
    X_train_pca = pca_selector.fit_transform(X_train_scaled, y_train)
    X_test_pca = pca_selector.transform(X_test_scaled)
    
    # Final evaluation model.
    final_model = LogisticRegression(max_iter=1000).fit(X_train_pca, y_train)
    y_pred = final_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)   
    num_components = pca_selector.get_num_components()
    
    return num_components, accuracy


# Apply Chi-square (Filter Method)
def apply_chi_square_comparison(X_train, X_test, y_train, y_test):

    # Chi-square selector targets the top 10 features. 
    chi_selector = ChiSquareSelector(k_features=10)
    
    # Fit and transform (ChiSquareSelector handles internal scaling for non-negativity).
    X_train_selected = chi_selector.fit_transform(X_train, y_train)
    X_test_selected = chi_selector.transform(X_test)
    
    # Final evaluation model.
    if X_train_selected.empty:
        return [], 0.0
    
    final_model = LogisticRegression(max_iter=1000).fit(X_train_selected, y_train)
    y_pred = final_model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    return chi_selector.get_selected_features_names(), accuracy


# Main Execution
if __name__ == "__main__":
    
    # Validate command line arguments
    if len(sys.argv) != 3:
        print("Error: Please provide the data file path and the target column name.")
        print("Usage: python comparator.py <data_file_name> <target_column_name>")
        sys.exit(1)

    # Read arguments
    data_path = sys.argv[1]
    target_name = sys.argv[2]
    
    print(f"\n--- Analyzing File: {data_path}, Target Column: {target_name} ---\n")

    project_data = load_data(data_path)


    if project_data is not None:
        
        # --- Stage 1: Lasso Calculation ---
        lasso_features, lasso_accuracy = apply_lasso_feature_selection(project_data, target_name)

        # --- Stage 2: Unified Preprocessing and Split for PCA/Chi-Square --- 
        X_train, X_test, y_train, y_test = preprocess_and_split(project_data, target_name)

        # Calculate PCA results
        pca_components, pca_accuracy = apply_pca_comparison(X_train, X_test, y_train, y_test)

        # Calculate Chi-square results
        chi_features, chi_accuracy = apply_chi_square_comparison(X_train, X_test, y_train, y_test)
        
        # --- Stage 3: Comparison Report ---      
        print("="*55)
        print(" TRADITIONAL FEATURE SELECTION METHODS REPORT: ")
        print("="*55 + "\n")
        print(f"Data file: {data_path}")
        print(f"Target Column: {target_name}")
        print("_"*55)        
        print("\nStart loading and processing data.........")

        # Lasso Regression 
        print(f"\n| 1. L1 Regularization (Lasso Effect):")
        print(f"|    > Final Model: Logistic Regression")
        print(f"|    > Features Selected: {len(lasso_features)} features")
        print(f"|    > Model Accuracy: {lasso_accuracy:.4f}")
        
        if len(lasso_features) < 50:
            print(f"|    > Identified Feature List: {lasso_features}")
        else:
            print(f"|    > Identified Feature List: (List contains {len(lasso_features)} features, omitted for brevity).")
          
        print("__"*50 + "\n")
        

        # PCA 
        print(f"| 2. Principal Component Analysis (PCA):")
        print(f"|    > Components Retained: {pca_components} components (Retaining 95% Variance)")
        print(f"|    > Model Accuracy: {pca_accuracy:.4f}")    
        print("__"*50 + "\n")
        

        # Chi-square 
        print(f"| 3. Chi-square Test:")
        print(f"|    > Features Selected: {len(chi_features)} features (Top 10)")
        print(f"|    > Model Accuracy: {chi_accuracy:.4f}")
        print(f"|    > Identified Feature List: {chi_features}")
        print("__"*50)