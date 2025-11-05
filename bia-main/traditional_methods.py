# traditional_methods.py
from lasso_expert import apply_lasso_feature_selection
from chi_square import ChiSquareSelector
from pca import PCASelector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def run_traditional_method(file_path, target_column, method="lasso"):
    df = pd.read_csv(file_path)

    if method == "lasso":
        selected_features, accuracy = apply_lasso_feature_selection(df, target_column)
        return {"method": "lasso", "selected_features": selected_features, "accuracy": accuracy}

    # Split common preprocessing for PCA and Chi-square
    X = df.drop(columns=[target_column, 'Name', 'Ticket', 'Cabin', 'PassengerId'], errors='ignore')
    y = df[target_column]
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if method == "pca":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca_selector = PCASelector(variance_threshold=0.95)
        X_train_pca = pca_selector.fit_transform(X_train_scaled)
        X_test_pca = pca_selector.transform(X_test_scaled)
        model = LogisticRegression(max_iter=1000).fit(X_train_pca, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_pca))
        return {
            "method": "pca",
            "components": pca_selector.get_num_components(),
            "accuracy": acc,
            "explained_variance": pca_selector.get_total_explained_variance()
        }

    elif method == "chi_square":
        chi_selector = ChiSquareSelector(k_features=10)
        X_train_sel = chi_selector.fit_transform(X_train, y_train)
        X_test_sel = chi_selector.transform(X_test)
        if X_train_sel.empty:
            return {"method": "chi_square", "selected_features": [], "accuracy": 0.0}
        model = LogisticRegression(max_iter=1000).fit(X_train_sel, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_sel))
        return {
            "method": "chi_square",
            "selected_features": chi_selector.get_selected_features_names(),
            "accuracy": acc
        }

    else:
        raise ValueError(f"Unknown method: {method}")

