"""
selectors.py
------------
Module for feature and model selection.

- Feature selection using correlation, variance, or model-based importance.
- Model selection using cross-validation with multiple regressors.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# -------------------------------
# Feature Selection Functions
# -------------------------------

def remove_low_variance_features(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove features with variance lower than the threshold.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support(indices=True)]
    return pd.DataFrame(X_reduced, columns=selected_features)


def select_k_best_features(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    Select top-k features based on univariate regression scores.
    """
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return pd.DataFrame(X_new, columns=selected_features)


def model_based_feature_importance(X: pd.DataFrame, y: pd.Series, top_n: int = 10) -> list:
    """
    Use RandomForestRegressor to get top N important features.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(top_n).index.tolist()
    return top_features


# -------------------------------
# Model Selection Functions
# -------------------------------

def compare_models(X: pd.DataFrame, y: pd.Series, scoring: str = "neg_root_mean_squared_error") -> pd.DataFrame:
    """
    Compare regression models and return cross-validation scores.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        results[name] = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
        }

    return pd.DataFrame(results).T


def train_best_model(X: pd.DataFrame, y: pd.Series, save_path: str = "models/best_model.pkl"):
    """
    Train and save the best-performing model.
    """
    model_scores = compare_models(X, y)
    best_model_name = model_scores["mean_score"].idxmax()

    if best_model_name == "LinearRegression":
        best_model = LinearRegression()
    else:
        best_model = RandomForestRegressor(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    print(f"Best Model: {best_model_name} | Test RMSE: {rmse:.4f}")

    # Save model
    joblib.dump(best_model, save_path)
    print(f"Model saved at: {save_path}")

    return best_model, rmse
