"""
train.py
--------
Training script for Amazon Delivery Time Prediction.

- Loads processed dataset
- Trains multiple regressors
- Evaluates performance
- Logs results with MLflow
- Saves best model
"""

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# -------------------------------
# Helper Functions
# -------------------------------

def evaluate_model(model, X_test, y_test):
    """Return RMSE, MAE, R² for a fitted model."""
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2


# -------------------------------
# Main Training Logic
# -------------------------------

def main():
    print("📂 Loading processed dataset...")
    data_path = os.path.join("data", "processed", "amazon_delivery_processed.csv")
    df = pd.read_csv(data_path)

    # Features & target
    X = df.drop("Delivery_Time", axis=1)
    y = df["Delivery_Time"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        )

    results = []
    best_model = None
    best_rmse = float("inf")
    best_name = None

    print("\n🚀 Training models...\n")
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            rmse, mae, r2 = evaluate_model(model, X_test, y_test)

            # Log metrics
            mlflow.log_param("model_name", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Save model in MLflow
            mlflow.sklearn.log_model(model, artifact_path=name)

            print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            results.append((name, rmse, mae, r2))

            # Track best
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name

    # Save best model
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\n✅ Best Model: {best_name} (RMSE={best_rmse:.4f}) saved at {best_model_path}")


if __name__ == "__main__":
    main()
