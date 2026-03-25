"""
engineering.py
--------------
Feature engineering script for Amazon Delivery Time Prediction.

Steps:
1. Load cleaned data (from data/interim/)
2. Create new features:
   - Geospatial distance
   - Date & time features
   - Encoded categoricals
   - Agent age groups
   - Speed estimate
3. Save final processed dataset to data/processed/
"""

import os
import pandas as pd
import numpy as np
from src.data.utils import (
    parse_date_column,
    extract_date_parts,
    extract_time_parts,
    encode_categoricals,
    add_distance_column,
)


# ==============================
# 1. Define Paths
# ==============================
INPUT_PATH = "data/interim/amazon_delivery_cleaned.csv"
OUTPUT_PATH = "data/processed/amazon_delivery_processed.csv"


# ==============================
# 2. Load Data
# ==============================
def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded cleaned data with shape {df.shape}")
    return df


# ==============================
# 3. Feature Engineering
# ==============================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on dataset"""

    # --- Geospatial distance
    df = add_distance_column(df,
                             store_lat="Store_Latitude", store_lon="Store_Longitude",
                             drop_lat="Drop_Latitude", drop_lon="Drop_Longitude",
                             new_col="Distance_km")

    # --- Convert dates & times
    df = parse_date_column(df, "Order_Date")
    df = extract_date_parts(df, "Order_Date", "Order")
    df = extract_time_parts(df, "Order_Time", "Order")

    # --- Encode categoricals
    categorical_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
    df = encode_categoricals(df, categorical_cols)

    # --- Agent age groups
    if "Agent_Age" in df.columns:
        df["Agent_Age_Group"] = pd.cut(
            df["Agent_Age"],
            bins=[18, 25, 35, 45, 60],
            labels=["18-25", "26-35", "36-45", "46-60"]
        )
        df["Agent_Age_Group"] = df["Agent_Age_Group"].astype("category").cat.codes

    # --- Speed estimate (Delivery_Time in hours, distance in km)
    if "Delivery_Time" in df.columns:
        df["Speed_kmph"] = df["Distance_km"] / (df["Delivery_Time"] + 1e-5)

    print("✅ Feature engineering complete")
    return df


# ==============================
# 4. Save Data
# ==============================
def save_data(df: pd.DataFrame, path: str):
    """Save dataframe to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"📁 Processed data saved to {path}")


# ==============================
# 5. Main
# ==============================
def main():
    df = load_data(INPUT_PATH)
    df = feature_engineering(df)
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
