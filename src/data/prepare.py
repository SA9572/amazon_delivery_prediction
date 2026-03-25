"""
prepare.py
----------
This script loads raw Amazon delivery data, performs cleaning,
and saves the cleaned dataset into `data/interim/`.

Usage:
    python src/data/prepare.py
"""

import os
import pandas as pd


# ==============================
# 1. Define Paths
# ==============================
RAW_DATA_PATH = "data/raw/amazon_delivery.csv"
OUTPUT_PATH = "data/interim/amazon_delivery_cleaned.csv"


# ==============================
# 2. Load Data
# ==============================
def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded raw data from {path} with shape {df.shape}")
    return df


# ==============================
# 3. Clean Data
# ==============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning on dataset"""

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values (basic strategy: drop rows with NA in target or key features)
    if "Delivery_Time" in df.columns:
        df = df.dropna(subset=["Delivery_Time"])
    else:
        raise KeyError("❌ Target column 'Delivery_Time' not found in dataset")

    # Fill missing categorical values with "Unknown"
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print("✅ Data cleaned successfully")
    return df


# ==============================
# 4. Save Data
# ==============================
def save_data(df: pd.DataFrame, path: str):
    """Save dataframe to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"📁 Cleaned data saved to {path}")


# ==============================
# 5. Main Function
# ==============================
def main():
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
