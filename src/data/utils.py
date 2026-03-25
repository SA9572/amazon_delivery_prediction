"""
utils.py
--------
Helper utility functions for Amazon Delivery Prediction project.
These functions are reusable across data preparation,
feature engineering, and modeling steps.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import Tuple


# =======================================================
# 1. Date & Time Utilities
# =======================================================
def parse_date_column(df: pd.DataFrame, col: str, fmt: str = None) -> pd.DataFrame:
    """
    Convert a column to datetime.
    Args:
        df: Input dataframe
        col: Column name
        fmt: Optional datetime format (e.g., '%Y-%m-%d')
    """
    try:
        df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
    except Exception:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def extract_date_parts(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """
    Extract year, month, day, weekday from a datetime column.
    Args:
        df: Input dataframe
        col: datetime column
        prefix: prefix for new columns
    """
    df[f"{prefix}_year"] = df[col].dt.year
    df[f"{prefix}_month"] = df[col].dt.month
    df[f"{prefix}_day"] = df[col].dt.day
    df[f"{prefix}_weekday"] = df[col].dt.weekday  # 0 = Monday
    return df


def extract_time_parts(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """
    Extract hour and minute from a time column.
    Args:
        df: Input dataframe
        col: time column
        prefix: prefix for new columns
    """
    df[f"{prefix}_hour"] = pd.to_datetime(df[col].astype(str), errors="coerce").dt.hour
    df[f"{prefix}_minute"] = pd.to_datetime(df[col].astype(str), errors="coerce").dt.minute
    return df


# =======================================================
# 2. Missing Value Handling
# =======================================================
def fill_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in categorical columns with 'Unknown'."""
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
    return df


def fill_missing_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in numeric columns with median."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


# =======================================================
# 3. Categorical Encoding
# =======================================================
def encode_categoricals(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Encode categorical variables using category codes.
    Args:
        df: Input dataframe
        cols: List of categorical columns
    """
    for col in cols:
        df[col] = df[col].astype("category").cat.codes
    return df


# =======================================================
# 4. Geospatial Utilities
# =======================================================
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance (km) between two points using geodesic.
    Args:
        lat1, lon1: Store coordinates
        lat2, lon2: Drop coordinates
    """
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).km
    except Exception:
        return np.nan


def add_distance_column(df: pd.DataFrame,
                        store_lat: str, store_lon: str,
                        drop_lat: str, drop_lon: str,
                        new_col: str = "Distance_km") -> pd.DataFrame:
    """Add geodesic distance column to dataframe."""
    df[new_col] = df.apply(
        lambda row: calculate_distance(row[store_lat], row[store_lon],
                                       row[drop_lat], row[drop_lon]),
        axis=1
    )
    return df
