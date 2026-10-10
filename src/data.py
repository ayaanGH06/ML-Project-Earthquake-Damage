"""Shared data loading, splitting and preprocessing.

EVERY notebook should use these functions so that both teammates train and
evaluate on exactly the same train/validation split.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "damage_grade"
ID_COL = "building_id"
GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]


def get_data_dir(data_dir=None):
    """Priority: argument > EQ_DATA_DIR environment variable > data/raw."""
    if data_dir is not None:
        return Path(data_dir)
    return Path(os.environ.get("EQ_DATA_DIR", "data/raw"))


def load_raw(data_dir=None):
    """Return (X_train_full, y_train_full, X_test). Test labels are not public."""
    d = get_data_dir(data_dir)
    X = pd.read_csv(d / "train_values.csv", index_col=ID_COL)
    y = pd.read_csv(d / "train_labels.csv", index_col=ID_COL)[TARGET]
    X_test = pd.read_csv(d / "test_values.csv", index_col=ID_COL)
    return X, y.loc[X.index], X_test


def train_val_split(X, y, val_size=0.2):
    """Stratified split with a fixed seed (same split for everyone)."""
    return train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=RANDOM_STATE
    )


def _column_groups(X):
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in X.columns if c not in categorical and c not in GEO_COLS]
    return numeric, categorical


def build_scaled_preprocessor(X):
    """For KNN and neural networks: one-hot categoricals + standardized numerics.

    Design choice: geo_level_1_id (31 regions) is one-hot encoded; geo_level_2/3
    (1,400 and 12,500 values) are dropped because one-hot would explode the
    feature count and using them as raw numbers is meaningless (they are IDs).
    Revisit this if you want to squeeze out more accuracy.
    """
    numeric, categorical = _column_groups(X)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical + ["geo_level_1_id"],
            ),
        ],
        remainder="drop",
    )


def build_tree_preprocessor(X):
    """For Random Forest / boosting: no scaling needed, ordinal-encode categoricals.

    Geo IDs are kept as plain numbers (trees can split on them).
    """
    numeric, categorical = _column_groups(X)
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("num", "passthrough", numeric + GEO_COLS),
        ],
        remainder="drop",
    )
