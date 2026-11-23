"""Leak-free feature engineering pipeline for IT audit anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "privilege_level_delta",
    "entitlement_count",
    "days_since_last_review",
    "is_off_hours",
    "hour_sin",
    "hour_cos",
    "peer_group_volume_zscore",
    "records_exported_log",
    "failed_auth_24h",
    "has_change_ticket",
    "cross_dept_access",
    "sod_rule_score",
    "external_ai_endpoint_flag",
    "dormancy_days",
    "asset_criticality",
]

CATEGORICAL_FEATURES = [
    "department",
    "system_name",
    "action_type",
]


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derive deterministic, leak-free engineered columns from raw events."""
    out = df.copy()
    radians = 2.0 * np.pi * (out["hour_of_day"].astype(float) / 24.0)
    out["hour_sin"] = np.sin(radians)
    out["hour_cos"] = np.cos(radians)
    out["records_exported_log"] = np.log1p(out["records_exported"].clip(lower=0).astype(float))
    return out


def build_preprocessor() -> ColumnTransformer:
    """Create scikit-learn ColumnTransformer (fit ONLY on training split)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_columns() -> list[str]:
    """Return raw + engineered input columns required by the preprocessor."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
