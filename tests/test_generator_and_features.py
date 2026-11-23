"""Tests for synthetic financial IT audit generator and leak-free feature engineering."""

from __future__ import annotations

import pandas as pd

from finguard.data.generator import DEPARTMENTS, SYSTEMS, generate_audit_events
from finguard.features.engineering import (
    build_preprocessor,
    get_feature_columns,
    prepare_feature_frame,
)


def test_generator_determinism_and_schema() -> None:
    df1 = generate_audit_events(num_events=400, num_days=30, seed=42)
    df2 = generate_audit_events(num_events=400, num_days=30, seed=42)
    pd.testing.assert_frame_equal(df1, df2)

    assert len(df1) == 400
    assert set(df1["department"]).issubset(set(DEPARTMENTS))
    assert set(df1["system_name"]).issubset(set(SYSTEMS.keys()))
    violation_rate = df1["is_violation"].mean()
    assert 0.05 <= violation_rate <= 0.18


def test_feature_pipeline_has_no_target_leakage() -> None:
    raw = generate_audit_events(num_events=300, num_days=30, seed=42)
    feat = prepare_feature_frame(raw)
    cols = get_feature_columns()

    assert "is_violation" not in cols
    assert "violation_type" not in cols
    assert "hour_sin" in feat.columns and "hour_cos" in feat.columns
    assert "records_exported_log" in feat.columns

    prep = build_preprocessor()
    X_trans = prep.fit_transform(feat[cols])
    assert X_trans.shape[0] == len(raw)
    assert X_trans.shape[1] >= len(cols)