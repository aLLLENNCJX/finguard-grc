"""Tests for 5-model 3-way chronological benchmark, non-trivial metrics, and composite risk scoring."""

from __future__ import annotations

from pathlib import Path
from finguard.data.generator import generate_audit_events
from finguard.models.benchmark import run_benchmark_and_train
from finguard.models.scoring import infer_primary_domain, map_risk_tier, score_events


def test_benchmark_outperforms_baseline_without_leakage(tmp_path: Path) -> None:
    df = generate_audit_events(num_events=1800, num_days=90, seed=42)
    results = run_benchmark_and_train(df, seed=42, output_dir=tmp_path)

    split = results["split_info"]
    assert split["train_events"] > 0 and split["val_events"] > 0 and split["test_events"] > 0
    assert split["train_events"] + split["val_events"] + split["test_events"] == len(df)

    rows = {r["model_name"]: r for r in results["benchmarks"]}
    assert len(rows) == 5

    heuristic = rows["Rule-Based SIEM Baseline"]
    champion = rows["HistGradientBoosting (Champion)"]

    # Realistic non-degenerate metrics that clearly beat SIEM heuristics
    assert 0.85 <= champion["roc_auc"] <= 0.995
    assert champion["f1"] > heuristic["f1"] + 0.12
    assert (tmp_path / "champion_model.joblib").exists()
    assert (tmp_path / "benchmark_results.json").exists()


def test_composite_risk_scoring_bounds_and_observable_domain() -> None:
    df = generate_audit_events(num_events=50, num_days=10, seed=42)
    probs = [0.92 if i == 0 else 0.05 for i in range(len(df))]
    scored = score_events(df, probs)

    assert (scored["risk_score"] >= 0.0).all()
    assert (scored["risk_score"] <= 100.0).all()
    assert "primary_domain" in scored.columns
    assert map_risk_tier(85.0) == "Critical"
    assert map_risk_tier(65.0) == "High"
    assert map_risk_tier(45.0) == "Medium"
    assert map_risk_tier(15.0) == "Low"
    assert all(isinstance(r, list) and len(r) >= 1 for r in scored["reason_codes"])

    # Verify infer_primary_domain works without violation_type
    row_no_label = df.iloc[0].drop(labels=["violation_type", "is_violation"])
    assert isinstance(infer_primary_domain(row_no_label), str)