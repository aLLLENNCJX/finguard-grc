"""Enterprise IT Governance Maturity, Risk Register & Remediation ROI Simulator."""

from __future__ import annotations

from typing import Any
import pandas as pd

from finguard.config import GOVERNANCE_DOMAINS
from finguard.models.scoring import infer_primary_domain


def compute_domain_maturity(scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute 0-100 control maturity scores across the 5 IT governance domains.

    Uses ONLY observable telemetry and inferred primary_domain (zero reliance on
    ground-truth violation_type labels).
    """
    df = scored_df.copy()
    if "primary_domain" not in df.columns:
        df["primary_domain"] = df.apply(infer_primary_domain, axis=1)

    high_risk_df = df[df["risk_score"] >= 60.0]
    total_events = max(1, len(df))
    results: list[dict[str, Any]] = []

    for domain in GOVERNANCE_DOMAINS:
        relevant = high_risk_df[high_risk_df["primary_domain"] == domain]
        incident_density = len(relevant) / total_events
        avg_sev = float(relevant["risk_score"].mean()) / 100.0 if len(relevant) else 0.0
        penalty = min(48.0, incident_density * 620.0 + avg_sev * 18.0)
        maturity = round(float(max(38.0, 96.0 - penalty)), 1)
        results.append(
            {
                "domain": domain,
                "maturity_score": maturity,
                "open_high_risk_findings": int(len(relevant)),
                "status": "Strong" if maturity >= 80 else ("Moderate" if maturity >= 65 else "Needs Attention"),
            }
        )
    return results


def compute_risk_heatmap(scored_df: pd.DataFrame) -> dict[str, Any]:
    """Compute Department x System average risk score & critical finding counts."""
    pivot = (
        scored_df.groupby(["department", "system_name"])
        .agg(
            avg_risk=("risk_score", "mean"),
            high_count=("risk_score", lambda s: int((s >= 60.0).sum())),
            total=("event_id", "count"),
        )
        .reset_index()
    )
    cells = [
        {
            "department": str(row["department"]),
            "system_name": str(row["system_name"]),
            "avg_risk": round(float(row["avg_risk"]), 1),
            "high_count": int(row["high_count"]),
            "total": int(row["total"]),
        }
        for _, row in pivot.iterrows()
    ]
    return {"cells": cells}


def simulate_remediation(
    scored_df: pd.DataFrame,
    alert_threshold: float = 0.45,
    budget_hours: int = 160,
    domain_focus: str | None = None,
    quarterly_scale_factor: float = 1.0,
    cohort_label: str = "Holdout Test Cohort (Annualized to 90-Day Quarter)",
) -> dict[str, Any]:
    """Simulate What-If tradeoff between alert threshold, analyst capacity, and risk reduction."""
    df = scored_df.copy()
    if "primary_domain" not in df.columns:
        df["primary_domain"] = df.apply(infer_primary_domain, axis=1)

    active_domain = domain_focus if (domain_focus and domain_focus != "All") else "All"
    if active_domain != "All":
        df = df[df["primary_domain"] == active_domain]

    if df.empty:
        return {
            "alert_threshold": round(float(alert_threshold), 2),
            "budget_hours": int(budget_hours),
            "domain_focus": active_domain,
            "cohort_label": cohort_label,
            "flagged_alerts": 0,
            "true_positives_caught": 0,
            "false_positives": 0,
            "missed_violations": 0,
            "recall": 0.0,
            "precision": 0.0,
            "hours_required": 0.0,
            "capacity_coverage_pct": 100.0,
            "risk_reduction_pct": 0.0,
            "residual_critical_findings": 0,
            "avoided_exposure_hkd": 0,
            "residual_exposure_hkd": 0,
        }

    flagged = df[df["ml_probability"] >= alert_threshold]
    y_true = df["is_violation"].values.astype(int)
    y_pred = (df["ml_probability"].values >= alert_threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total_pos = max(1, int(y_true.sum()))

    recall = round(tp / total_pos, 4)
    precision = round(tp / max(1, tp + fp), 4)

    # Scale alert volume & remediation hours to a 90-day quarter when simulating on holdout test window
    scaled_flagged = int(round(len(flagged) * quarterly_scale_factor))
    hours_required = round(scaled_flagged * 0.35, 1)
    capacity_coverage = min(1.0, budget_hours / max(1.0, hours_required))
    remediated_tp = int(round(tp * capacity_coverage))

    # Each remediated critical/high violation avoids ~$18,500 HKD expected regulatory/operational loss
    exposure_per_violation_hkd = 18500
    scaled_total_pos = int(round(total_pos * quarterly_scale_factor))
    scaled_remediated_tp = int(round(remediated_tp * quarterly_scale_factor))
    baseline_exposure_hkd = int(scaled_total_pos * exposure_per_violation_hkd)
    avoided_exposure_hkd = int(scaled_remediated_tp * exposure_per_violation_hkd)
    residual_exposure_hkd = max(0, baseline_exposure_hkd - avoided_exposure_hkd)

    risk_reduction_pct = round(100.0 * (remediated_tp / total_pos), 1)
    residual_critical = int(
        max(0, round((df["risk_tier"] == "Critical").sum() * quarterly_scale_factor) - round(scaled_remediated_tp * 0.55))
    )

    return {
        "alert_threshold": round(float(alert_threshold), 2),
        "budget_hours": int(budget_hours),
        "domain_focus": active_domain,
        "cohort_label": cohort_label,
        "flagged_alerts": scaled_flagged,
        "true_positives_caught": int(round(tp * quarterly_scale_factor)),
        "false_positives": int(round(fp * quarterly_scale_factor)),
        "missed_violations": int(round(fn * quarterly_scale_factor)),
        "recall": recall,
        "precision": precision,
        "hours_required": hours_required,
        "capacity_coverage_pct": round(capacity_coverage * 100.0, 1),
        "risk_reduction_pct": risk_reduction_pct,
        "residual_critical_findings": residual_critical,
        "avoided_exposure_hkd": avoided_exposure_hkd,
        "residual_exposure_hkd": residual_exposure_hkd,
    }
