"""Composite IT Governance Risk Scoring & Explainable Reason Codes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finguard.config import RISK_TIER_THRESHOLDS


def map_risk_tier(score: float) -> str:
    """Map 0-100 risk score to discrete governance risk tier."""
    if score >= RISK_TIER_THRESHOLDS["Critical"]:
        return "Critical"
    if score >= RISK_TIER_THRESHOLDS["High"]:
        return "High"
    if score >= RISK_TIER_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


def compute_policy_signal(row: pd.Series) -> float:
    """Compute normalized 0-1 deterministic policy & SoD severity signal."""
    signal = 0.0
    if float(row.get("sod_rule_score", 0.0)) >= 0.50:
        signal += 0.35
    if int(row.get("external_ai_endpoint_flag", 0)) == 1:
        signal += 0.35
    if str(row.get("action_type", "")) == "config_change" and int(row.get("has_change_ticket", 1)) == 0:
        signal += 0.30
    if int(row.get("dormancy_days", 0)) >= 30:
        signal += 0.25
    if int(row.get("is_off_hours", 0)) == 1 and float(row.get("peer_group_volume_zscore", 0.0)) >= 2.0:
        signal += 0.25
    if int(row.get("privilege_level_delta", 0)) >= 2 and int(row.get("days_since_last_review", 0)) >= 90:
        signal += 0.25
    return float(min(1.0, signal))


def extract_reason_codes(row: pd.Series) -> list[str]:
    """Produce human-readable auditor reason codes for an access event."""
    reasons: list[str] = []
    if float(row.get("sod_rule_score", 0.0)) >= 0.50:
        reasons.append(f"SoD conflict score elevated ({float(row['sod_rule_score']):.2f})")
    if int(row.get("external_ai_endpoint_flag", 0)) == 1:
        reasons.append("Unapproved external AI endpoint egress detected")
    if str(row.get("action_type", "")) == "config_change" and int(row.get("has_change_ticket", 1)) == 0:
        reasons.append("Production configuration change without ITSM ticket")
    if int(row.get("dormancy_days", 0)) >= 30:
        reasons.append(f"Dormant account reactivated after {int(row['dormancy_days'])} days")
    if int(row.get("is_off_hours", 0)) == 1 and int(row.get("records_exported", 0)) >= 2500:
        reasons.append(
            f"Off-hours bulk export ({int(row['records_exported']):,} records, z={float(row['peer_group_volume_zscore']):.1f})"
        )
    if int(row.get("privilege_level_delta", 0)) >= 2 or (
        int(row.get("cross_dept_access", 0)) == 1 and int(row.get("days_since_last_review", 0)) >= 90
    ):
        reasons.append(
            f"Uncertified privilege expansion (delta=+{int(row['privilege_level_delta'])}, {int(row['days_since_last_review'])}d since review)"
        )
    if int(row.get("failed_auth_24h", 0)) >= 3:
        reasons.append(f"Repeated failed authentication attempts ({int(row['failed_auth_24h'])})")
    if not reasons:
        reasons.append("Behavioral deviation relative to departmental peer baseline")
    return reasons[:3]


def infer_primary_domain(row: pd.Series) -> str:
    """Infer governance domain strictly from observable telemetry (never ground-truth labels)."""
    if int(row.get("external_ai_endpoint_flag", 0)) == 1 or str(row.get("action_type", "")) == "external_ai_upload":
        return "AI & Third-Party Governance"
    if str(row.get("action_type", "")) == "config_change" and int(row.get("has_change_ticket", 1)) == 0:
        return "Change & Operations"
    if (
        float(row.get("sod_rule_score", 0.0)) >= 0.48
        or int(row.get("privilege_level_delta", 0)) >= 2
        or (str(row.get("action_type", "")) == "privilege_grant" and int(row.get("days_since_last_review", 0)) >= 90)
    ):
        return "Access & Identity"
    if int(row.get("records_exported", 0)) >= 2500 or (
        int(row.get("is_off_hours", 0)) == 1 and float(row.get("peer_group_volume_zscore", 0.0)) >= 1.5
    ):
        return "Data Protection"
    if int(row.get("dormancy_days", 0)) >= 25 or int(row.get("failed_auth_24h", 0)) >= 2:
        return "Monitoring & Audit"

    action = str(row.get("action_type", ""))
    system = str(row.get("system_name", ""))
    if action in {"privilege_grant", "payment_approve"} or "IAM" in system:
        return "Access & Identity"
    if action == "config_change":
        return "Change & Operations"
    if action == "bulk_export" or "KYC" in system or "Data Lake" in system:
        return "Data Protection"
    return "Monitoring & Audit"


def score_events(df: pd.DataFrame, ml_probabilities: np.ndarray) -> pd.DataFrame:
    """Compute composite 0-100 IT Governance Risk Score, tier, domain, and reason codes."""
    scored = df.copy()
    probs = np.clip(np.asarray(ml_probabilities, dtype=float), 0.0, 1.0)
    criticality_norm = (scored["asset_criticality"].astype(float) - 1.0) / 4.0
    policy_signals = scored.apply(compute_policy_signal, axis=1).astype(float)

    composite = 100.0 * (0.60 * probs + 0.20 * criticality_norm + 0.20 * policy_signals)
    scored["ml_probability"] = np.round(probs, 4)
    scored["risk_score"] = np.round(np.clip(composite, 0.0, 100.0), 2)
    scored["risk_tier"] = scored["risk_score"].apply(map_risk_tier)
    scored["primary_domain"] = scored.apply(infer_primary_domain, axis=1)
    scored["reason_codes"] = scored.apply(extract_reason_codes, axis=1)
    return scored
