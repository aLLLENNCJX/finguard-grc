"""Pydantic data contracts for FinGuard-GRC."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


RiskTier = Literal["Critical", "High", "Medium", "Low"]
ReviewStatus = Literal[
    "Pending Review",
    "Confirmed Violation",
    "False Positive",
    "Remediation Assigned",
    "Escalated to CISO",
]


class AccessEvent(BaseModel):
    """Raw enterprise IAM & IT audit event."""

    event_id: str
    timestamp: str
    day_index: int = Field(ge=0)
    employee_id: str
    employee_role: str
    department: str
    system_name: str
    asset_criticality: int = Field(ge=1, le=5)
    action_type: str
    privilege_level_delta: int
    entitlement_count: int = Field(ge=1)
    days_since_last_review: int = Field(ge=0)
    is_off_hours: int = Field(ge=0, le=1)
    hour_of_day: int = Field(ge=0, le=23)
    records_exported: int = Field(ge=0)
    peer_group_volume_zscore: float
    failed_auth_24h: int = Field(ge=0)
    has_change_ticket: int = Field(ge=0, le=1)
    cross_dept_access: int = Field(ge=0, le=1)
    sod_rule_score: float = Field(ge=0.0, le=1.0)
    external_ai_endpoint_flag: int = Field(ge=0, le=1)
    dormancy_days: int = Field(ge=0)
    is_violation: int = Field(ge=0, le=1)
    violation_type: str
    event_summary: str


class ControlClause(BaseModel):
    """Regulatory governance control clause."""

    control_id: str
    framework: str
    domain: str
    title: str
    description: str
    keywords: list[str]
    violation_types: list[str]
    remediation_steps: list[str]
    estimated_effort_hours: int = Field(ge=1)
    regulatory_weight: float = Field(ge=0.1, le=1.0)


class AuditFinding(BaseModel):
    """Scored IT audit finding presented in the triage queue."""

    event_id: str
    timestamp: str
    employee_id: str
    department: str
    system_name: str
    asset_criticality: int
    action_type: str
    violation_type: str
    is_violation: int
    ml_probability: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_tier: RiskTier
    reason_codes: list[str]
    mapped_controls: list[str]
    review_status: ReviewStatus = "Pending Review"
    event_summary: str


class ReviewDecision(BaseModel):
    """Human-in-the-loop auditor triage decision payload."""

    status: ReviewStatus
    reviewer: str = Field(min_length=2, max_length=80)
    comment: str = Field(default="", max_length=600)


class ModelBenchmarkRow(BaseModel):
    """Evaluation metrics for one anomaly detection model."""

    model_name: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    f2: float
    fpr: float
    alert_compression_ratio: float
    analyst_hours_saved_quarter: float
    threshold: float


class RemediationScenario(BaseModel):
    """Parameters for What-If Remediation ROI simulation."""

    alert_threshold: float = Field(default=0.45, ge=0.10, le=0.95)
    budget_hours: int = Field(default=160, ge=20, le=600)
    domain_focus: Optional[str] = None
