"""Synthetic enterprise financial IT audit & IAM event generator."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
import numpy as np
import pandas as pd

from finguard.config import DEFAULT_NUM_DAYS, DEFAULT_NUM_EVENTS, DEFAULT_RANDOM_SEED

DEPARTMENTS = [
    "Asset Management",
    "Risk & Compliance",
    "Treasury & Markets",
    "IT Infrastructure",
    "Finance & Accounting",
    "Operations",
]

SYSTEMS: dict[str, int] = {
    "SWIFT Payment Gateway": 5,
    "Core Portfolio Ledger": 5,
    "Production IAM Console": 5,
    "Client KYC Vault": 4,
    "Risk Analytics Data Lake": 4,
    "ERP General Ledger": 3,
    "AI Research Sandbox": 3,
    "HR & Payroll Portal": 2,
}

ACTION_TYPES = [
    "read_records",
    "bulk_export",
    "privilege_grant",
    "config_change",
    "payment_approve",
    "external_ai_upload",
    "login_session",
]

ROLES_BY_DEPT = {
    "Asset Management": ["Portfolio Analyst", "Fund Manager", "Investment Ops"],
    "Risk & Compliance": ["Compliance Officer", "Market Risk Analyst", "Internal Auditor"],
    "Treasury & Markets": ["Treasury Dealer", "Settlement Specialist", "FX Analyst"],
    "IT Infrastructure": ["Cloud Engineer", "IAM Administrator", "Site Reliability Engineer"],
    "Finance & Accounting": ["Financial Controller", "GL Accountant", "Tax Analyst"],
    "Operations": ["Operations Associate", "KYC Specialist", "Vendor Coordinator"],
}

VIOLATION_ARCHETYPES = [
    "privilege_creep",
    "sod_conflict",
    "off_hours_exfiltration",
    "dormant_account_abuse",
    "unapproved_prod_change",
    "shadow_ai_leakage",
]


def _build_employees(rng: np.random.Generator, count: int = 120) -> list[dict[str, Any]]:
    employees: list[dict[str, Any]] = []
    for idx in range(count):
        dept = DEPARTMENTS[idx % len(DEPARTMENTS)]
        role = ROLES_BY_DEPT[dept][idx % len(ROLES_BY_DEPT[dept])]
        base_entitlements = int(rng.integers(4, 12))
        if dept == "IT Infrastructure":
            base_entitlements += 4
        employees.append(
            {
                "employee_id": f"EMP-{1001 + idx}",
                "department": dept,
                "employee_role": role,
                "base_entitlements": base_entitlements,
                "is_contractor": bool(rng.random() < 0.14),
            }
        )
    return employees


def generate_audit_events(
    num_events: int = DEFAULT_NUM_EVENTS,
    num_days: int = DEFAULT_NUM_DAYS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate deterministic enterprise financial IT audit & IAM access events.

    Includes realistic benign edge cases (approved weekend maintenance, quarterly
    compliance bulk exports, occasional failed logins) alongside 6 policy violation
    archetypes so that heuristic rules face real tradeoffs and supervised ML models
    can be rigorously evaluated without artificial target leakage.
    """
    rng = np.random.default_rng(seed)
    employees = _build_employees(rng, count=120)
    system_names = list(SYSTEMS.keys())
    start_dt = datetime(2026, 7, 1, 8, 0, 0)

    records: list[dict[str, Any]] = []
    violation_rate = 0.102

    for i in range(num_events):
        emp = employees[int(rng.integers(0, len(employees)))]
        day_index = int((i * num_days) / num_events)
        day_index = min(day_index, num_days - 1)

        is_violation = int(rng.random() < violation_rate)
        violation_type = "normal"

        # Default benign baselines
        dept = emp["department"]
        system_name = str(rng.choice(system_names))
        criticality = SYSTEMS[system_name]
        action_type = str(
            rng.choice(
                ["read_records", "login_session", "bulk_export", "privilege_grant", "config_change"],
                p=[0.52, 0.25, 0.10, 0.07, 0.06],
            )
        )
        hour = int(np.clip(rng.normal(13.5, 2.8), 7, 20))
        minute = int(rng.integers(0, 60))
        is_off_hours = int(hour < 8 or hour >= 20)

        privilege_delta = int(rng.choice([0, 0, 0, 1], p=[0.78, 0.10, 0.07, 0.05]))
        entitlement_count = max(2, int(emp["base_entitlements"] + rng.integers(-2, 4)))
        days_since_review = int(np.clip(rng.gamma(3.0, 18.0), 3, 140))
        records_exported = int(max(0, rng.exponential(180))) if action_type in {"read_records", "bulk_export"} else 0
        if action_type == "bulk_export":
            records_exported = int(rng.integers(450, 3200))
        peer_zscore = float(np.round(rng.normal(0.1, 0.85), 3))
        failed_auth = int(rng.choice([0, 0, 0, 1, 2], p=[0.74, 0.14, 0.07, 0.04, 0.01]))
        has_change_ticket = 1 if action_type in {"config_change", "privilege_grant"} else int(rng.random() < 0.25)
        cross_dept = int(rng.random() < 0.12)
        sod_score = float(np.round(np.clip(rng.beta(1.2, 12.0), 0.0, 0.45), 3))
        ext_ai_flag = 0
        dormancy_days = int(rng.choice([0, 0, 0, 5, 12], p=[0.80, 0.08, 0.05, 0.04, 0.03]))

        # Realistic benign edge cases (~24% of benign events create genuine FP/overlap tradeoffs)
        benign_edge = rng.random()
        if not is_violation and benign_edge < 0.065:
            # Approved weekend / off-hours IT maintenance with valid ticket
            dept = "IT Infrastructure"
            system_name = str(rng.choice(["Production IAM Console", "Core Portfolio Ledger", "ERP General Ledger"]))
            criticality = SYSTEMS[system_name]
            action_type = "config_change"
            hour = int(rng.choice([21, 22, 23, 0, 1, 2]))
            is_off_hours = 1
            has_change_ticket = 1 if rng.random() < 0.72 else 0
            privilege_delta = int(rng.choice([0, 1, 2], p=[0.35, 0.40, 0.25]))
            failed_auth = int(rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15]))
            sod_score = float(np.round(rng.uniform(0.20, 0.60), 3))
        elif not is_violation and benign_edge < 0.125:
            # Authorized quarterly HKMA regulatory bulk export by Risk & Compliance
            dept = "Risk & Compliance"
            system_name = str(rng.choice(["Client KYC Vault", "Risk Analytics Data Lake", "Core Portfolio Ledger"]))
            criticality = SYSTEMS[system_name]
            action_type = "bulk_export"
            records_exported = int(rng.integers(3800, 18500))
            peer_zscore = float(np.round(rng.uniform(1.6, 4.2), 3))
            has_change_ticket = 1 if rng.random() < 0.75 else 0
            hour = int(rng.choice([17, 18, 19, 20, 21, 22]))
            is_off_hours = int(hour >= 20)
        elif not is_violation and benign_edge < 0.17:
            # Legitimate internal AI Sandbox experimentation (including external API test with dummy data)
            system_name = str(rng.choice(["AI Research Sandbox", "Risk Analytics Data Lake"]))
            criticality = SYSTEMS[system_name]
            action_type = "external_ai_upload" if rng.random() < 0.45 else "read_records"
            ext_ai_flag = 1 if (action_type == "external_ai_upload" and rng.random() < 0.40) else 0
            records_exported = int(rng.integers(350, 3200))
            peer_zscore = float(np.round(rng.uniform(0.5, 2.5), 3))
        elif not is_violation and benign_edge < 0.24:
            # Benign emergency hotfix, temporary cross-dept project grant, or annual audit contractor login
            action_type = str(rng.choice(["config_change", "privilege_grant", "login_session", "payment_approve"]))
            has_change_ticket = 0 if action_type == "config_change" and rng.random() < 0.55 else 1
            cross_dept = 1
            days_since_review = int(rng.integers(75, 210))
            dormancy_days = int(rng.choice([0, 24, 38, 55, 85], p=[0.35, 0.20, 0.20, 0.15, 0.10]))
            sod_score = float(np.round(rng.uniform(0.28, 0.72), 3))
            privilege_delta = int(rng.choice([0, 1, 2], p=[0.30, 0.40, 0.30]))
            failed_auth = int(rng.choice([0, 1, 2, 3], p=[0.45, 0.25, 0.18, 0.12]))

        if is_violation:
            violation_type = str(
                rng.choice(
                    VIOLATION_ARCHETYPES,
                    p=[0.20, 0.18, 0.18, 0.14, 0.16, 0.14],
                )
            )
            if violation_type == "privilege_creep":
                action_type = "privilege_grant"
                system_name = str(rng.choice(["Production IAM Console", "Core Portfolio Ledger", "Client KYC Vault"]))
                criticality = SYSTEMS[system_name]
                privilege_delta = int(rng.choice([1, 2, 2, 3]))
                entitlement_count = int(rng.integers(14, 28))
                days_since_review = int(rng.integers(95, 290))
                cross_dept = int(rng.random() < 0.85)
                has_change_ticket = int(rng.random() < 0.35)
                sod_score = float(np.round(rng.uniform(0.28, 0.65), 3))
            elif violation_type == "sod_conflict":
                action_type = "payment_approve" if rng.random() < 0.6 else "privilege_grant"
                system_name = str(rng.choice(["SWIFT Payment Gateway", "Core Portfolio Ledger", "ERP General Ledger"]))
                criticality = SYSTEMS[system_name]
                sod_score = float(np.round(rng.uniform(0.62, 0.98), 3))
                cross_dept = int(rng.random() < 0.72)
                privilege_delta = int(rng.choice([1, 2]))
                has_change_ticket = int(rng.random() < 0.30)
            elif violation_type == "off_hours_exfiltration":
                action_type = "bulk_export"
                system_name = str(rng.choice(["Client KYC Vault", "Core Portfolio Ledger", "Risk Analytics Data Lake"]))
                criticality = SYSTEMS[system_name]
                hour = int(rng.choice([0, 1, 2, 3, 21, 22, 23]))
                is_off_hours = 1
                records_exported = int(rng.integers(4500, 26000))
                peer_zscore = float(np.round(rng.uniform(2.1, 5.2), 3))
                has_change_ticket = int(rng.random() < 0.15)
                failed_auth = int(rng.choice([0, 1, 2, 3]))
            elif violation_type == "dormant_account_abuse":
                action_type = str(rng.choice(["login_session", "read_records", "bulk_export"]))
                system_name = str(rng.choice(["Client KYC Vault", "HR & Payroll Portal", "Core Portfolio Ledger"]))
                criticality = SYSTEMS[system_name]
                dormancy_days = int(rng.integers(38, 195))
                failed_auth = int(rng.integers(1, 5))
                is_off_hours = int(rng.random() < 0.65)
                if is_off_hours:
                    hour = int(rng.choice([0, 2, 4, 21, 23]))
                records_exported = int(rng.integers(300, 5200))
            elif violation_type == "unapproved_prod_change":
                action_type = "config_change"
                system_name = str(rng.choice(["Production IAM Console", "SWIFT Payment Gateway", "Core Portfolio Ledger"]))
                criticality = SYSTEMS[system_name]
                has_change_ticket = 0
                privilege_delta = int(rng.choice([1, 2]))
                sod_score = float(np.round(rng.uniform(0.35, 0.78), 3))
                is_off_hours = int(rng.random() < 0.55)
                if is_off_hours:
                    hour = int(rng.choice([21, 22, 23, 1, 3]))
            elif violation_type == "shadow_ai_leakage":
                action_type = "external_ai_upload"
                system_name = str(rng.choice(["Client KYC Vault", "AI Research Sandbox", "Risk Analytics Data Lake"]))
                criticality = SYSTEMS[system_name]
                ext_ai_flag = 1
                records_exported = int(rng.integers(600, 8500))
                peer_zscore = float(np.round(rng.uniform(1.4, 3.8), 3))
                has_change_ticket = 0

            # Subtle / stealth violations (~40% of violations) + realistic feature overlap
            if rng.random() < 0.40:
                is_off_hours = int(rng.random() < 0.18)
                if not is_off_hours:
                    hour = int(rng.integers(9, 18))
                peer_zscore = float(np.round(rng.uniform(-0.6, 1.2), 3))
                records_exported = min(records_exported, int(rng.integers(120, 1800)))
                failed_auth = int(rng.choice([0, 0, 1]))
                has_change_ticket = 1 if rng.random() < 0.72 else 0
                dormancy_days = min(dormancy_days, int(rng.integers(0, 26)))
                ext_ai_flag = 0 if rng.random() < 0.65 else ext_ai_flag
                privilege_delta = int(rng.choice([0, 1]))
                days_since_review = int(rng.integers(15, 92))
                entitlement_count = max(3, entitlement_count - int(rng.integers(3, 8)))
                cross_dept = int(rng.random() < 0.35)
                if violation_type == "sod_conflict":
                    sod_score = float(np.round(rng.uniform(0.18, 0.46), 3))
                else:
                    sod_score = float(np.round(rng.uniform(0.05, 0.35), 3))

        ts = start_dt + timedelta(days=day_index, hours=hour - 8, minutes=minute)
        summary = (
            f"{emp['employee_id']} ({dept}) performed {action_type} on {system_name} "
            f"(criticality={criticality}, exported={records_exported}, ticket={has_change_ticket}, "
            f"sod={sod_score:.2f}, dormant={dormancy_days}d, ext_ai={ext_ai_flag})"
        )

        records.append(
            {
                "event_id": f"EVT-{100001 + i}",
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "day_index": day_index,
                "employee_id": emp["employee_id"],
                "employee_role": emp["employee_role"],
                "department": dept,
                "system_name": system_name,
                "asset_criticality": criticality,
                "action_type": action_type,
                "privilege_level_delta": privilege_delta,
                "entitlement_count": entitlement_count,
                "days_since_last_review": days_since_review,
                "is_off_hours": is_off_hours,
                "hour_of_day": hour,
                "records_exported": records_exported,
                "peer_group_volume_zscore": peer_zscore,
                "failed_auth_24h": failed_auth,
                "has_change_ticket": has_change_ticket,
                "cross_dept_access": cross_dept,
                "sod_rule_score": sod_score,
                "external_ai_endpoint_flag": ext_ai_flag,
                "dormancy_days": dormancy_days,
                "is_violation": is_violation,
                "violation_type": violation_type,
                "event_summary": summary,
            }
        )

    return pd.DataFrame(records)
