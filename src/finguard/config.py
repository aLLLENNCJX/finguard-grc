"""Centralized configuration for FinGuard-GRC."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base" / "regulatory_controls.json"
GENERATED_DIR = DATA_DIR / "generated"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_ASSETS_DIR = DOCS_DIR / "assets"
WEB_DIR = PACKAGE_ROOT / "web"
DASHBOARD_TEMPLATE_PATH = WEB_DIR / "dashboard.html"

DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_EVENTS = 3600
DEFAULT_NUM_DAYS = 90
TRAIN_SPLIT_RATIO = 0.75

RISK_TIER_THRESHOLDS = {
    "Critical": 80.0,
    "High": 60.0,
    "Medium": 40.0,
    "Low": 0.0,
}

GOVERNANCE_DOMAINS = [
    "Access & Identity",
    "Change & Operations",
    "Data Protection",
    "AI & Third-Party Governance",
    "Monitoring & Audit",
]


@dataclass(frozen=True)
class LLMConfig:
    """Optional OpenAI-compatible LLM advisor configuration (disabled by default)."""

    enabled: bool
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        enabled = os.getenv("FINGUARD_LLM_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        api_key = os.getenv("FINGUARD_LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
        base_url = os.getenv(
            "FINGUARD_LLM_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        model = os.getenv("FINGUARD_LLM_MODEL", "gpt-4o-mini").strip()
        return cls(enabled=enabled and bool(api_key), api_key=api_key, base_url=base_url, model=model)


def ensure_directories() -> None:
    """Ensure runtime output directories exist."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
