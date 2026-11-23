"""Deterministic Audit Brief Generator with Optional OpenAI-Compatible LLM Advisor."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from finguard.config import LLMConfig


def generate_audit_brief(
    finding: dict[str, Any],
    matched_controls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce a concise executive remediation brief (deterministic by default)."""
    cfg = LLMConfig.from_env()
    control_ids = ", ".join(c["control_id"] for c in matched_controls) or "HKMA-TMG1-4.1"
    steps = []
    for c in matched_controls:
        steps.extend(c.get("remediation_steps", [])[:2])

    deterministic_brief = (
        f"Finding {finding['event_id']} ({finding['risk_tier']} Risk, Score {finding['risk_score']}/100) "
        f"in {finding['department']} on {finding['system_name']}. "
        f"Primary triggers: {'; '.join(finding.get('reason_codes', []))}. "
        f"Applicable controls: {control_ids}. "
        f"Recommended next action: {steps[0] if steps else 'Escalate to IT Security & Compliance review.'}"
    )

    if not cfg.enabled:
        return {"mode": "deterministic", "brief": deterministic_brief}

    payload = {
        "model": cfg.model,
        "temperature": 0.2,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": "You are a Hong Kong financial IT auditor. Summarize the finding and remediation in 3 concise sentences.",
            },
            {"role": "user", "content": deterministic_brief},
        ],
    }
    req = urllib.request.Request(
        f"{cfg.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"].strip()
            return {"mode": "llm_enhanced", "brief": text}
    except Exception:
        return {"mode": "deterministic_fallback", "brief": deterministic_brief}
