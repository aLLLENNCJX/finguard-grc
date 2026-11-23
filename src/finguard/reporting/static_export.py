"""Static GitHub Pages HTML Exporter (embeds JSON snapshot into dashboard.html)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finguard.config import DASHBOARD_TEMPLATE_PATH, DOCS_DIR


def export_static_dashboard(
    snapshot: dict[str, Any],
    output_path: Path = DOCS_DIR / "index.html",
) -> Path:
    """Write standalone static docs/index.html for GitHub Pages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = DASHBOARD_TEMPLATE_PATH.read_text(encoding="utf-8")
    payload_js = (
        "window.__FINGUARD_API_MODE__ = false;\n"
        f"window.__FINGUARD_SNAPSHOT__ = {json.dumps(snapshot)};"
    )
    rendered = template.replace("/*__FINGUARD_BOOTSTRAP__*/", payload_js)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path
