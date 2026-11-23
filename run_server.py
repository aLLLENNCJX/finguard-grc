"""Convenience launcher for the FinGuard-GRC web dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finguard.api.app import create_app  # noqa: E402
from finguard.cli import run_full_pipeline  # noqa: E402
from finguard.config import GENERATED_DIR  # noqa: E402


if __name__ == "__main__":
    if not (GENERATED_DIR / "dashboard_snapshot.json").exists():
        run_full_pipeline(verify_only=False)
    app = create_app()
    app.run(host="127.0.0.1", port=8050, debug=False)