#!/usr/bin/env python3
"""Tiny Gemini billing/connectivity preflight for an already-exported API key."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egolife_two_user_qa.qwen3vl_runner import make_runner


def main() -> None:
    runner = make_runner("gemini-api", model_id="gemini-2.5-pro", max_new_tokens=32)
    text = runner.generate('Return exactly {"ok": true} as JSON.')
    print(text)


if __name__ == "__main__":
    main()
