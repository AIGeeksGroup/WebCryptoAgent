from __future__ import annotations

import logging
from typing import Any, Dict

import requests


def post_signal_to_node(payload: Dict[str, Any], url: str) -> Dict[str, Any]:
    """Send signal payload to the execution node."""
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logging.warning("Signal post failed: %s", exc)
        return {"error": str(exc)}
