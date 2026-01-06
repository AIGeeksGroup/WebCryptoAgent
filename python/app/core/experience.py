from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import ensure_dir


class ExperienceStore:
    """Lightweight JSONL-backed experience tracker."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        ensure_dir(self._path.parent)

    def _load_all(self) -> List[Dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def _write_all(self, items: List[Dict[str, Any]]) -> None:
        ensure_dir(self._path.parent)
        with self._path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    def fetch_recent(
        self,
        coin: str,
        agent: str,
        limit: int = 3,
        require_reward: bool = False,
    ) -> List[Dict[str, Any]]:
        items = self._load_all()
        filtered: List[Dict[str, Any]] = []
        for entry in reversed(items):
            if entry.get("coin") != coin or entry.get("agent") != agent:
                continue
            if require_reward and not entry.get("reward"):
                continue
            filtered.append(entry)
            if len(filtered) >= limit:
                break
        return filtered

    def record_open_experience(self, record: Dict[str, Any]) -> None:
        items = self._load_all()
        items.append(record)
        self._write_all(items)

    def update_experience(self, experience_id: str, updates: Dict[str, Any]) -> None:
        items = self._load_all()
        updated = False
        for entry in items:
            if entry.get("id") == experience_id:
                entry.update(updates)
                updated = True
                break
        if not updated:
            logging.warning("Experience %s not found for update", experience_id)
            return
        self._write_all(items)
