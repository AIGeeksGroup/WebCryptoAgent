from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_path(path: str | Path) -> None:
    ensure_dir(Path(path).parent)


def write_jsonl_line(path: str | Path, record: Dict) -> None:
    ensure_path(path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
