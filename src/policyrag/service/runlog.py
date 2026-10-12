"""Agent 运行日志：单文件 JSONL 追加，回溯每次问答/入库/任务/错误。"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()


def _log_path() -> Path:
    p = Path(os.environ.get("POLICYRAG_RUNLOG", "data/agent_run.log"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log_event(kind: str, **fields) -> None:
    """追加一行 JSON。任何异常都不影响主链路。"""
    try:
        line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "kind": kind, **fields}, ensure_ascii=False, default=str)
        with _LOCK, open(_log_path(), "a", encoding="utf-8") as f:
            f.write(line + chr(10))
    except Exception:  # noqa: BLE001
        pass
