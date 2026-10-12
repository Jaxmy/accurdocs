"""TraceRecorder：每步输入输出落盘，可回放（计划 §四 通用机制）。

traces/{trace_id}/step_{n:02d}_{name}.json —— LLM 步额外存完整 prompt 与原始响应。
replay.py --trace-id X --from-step N 可从任意步重建。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

__all__ = ["TraceRecorder"]


class TraceRecorder:
    def __init__(self, trace_id: str, root: str | Path = "traces"):
        self.trace_id = trace_id
        self.dir = Path(root) / trace_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._manifest: list[dict[str, Any]] = []

    def record(
        self,
        name: str,
        input_data: Any,
        output_data: Any,
        *,
        usage: dict[str, Any] | None = None,
        model: str = "",
        prompt: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        self._step += 1
        payload = {
            "step_no": self._step,
            "name": name,
            "ts": time.time(),
            "input": _safe(input_data),
            "output": _safe(output_data),
            "usage": usage or {},
            "model": model,
            "prompt_hash": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12] if prompt else None,
            **(extra or {}),
        }
        path = self.dir / f"step_{self._step:02d}_{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
        if prompt is not None:
            (self.dir / f"step_{self._step:02d}_{name}.prompt.txt").write_text(
                prompt, encoding="utf-8"
            )
        self._manifest.append({"step_no": self._step, "name": name, "file": path.name})
        (self.dir / "manifest.json").write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return path

    def step_no(self) -> int:
        return self._step


def _safe(obj: Any) -> Any:
    """JSON 可序列化化；dataclass 转 dict。"""
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        try:
            return asdict(obj)
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    return str(obj)
