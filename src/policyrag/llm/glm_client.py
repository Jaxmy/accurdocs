"""智谱 GLM 客户端。capability：json_mode 可用性随模型版本，走兜底 JSON 提取。"""

from __future__ import annotations

import os
from typing import Any

from ..models.answer import Usage
from .base import Caps, ChatMessage, LLMResult
from .openai_stream import openai_complete as _openai_complete

DEFAULT_MODEL = "glm-4-plus"

_CAPS = Caps(structured_output=False, json_mode=False, max_context=128_000)


class GLMClient:
    name = "glm"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        try:
            from zhipuai import ZhipuAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 zhipuai SDK") from e
        self.model = model
        self.caps = _CAPS
        self._client = ZhipuAI(api_key=api_key or os.environ.get("ZHIPUAI_API_KEY"))

    def complete(self, messages: list[ChatMessage], *, schema: dict[str, Any] | None = None,
                 max_tokens: int = 4096, temperature: float = 0.0,
                 effort: str = "high", on_delta: Any = None) -> LLMResult:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model, "messages": msgs,
            "max_tokens": max_tokens, "temperature": temperature,
        }
        return _openai_complete(self._client, kwargs, self.model, on_delta)
