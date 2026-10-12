"""DeepSeek 客户端（OpenAI 兼容协议）。

capability：json_mode=True（response_format），无 prompt_caching / citations。
"""

from __future__ import annotations

import os
from typing import Any

from ..models.answer import Usage
from .base import Caps, ChatMessage, LLMResult
from .openai_stream import openai_complete as _openai_complete

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

DEFAULT_MODEL = "deepseek-chat"

_CAPS = Caps(structured_output=False, json_mode=True, max_context=64_000)


class DeepSeekClient:
    name = "deepseek"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 base_url: str = "https://api.deepseek.com"):
        if OpenAI is None:
            raise RuntimeError("未安装 openai SDK")
        self.model = model
        self.caps = _CAPS
        self._client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )

    def complete(self, messages: list[ChatMessage], *, schema: dict[str, Any] | None = None,
                 max_tokens: int = 4096, temperature: float = 0.0,
                 effort: str = "high", on_delta: Any = None) -> LLMResult:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if schema is not None and self.caps.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return _openai_complete(self._client, kwargs, self.model, on_delta)
