"""OpenAI 兼容协议共用完成/流式实现（DeepSeek / GLM 共用）。

on_delta 提供时走 stream=True 逐增量前向；网关不支持流式则回落整段推送。
"""

from __future__ import annotations

from typing import Any

from ..models.answer import Usage
from .base import LLMResult

__all__ = ["openai_complete"]


def openai_complete(client: Any, kwargs: dict[str, Any], model: str,
                    on_delta: Any = None) -> LLMResult:
    if on_delta is None:
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            output_tokens=getattr(u, "completion_tokens", 0) if u else 0,
        )
        return LLMResult(text=text, usage=usage, model=model,
                         stop_reason=resp.choices[0].finish_reason or "")

    stream_kwargs = dict(kwargs, stream=True)
    try:
        stream_kwargs["stream_options"] = {"include_usage": True}
        stream = client.chat.completions.create(**stream_kwargs)
    except Exception:  # noqa: BLE001 - 网关不支持 stream_options
        stream_kwargs.pop("stream_options", None)
        try:
            stream = client.chat.completions.create(**stream_kwargs)
        except Exception:  # noqa: BLE001 - 完全不支持流式
            return openai_complete(client, kwargs, model, on_delta=None)

    parts: list[str] = []
    usage = Usage()
    stop = ""
    try:
        for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage = Usage(
                    input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(u, "completion_tokens", 0) or 0,
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                parts.append(delta)
                on_delta(delta)
            fr = chunk.choices[0].finish_reason
            if fr:
                stop = fr
    except Exception:  # noqa: BLE001 - 流中断：已收内容照常返回
        stop = stop or "interrupted"
    text = "".join(parts)
    return LLMResult(text=text, usage=usage, model=model, stop_reason=stop)
