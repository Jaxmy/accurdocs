"""Anthropic Claude 客户端（capability: structured_output, prompt_caching, thinking）。

API 细节（计划 §五.4-7，以 claude-api skill 为准）：
- thinking 用 adaptive；不发 budget_tokens（400）；
- output_config.format 做 structured outputs；与 citations 互斥；
- cache_control 断点；前缀字节稳定；
- 不索要原始 CoT。
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..models.answer import Usage
from .base import Caps, ChatMessage, LLMResult

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


DEFAULT_MODEL = "claude-opus-5-5"

_CAPS = Caps(
    structured_output=True,
    prompt_caching=True,
    citations=False,   # 与 structured_output 互斥，主路径不用
    thinking=True,
    max_context=200_000,
    json_mode=True,
)

# 每 1M token 单价（USD），成本估算用
_PRICE = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.1, 0.5),
}


class AnthropicClient:
    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        effort: str = "high",
    ):
        if anthropic is None:
            raise RuntimeError("未安装 anthropic SDK：uv pip install anthropic")
        # 模型名优先级：显式参数 > POLICYRAG_MODEL 环境变量 > 默认值
        self.model = model or os.environ.get("POLICYRAG_MODEL") or DEFAULT_MODEL
        self.caps = _CAPS
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._default_effort = effort

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        effort: str = "high",
        on_delta: Any = None,
    ) -> LLMResult:
        system_blocks: list[dict[str, Any]] = []
        msg_list: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                block: dict[str, Any] = {"type": "text", "text": m.content}
                if m.cache and self.caps.prompt_caching:
                    block["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(block)
            elif m.images:
                # 视觉输入：image blocks + 文本块（messages API content 数组）
                import base64 as _b64
                content: list[dict[str, Any]] = [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/png" if img[:4] == b"\x89PNG" else "image/jpeg",
                                "data": _b64.b64encode(img).decode("ascii")}}
                    for img in m.images
                ]
                content.append({"type": "text", "text": m.content})
                msg_list.append({"role": m.role, "content": content})
            else:
                msg_list.append({"role": m.role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": msg_list,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if system_blocks:
            kwargs["system"] = system_blocks

        # structured outputs：output_config.format（与 citations 互斥）
        if schema is not None and self.caps.structured_output:
            kwargs["output_config"] = {
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            }

        # 流式：on_delta 前向文本增量；网关不支持流式则回落整段推送
        resp = None
        if on_delta is not None:
            sent = False
            try:
                with self._client.messages.stream(**kwargs) as stream:
                    for piece in stream.text_stream:
                        if piece:
                            sent = True
                            on_delta(piece)
                    resp = stream.get_final_message()
            except Exception:  # noqa: BLE001 - 网关不支持/中断 → 非流式兜底
                resp = None
                if not sent:
                    resp = self._client.messages.create(**kwargs)
                    text = "".join(
                        b.text for b in resp.content if getattr(b, "type", "") == "text")
                    if text:
                        on_delta(text)
        if resp is None:
            resp = self._client.messages.create(**kwargs)

        # stop_reason == refusal 时不读 content（计划 §五）
        if resp.stop_reason == "refusal":
            return LLMResult(
                text="", usage=_usage_of(resp), model=self.model,
                stop_reason="refusal", degraded="refusal",
            )

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        parsed: Any = None
        # structured outputs 优先走 parsed 字段（messages.parse 形态）；
        # create 形态下 JSON 在 text 中，交由 llm_call_json 提取
        if schema is not None:
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed = None

        return LLMResult(
            text=text, usage=_usage_of(resp), parsed=parsed, raw=resp,
            model=self.model, stop_reason=resp.stop_reason or "",
        )


def _usage_of(resp: Any) -> Usage:
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cc = getattr(u, "cache_creation_input_tokens", 0) or 0
    # 成本估算：cache read 按 0.05× 输入、cache write 按 1.25× 输入
    model = getattr(resp, "model", DEFAULT_MODEL)
    p_in, p_out = _PRICE.get(model, (4.0, 20.0))
    cost = (inp * p_in + cr * p_in * 0.05 + cc * p_in * 1.25 + out * p_out) / 1_000_000
    return Usage(
        input_tokens=inp, cache_read_input_tokens=cr,
        cache_creation_input_tokens=cc, output_tokens=out, cost_usd=cost,
    )
