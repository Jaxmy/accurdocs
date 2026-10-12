"""LLMClient 抽象层：多供应商统一接口 + capability flags（计划 §二）。

红线（计划 §五）：
- 不索要原始 CoT（reasoning_extraction 安全类目）；
- structured outputs 优先，无则 JSON-mode + 解析重试；
- prompt cache 前缀字节稳定；缓存能力缺失的供应商静默跳过断点；
- 供应商降级链：超时/限流 → 次选；全部失败 → 明确报错不静默降质。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..models.answer import Usage

__all__ = ["Caps", "LLMResult", "LLMClient", "ChatMessage", "llm_call_json",
           "partial_answer_text"]


@dataclass(frozen=True)
class Caps:
    structured_output: bool = False   # 原生 JSON schema 约束输出
    prompt_caching: bool = False      # 前缀缓存断点
    citations: bool = False           # 原生引用（v2 长报告模式用）
    thinking: bool = False            # adaptive thinking
    max_context: int = 128_000
    json_mode: bool = False           # response_format=json_object


@dataclass
class ChatMessage:
    role: str          # system | user | assistant
    content: str
    cache: bool = False   # 是否在此块后打 cache 断点（仅支持缓存的供应商生效）
    images: list[bytes] = field(default_factory=list)   # 视觉输入（PNG/JPEG bytes）


@dataclass
class LLMResult:
    text: str
    usage: Usage
    parsed: dict[str, Any] | list[Any] | None = None
    raw: Any = None
    model: str = ""
    stop_reason: str = ""
    degraded: str | None = None   # 降级说明（如 JSON 解析重试）


class LLMClient(Protocol):
    name: str
    caps: Caps

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        effort: str = "high",
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResult: ...


# ---------------------------------------------------------------------------
# JSON 输出解析（供应商无 structured_output 时的兜底）
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _extract_json(text: str) -> Any | None:
    """从模型输出中提取 JSON：优先整体解析，再找 ```json 块/首个花括号块。"""
    t = text.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)```", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = _JSON_BLOCK_RE.search(t)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def llm_call_json(
    client: LLMClient,
    messages: list[ChatMessage],
    *,
    schema: dict[str, Any] | None = None,
    max_tokens: int = 4096,
    effort: str = "high",
    max_retries: int = 1,
    on_delta: Callable[[str], None] | None = None,
) -> LLMResult:
    """结构化 JSON 调用：原生 structured_output → 直接用；否则 JSON-mode + 重试。

    on_delta 仅在首次尝试时前向（重试不重复推送增量）。
    """
    result = client.complete(
        messages, schema=schema, max_tokens=max_tokens, effort=effort,
        on_delta=on_delta,
    )
    if result.parsed is not None:
        return result

    parsed = _extract_json(result.text)
    if parsed is not None:
        result.parsed = parsed
        return result

    # 一次修复重试（追加纠错提示——append-only，不改历史）
    for _ in range(max_retries):
        retry_messages = messages + [
            ChatMessage(role="assistant", content=result.text),
            ChatMessage(
                role="user",
                content="上面的输出不是合法 JSON。请只输出一个符合给定 schema 的 JSON 对象，"
                        "不要任何解释文字、不要 markdown 代码块。",
            ),
        ]
        result = client.complete(
            retry_messages, schema=schema, max_tokens=max_tokens, effort=effort
        )
        parsed = _extract_json(result.text)
        if parsed is not None:
            result.parsed = parsed
            result.degraded = "json_retry"
            return result

    result.degraded = "json_parse_failed"
    return result


# ---------------------------------------------------------------------------
# 流式展示：从累积 JSON 缓冲中提取 answer 字段的当前可显示文本
# ---------------------------------------------------------------------------

_PARTIAL_ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)')


def partial_answer_text(buffer: str) -> str:
    """流式 JSON 缓冲 → answer 字段已生成部分（供前端渐进显示）。"""
    ms = list(_PARTIAL_ANSWER_RE.finditer(buffer))
    if not ms:
        return ""
    s = ms[-1].group(1)
    return (
        s.replace("\\n", "\n").replace("\\t", " ")
        .replace('\\"', '"').replace("\\\\", "\\")
    )
