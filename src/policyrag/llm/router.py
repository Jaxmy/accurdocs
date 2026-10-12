"""多供应商路由/降级链（计划 §五.12）。

主供应商超时/限流 → 次选；全部失败 → LLMUnavailableError（明确报错不静默降质）。
降级事件写入 envelope.degradations。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Caps, ChatMessage, LLMClient, LLMResult

__all__ = ["LLMRouter", "LLMUnavailableError"]


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class _Route:
    client: LLMClient
    role: str   # primary | fallback | cheap


_RETRYABLE = ("timeout", "rate_limit", "overloaded", "server_error", "connection")


@dataclass
class LLMRouter:
    """按 role 路由：关键路径（起草/审查）走 primary，轻任务走 cheap。"""

    primary: LLMClient
    cheap: LLMClient | None = None
    fallbacks: list[LLMClient] = field(default_factory=list)

    @property
    def caps(self) -> Caps:
        return self.primary.caps

    def for_role(self, role: str) -> LLMClient:
        if role == "cheap" and self.cheap is not None:
            return self.cheap
        return self.primary

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        role: str = "primary",
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        effort: str = "high",
        degraded: list[str] | None = None,
        on_delta: Any = None,
    ) -> LLMResult:
        chain = [self.for_role(role)] + (
            self.fallbacks if role == "primary" else []
        )
        last_err: Exception | None = None
        for i, client in enumerate(chain):
            try:
                result = client.complete(
                    messages, schema=schema, max_tokens=max_tokens, effort=effort,
                    on_delta=on_delta,
                )
                if result.degraded == "refusal":
                    return result  # 拒答不是故障，不降级
                if i > 0 and degraded is not None:
                    degraded.append(f"llm_fallback_to:{client.name}")
                return result
            except Exception as e:  # noqa: BLE001 —— 供应商异常类型各异，统一转级
                last_err = e
                kind = type(e).__name__.lower()
                if not any(k in kind or k in str(e).lower() for k in _RETRYABLE):
                    raise
                continue
        raise LLMUnavailableError(f"全部 LLM 供应商不可用：{last_err}")
