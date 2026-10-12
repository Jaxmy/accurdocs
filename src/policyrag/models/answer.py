"""全链路输出契约：AnswerEnvelope / Citation / Refusal / VerifyReport。

这是 REST 与 MCP 双 adapter 的统一 schema（计划 §四 / §六），
外部调用方只依赖这个文件里的形状。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "AnswerStatus",
    "Citation",
    "Refusal",
    "VerifyReport",
    "Usage",
    "AnswerEnvelope",
]


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"
    PARTIAL = "partial"


@dataclass
class Citation:
    chunk_id: str
    quote: str                 # 从 chunk 逐字摘录（verify 阶段比对）
    doc_title: str
    doc_version: str
    clause_path: str
    page: int | None = None
    verified: bool = False     # verify_citations 通过后置 True
    demoted: bool = False      # F4c 降级转述：quote 已剥除，不再声称逐字


@dataclass
class Refusal:
    reason: str                # 机器可读：insufficient_evidence / verification_failed / ...
    detail: str                # 给人看的缺口说明
    missing_hints: list[str] = field(default_factory=list)


@dataclass
class VerifyReport:
    quotes_ok: bool = True
    numbers_ok: bool = True
    review_ok: bool | None = None      # None = 未触发审查
    failures: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return self.quotes_ok and self.numbers_ok and (self.review_ok is not False)


@dataclass
class Usage:
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd


@dataclass
class AnswerEnvelope:
    """REST/MCP 统一响应体。"""

    request_id: str
    status: AnswerStatus
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    as_of: str = ""
    corpus_version: str = ""
    refusal: Refusal | None = None
    verification: VerifyReport = field(default_factory=VerifyReport)
    degradations: list[str] = field(default_factory=list)
    trace_id: str = ""
    usage: Usage = field(default_factory=Usage)
    meta: dict[str, Any] = field(default_factory=dict)
    formulas: list[dict] = field(default_factory=list)   # 引用到的公式块（latex+符号+图）
    charts: list[dict] = field(default_factory=list)     # 引用到的图表块
    rich_jobs: list[dict] = field(default_factory=list)  # 按需深转任务（进度条用）
    synthesis: dict[str, Any] | None = None              # 两阶段综合推理（WS-C，顶层一次）

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "answer": self.answer,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "quote": c.quote,
                    "doc_title": c.doc_title,
                    "doc_version": c.doc_version,
                    "clause_path": c.clause_path,
                    "page": c.page,
                    "verified": c.verified,
                }
                for c in self.citations
            ],
            "confidence": self.confidence,
            "as_of": self.as_of,
            "corpus_version": self.corpus_version,
            "refusal": (
                {
                    "reason": self.refusal.reason,
                    "detail": self.refusal.detail,
                    "missing_hints": self.refusal.missing_hints,
                }
                if self.refusal
                else None
            ),
            "verification": {
                "quotes_ok": self.verification.quotes_ok,
                "numbers_ok": self.verification.numbers_ok,
                "review_ok": self.verification.review_ok,
                "failures": self.verification.failures,
            },
            "degradations": self.degradations,
            "formulas": self.formulas,
            "charts": self.charts,
            "rich_jobs": self.rich_jobs,
            "synthesis": self.synthesis,
            "trace_id": self.trace_id,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cost_usd": self.usage.cost_usd,
            },
            "meta": self.meta,
        }
