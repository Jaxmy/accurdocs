"""检索侧数据模型：RetrievalDirective / Hit / Evidence / QueryRequest。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["RetrievalDirective", "Hit", "Evidence", "QueryRequest", "DocFilter"]


@dataclass
class DocFilter:
    doc_ids: list[str] | None = None
    doc_keys: list[str] | None = None   # 精确范围：doc_id@version 列表（GUI 勾选）
    chunk_ids: list[str] | None = None  # 精确范围：块级勾选（限定在某块内检索）
    library_ids: list[str] | None = None  # 库级范围（WS-E 分库）
    doc_family: str | None = None       # 模糊文档族（标题/文号前缀）
    issuer: str | None = None
    include_repealed: bool = False      # 历史条款需显式 opt-in

    @property
    def has_scope(self) -> bool:
        return bool(self.chunk_ids or self.doc_keys or self.doc_ids or self.library_ids)


@dataclass
class RetrievalDirective:
    """查询重写的结构化检索指令（技术 8）。空字段 = 不约束。

    过度过滤会导致召回坍塌——directive 过滤后结果 <k 时 pipeline 自动放宽。
    """

    intent: Literal["quote", "locate", "compare", "analyze"] = "quote"
    doc_family: str | None = None
    as_of: str | None = None            # ISO 日期
    clause_refs: list[str] = field(default_factory=list)   # ["第十二条", "第三款"]
    terms: list[str] = field(default_factory=list)          # 规范化专业术语
    numeric_anchors: list[str] = field(default_factory=list)  # 原文形态的数值锚点
    date_range: tuple[str, str] | None = None
    original_query: str = ""            # 用户原话，兜底召回防重写漂移（计划 §五.8）
    question_kind: str | None = None    # enum|numeric|compare|definition|procedure|analysis
    dimension: str | None = None        # power|capacity|money|ratio|time|count|none

    def constraint_count(self) -> int:
        return sum(
            1
            for x in (self.doc_family, self.clause_refs, self.terms, self.numeric_anchors)
            if x
        )


@dataclass
class Hit:
    chunk_id: str
    score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float | None = None
    snippet: str = ""
    doc_title: str = ""
    doc_version: str = ""
    clause_path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """进入生成上下文的证据块。text 是 evidence_text()（表格行带表头）。"""

    chunk_id: str
    text: str
    doc_title: str
    doc_version: str
    clause_path: str
    page: int | None = None
    from_expansion: bool = False   # 渐进披露 get_chunk 展开得到


@dataclass
class QueryRequest:
    query: str
    as_of: str = ""                # 默认今天，一等参数
    filters: DocFilter = field(default_factory=DocFilter)
    mode: Literal["quote", "analyze"] = "quote"
    top_k: int = 8
    request_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    # WS-B execute 无状态红线（G1）：注入已确认指令跳过 B2 重写；扩展召回串并入检索
    pre_directive: dict[str, Any] | None = None
    variant_queries: list[str] = field(default_factory=list)
