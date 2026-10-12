"""文档 AST 与 Chunk 数据模型。

DocAST：解析产物，章/节/条/款/项/表 的树。
Chunk：检索与引用的原子单位（计划 §四 Pipeline A）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .ids import ClausePath

__all__ = ["BlockKind", "DocNode", "DocAST", "Chunk", "TableBlock", "PageBlock"]


class BlockKind(str, Enum):
    CHAPTER = "chapter"      # 章
    SECTION = "section"      # 节
    ARTICLE = "article"      # 条
    PARAGRAPH = "paragraph"  # 款
    ITEM = "item"            # 项 （一）（二）...
    HEADING = "heading"      # 标题/篇名/栏目名（总标题、分则标题、研报栏目）
    TABLE = "table"
    FORMULA = "formula"
    CHART = "chart"
    TEXT = "text"            # 无结构段落（降级路径）


@dataclass
class DocNode:
    """AST 节点。kind + 编号 + 正文 + 子节点。"""

    kind: BlockKind
    number: str          # "第三条"、"（二）"、"" ；display 用
    text: str            # 本节点正文（不含子节点）
    children: list["DocNode"] = field(default_factory=list)
    path: ClausePath = field(default_factory=lambda: ClausePath.of())
    page: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class DocAST:
    doc_id: str
    version: str
    title: str
    nodes: list[DocNode]
    effective_date: str | None = None   # ISO；None = temporal_unknown
    repealed_date: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_key(self) -> str:
        return f"{self.doc_id}@{self.version}"


@dataclass
class TableBlock:
    """结构化表格。caption + 表头 + 行。行级引用走 chunk 的 row_index。"""

    caption: str
    headers: list[str]
    rows: list[list[str]]
    page: int | None = None

    def row_records(self) -> list[dict[str, str]]:
        out = []
        for row in self.rows:
            rec = {}
            for i, h in enumerate(self.headers):
                rec[h] = row[i] if i < len(row) else ""
            out.append(rec)
        return out


@dataclass
class Chunk:
    """检索/引用原子块。每个 chunk 必有稳定 id 与原文归属。"""

    chunk_id: str
    doc_id: str
    version: str
    path: ClausePath
    text: str               # 原文（未归一化，生成引用 quote 的来源）
    text_norm: str          # 归一化文本（ID 与校验比对用）
    kind: BlockKind
    breadcrumb: str         # 上下文标题链，如 "第三章 第十二条"
    page: int | None = None
    parent_chunk_id: str | None = None   # 超长条款内切时的父块
    summary: str = ""        # L1 导航摘要——永不进证据池（计划 §一.2）
    table: TableBlock | None = None
    row_index: int | None = None          # 表格行级 chunk
    meta: dict[str, Any] = field(default_factory=dict)

    def evidence_text(self) -> str:
        """证据池用文本：表格行带表头上下文。"""
        if self.table is not None and self.row_index is not None:
            hdr = " | ".join(self.table.headers)
            row = " | ".join(self.table.rows[self.row_index])
            return f"{self.table.caption}\n{hdr}\n{row}"
        return self.text
