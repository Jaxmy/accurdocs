"""中文条款原子分块 + 稳定 ID 绑定。

红线（计划 §五.1）：
- 条款即原子块；超长条款内切时子块带 parent 引用；
- 切分只落在句号/分号/换行处，保护 span 内不断开；
- "但书/除外"句与本款同块（不切）；
- 目标 300-800 汉字/chunk（表格除外）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models.document import BlockKind, Chunk, DocAST, DocNode, TableBlock
from ..models.ids import ClausePath, make_chunk_id, normalize_text_for_id
from .normalize import normalize
from .structure import PROTECT_RE

__all__ = ["ChunkConfig", "chunk_document"]


@dataclass
class ChunkConfig:
    min_chars: int = 80
    target_chars: int = 500
    max_chars: int = 800
    split_tables_to_rows: bool = True


_SENT_END = re.compile(r"[。；;！!？?]")
_BUT_CLAUSE = re.compile(r"(但\s*书|除[此该本][^。]{0,30}外|但是)")


def chunk_document(ast: DocAST, cfg: ChunkConfig | None = None) -> list[Chunk]:
    """DocAST → 原子 Chunk 列表（含表格 chunk）。确定性纯函数。"""
    cfg = cfg or ChunkConfig()
    chunks: list[Chunk] = []

    for node in ast.nodes:
        _chunk_node(node, ast, cfg, chunks, parent_id=None)

    return chunks


def _chunk_node(
    node: DocNode,
    ast: DocAST,
    cfg: ChunkConfig,
    out: list[Chunk],
    parent_id: str | None,
) -> None:
    if node.kind == BlockKind.TABLE:
        tb: TableBlock | None = node.meta.get("table")
        if tb is not None:
            out.extend(_table_chunks(tb, ast, node.path, cfg))
        return

    # 叶子文本节点（条/款/项/TEXT）——先收集本节点直挂文本
    text = node.text.strip()
    if text:
        pieces = _split_long_text(text, cfg) if len(text) > cfg.max_chars else [text]
        for idx, piece in enumerate(pieces):
            chunk = _make_chunk(ast, node, piece, parent_id=parent_id if idx > 0 else parent_id)
            if idx > 0 and len(pieces) > 1:
                chunk.parent_chunk_id = parent_id or out[-1].chunk_id if out else None
                chunk.path = ClausePath.of(*node.path.parts, f"续{idx}") if idx > 0 else node.path
                # 重新绑定 ID（path 变了必须重算）
                chunk.chunk_id = make_chunk_id(
                    ast.doc_id, ast.version, chunk.path,
                    normalize_text_for_id(piece),
                )
            out.append(chunk)

    for child in node.children:
        _chunk_node(child, ast, cfg, out, parent_id=out[-1].chunk_id if out else None)


def _make_chunk(ast: DocAST, node: DocNode, text: str, parent_id: str | None) -> Chunk:
    nt = normalize(text)
    text_norm = normalize_text_for_id(text)
    cid = make_chunk_id(ast.doc_id, ast.version, node.path, text_norm)
    breadcrumb = node.path.display
    return Chunk(
        chunk_id=cid,
        doc_id=ast.doc_id,
        version=ast.version,
        path=node.path,
        text=text,
        text_norm=nt.norm,
        kind=node.kind,
        breadcrumb=breadcrumb,
        page=node.page,
        parent_chunk_id=parent_id,
    )


def _split_long_text(text: str, cfg: ChunkConfig) -> list[str]:
    """超长条款内切：保护 span 内不断开；句末切；但书句并入前块。"""
    # 标记保护区间
    protected = [False] * len(text)
    for m in PROTECT_RE.finditer(text):
        for i in range(m.start(), m.end()):
            protected[i] = True

    cut_points: list[int] = []
    for i, ch in enumerate(text):
        if ch in "。；;！!？?" and not protected[i]:
            # 但书检测：句末向前看 25 字，若发现但书标记则不在此切
            window = text[max(0, i - 25) : i + 1]
            if _BUT_CLAUSE.search(window):
                continue
            cut_points.append(i + 1)

    pieces: list[str] = []
    start = 0
    for cp in cut_points:
        if cp - start >= cfg.target_chars:
            piece = text[start:cp].strip()
            if piece:
                pieces.append(piece)
            start = cp
    tail = text[start:].strip()
    if tail:
        if pieces and len(tail) < cfg.min_chars:
            pieces[-1] = (pieces[-1] + " " + tail).strip()
        else:
            pieces.append(tail)
    return pieces or [text]


def _table_chunks(tb: TableBlock, ast: DocAST, path: ClausePath, cfg: ChunkConfig) -> list[Chunk]:
    """表格：整表 chunk（caption+表头）+ 行级 chunk（行级 ID 供数值引用）。"""
    out: list[Chunk] = []

    # 整表 chunk
    full_text = tb.caption + "\n" + " | ".join(tb.headers) + "\n" + "\n".join(
        " | ".join(r) for r in tb.rows
    )
    nt = normalize(full_text)
    table_path = ClausePath.of(*path.parts, "表")
    cid = make_chunk_id(ast.doc_id, ast.version, table_path, normalize_text_for_id(full_text))
    out.append(Chunk(
        chunk_id=cid, doc_id=ast.doc_id, version=ast.version, path=table_path,
        text=full_text, text_norm=nt.norm, kind=BlockKind.TABLE,
        breadcrumb=path.display, page=tb.page, table=tb,
    ))

    # 行级 chunk
    if cfg.split_tables_to_rows:
        for ri, row in enumerate(tb.rows):
            row_text = f"{tb.caption} | " + " | ".join(
                f"{h}={row[i] if i < len(row) else ''}" for i, h in enumerate(tb.headers)
            )
            row_path = ClausePath.of(*path.parts, "表", f"行{ri + 1}")
            nt = normalize(row_text)
            rcid = make_chunk_id(
                ast.doc_id, ast.version, row_path, normalize_text_for_id(row_text)
            )
            out.append(Chunk(
                chunk_id=rcid, doc_id=ast.doc_id, version=ast.version, path=row_path,
                text=row_text, text_norm=nt.norm, kind=BlockKind.TABLE,
                breadcrumb=f"{path.display} 表 行{ri + 1}", page=tb.page,
                table=tb, row_index=ri,
            ))
    return out
