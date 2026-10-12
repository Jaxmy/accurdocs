"""原文阅读器展示层：同一条款拼合 + 块下标定位（纯函数，零 I/O）。

检索切分保持不变（chunk_id 稳定红线）；本模块只把已入库的 chunk 行
分组为"展示块"供阅读页渲染——拼合只分组现有行，绝不重新生成/替换 id。
"""
from __future__ import annotations

import json
import re
from typing import Any

# "clause" = 整条拼合（含款/项/续）；"cont" = 仅拼 800 字续块
MERGE_MODE = "clause"

_ARTICLE_SEG_RE = re.compile(
    r"^第[0-9０-９一二三四五六七八九十百千零〇两]+条(之[0-9０-９一二三四五六七八九十百千零〇两]+)?$")
_CONT_RE = re.compile(r"/续\d+$")
_TABLE_PATH_RE = re.compile(r"/表(/行\d+)?$")
# 公式/图表独立成块，永不参与条款拼合
_SOLO_PATH_PREFIXES = ("公式/", "图表/")
_MERGE_EXCLUDE_KINDS = {"table", "formula", "chart"}


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """页序 + path_key 序（与 fulltext 历史口径一致）。"""
    return sorted(rows, key=lambda r: (r.get("page") or 9999, r.get("path_key") or ""))


def _meta_of(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(row.get("meta_json") or "{}")
    except Exception:  # noqa: BLE001
        return {}


def merge_key(row: dict[str, Any]) -> str | None:
    """条款级拼合键；None = 独立成块（不拼合）。"""
    if row.get("kind") in _MERGE_EXCLUDE_KINDS:
        return None
    pk = row.get("path_key") or ""
    if not pk or pk.startswith(_SOLO_PATH_PREFIXES) or _TABLE_PATH_RE.search(pk):
        return None
    if MERGE_MODE == "cont":
        return _CONT_RE.sub("", pk)
    # 整条拼合：截到最后一个"第X条"段，吸收款/项/续段
    parts = pk.split("/")
    cut = None
    for i, p in enumerate(parts):
        if _ARTICLE_SEG_RE.match(p):
            cut = i
    if cut is None:
        return _CONT_RE.sub("", pk)   # 章/节/无条段 → 退化为仅续拼合
    return "/".join(parts[:cut + 1])


def display_blocks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """chunk 行 → 拼合后的展示块（有序）。

    按 key 全表分组（非仅连续段——款 U+6B3E < 续 U+7EED 会插在基块与续块
    之间），组输出位置 = 首成员位置。多成员块带 chunk_ids/pieces。
    """
    rows = sort_rows(rows)
    slots: list[list[dict[str, Any]]] = []
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        k = merge_key(r)
        if k is None:
            slots.append([r])
            continue
        if k not in by_key:
            by_key[k] = []
            slots.append(by_key[k])
        by_key[k].append(r)
    out: list[dict[str, Any]] = []
    for members in slots:
        first = members[0]
        out.append({
            "chunk_id": first["chunk_id"],
            "chunk_ids": [m["chunk_id"] for m in members],
            "pieces": [{"chunk_id": m["chunk_id"], "text": m["text"]} for m in members],
            "kind": first["kind"],
            "breadcrumb": first["breadcrumb"],
            "path_key": first["path_key"],
            "page": first["page"],
            "text": "\n".join(m["text"] for m in members),
            "meta": _meta_of(first),
        })
    return out


def block_index_of(rows: list[dict[str, Any]], chunk_id: str) -> int | None:
    """chunk（主块或成员）在展示块列表中的下标；找不到返回 None。"""
    for i, b in enumerate(display_blocks(rows)):
        if chunk_id in b["chunk_ids"]:
            return i
    return None


def chunk_to_primary(rows: list[dict[str, Any]]) -> dict[str, str]:
    """成员 id → 所属展示块主 id（结构树叶映射用）。"""
    out: dict[str, str] = {}
    for b in display_blocks(rows):
        for cid in b["chunk_ids"]:
            out[cid] = b["chunk_id"]
    return out
