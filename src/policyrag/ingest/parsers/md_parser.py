"""Markdown 解析：按标题层级构造页面流。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...models.document import TableBlock

__all__ = ["MdParseResult", "parse_md"]


@dataclass
class MdParseResult:
    pages: list[tuple[str, int | None]] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ocr_required: bool = False


def parse_md(path: str) -> MdParseResult:
    # errors="replace"：抽取/转换产物常带非法 UTF-8 字节，替换而非拒收
    text = open(path, encoding="utf-8", errors="replace").read()
    result = MdParseResult()

    lines: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []
    table_caption = ""

    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not in_table:
                in_table = True
                table_rows = []
                table_caption = f"（第 {len(lines) + 1} 行附近表）"
            # 跳过分隔行 |---|---|
            if not all(set(c) <= set("-: ") for c in cells):
                table_rows.append(cells)
            continue
        if in_table:
            _flush_table(result, table_caption, table_rows)
            in_table = False
        if s:
            lines.append(s)

    if in_table:
        _flush_table(result, table_caption, table_rows)

    result.pages.append(("\n".join(lines), 1))
    return result


def _flush_table(result: MdParseResult, caption: str, rows: list[list[str]]) -> None:
    if len(rows) >= 2:
        result.tables.append(TableBlock(
            caption=caption, headers=rows[0], rows=rows[1:],
        ))
