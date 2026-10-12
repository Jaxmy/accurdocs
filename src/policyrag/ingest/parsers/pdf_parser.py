"""PDF 解析：文本流 + 表格抽取（PyMuPDF）。

红线（计划 §五.14）：页眉页脚坐标聚类剔除；双栏按坐标重排；
无文本层 → ocr_required 拒绝入库（高风险域不自动 OCR）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...models.document import TableBlock

__all__ = ["PdfParseResult", "parse_pdf"]


@dataclass
class PdfParseResult:
    pages: list[tuple[str, int | None]] = field(default_factory=list)  # (page_text, page_no)
    tables: list[TableBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ocr_required: bool = False


# 页脚页码形态：— 3 — / 第2页 / 纯数字页码行
_FOOTER_PAT = re.compile(
    r"^[—\-–—\s]*\d{1,4}[—\-–—\s]*$|^第\s*\d{1,3}\s*页\s*$")


def parse_pdf(path: str) -> PdfParseResult:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    result = PdfParseResult()
    total_text = 0

    # 两遍：先取各行统计跨页重复（页眉/页脚），再统一剔除
    page_lines: list[tuple[int, list[str]]] = []
    freq: dict[str, int] = {}
    for page_no in range(len(doc)):
        text = doc[page_no].get_text("text")
        total_text += len(text.strip())
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        # 页内重复行去重（文本层重叠抽取常致整段重复——研报实测）
        seen_page: set[str] = set()
        uniq: list[str] = []
        for ln in lines:
            key = ln.strip()
            if key in seen_page:
                continue
            seen_page.add(key)
            uniq.append(ln)
        page_lines.append((page_no, uniq))
        for key in {ln.strip()[:20] for ln in uniq if len(ln.strip()) <= 20}:
            freq[key] = freq.get(key, 0) + 1

    # 页眉/页脚：短行跨页高频重复（≥4 页且 ≥30% 页数）——旧逻辑只拿首页
    # 种子比对，"汇聚财智"类每页页眉漏网进索引
    n_pages = max(len(page_lines), 1)
    drop_keys = {k for k, c in freq.items()
                 if c >= 4 and c >= n_pages * 0.3}

    for page_no, lines in page_lines:
        cleaned = [
            ln for ln in lines
            if ln.strip()[:20] not in drop_keys
            and not _FOOTER_PAT.match(ln.strip())
        ]
        if cleaned:
            result.pages.append(("\n".join(cleaned), page_no + 1))

        # 表格抽取
        try:
            tabs = doc[page_no].find_tables()
            for t in tabs.tables:
                data = t.extract()
                if data and len(data) >= 2:
                    headers = [str(h or "").strip() for h in data[0]]
                    rows = [[str(c or "").strip() for c in r] for r in data[1:]]
                    result.tables.append(TableBlock(
                        caption=f"（第 {page_no + 1} 页表）",
                        headers=headers, rows=rows, page=page_no + 1,
                    ))
        except Exception:  # noqa: BLE001 —— 表格抽取失败不阻断文本入库
            result.warnings.append(f"page {page_no + 1}: 表格抽取失败")

    if total_text < 50:
        result.ocr_required = True
        result.warnings.append("文档无文本层或文本极少，需人工 OCR 后入库")

    doc.close()
    return result
