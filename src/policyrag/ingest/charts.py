"""图表抽取（图表线）：图区域检测 → RapidOCR 标签 → ChartRecord。

实测结论（PoC）：带标签柱图/饼图/图内表格 OCR 即得精确值；
无标签折线的系列数据由可选 DePlot 离线补充（scripts/chart_deplot.py）。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models.ids import desurrogate
from ..models.rich import ChartRecord

__all__ = ["build_chart_records", "chart_placeholders", "classify_chart"]

# 有效图表门槛：位图边长（瓦片碎图/水印排除）
_MIN_SIDE = 120
_TITLE_RE = re.compile(r"(图\s*\d+[：:．.]\s*[^\n]{2,60}|图表[：:][^\n]{2,60})")
_SOURCE_RE = re.compile(r"(资料来源[：:][^\n]{2,80}|来源[：:][^\n]{2,80})")
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")


def classify_chart(texts: list[str]) -> tuple[str, str]:
    """OCR 文字 → (chart_type, confidence) 启发式分类（PoC 实测定型）。"""
    joined = " ".join(texts)
    n_num = len(_NUMERIC_RE.findall(joined))
    if sum(1 for t in texts if "|" in t or t.count(" ") >= 2) > 5 and n_num >= 6:
        ctype = "table"
    elif n_num >= 8:
        ctype = "bar"
    elif any("%" in t for t in texts) and n_num >= 3:
        ctype = "pie"
    else:
        ctype = "unknown"
    conf = "high" if (ctype in ("table", "bar") and n_num >= 6) else "medium"
    return ctype, conf


def chart_placeholders(pdf_path: str | Path, images_dir: str | Path,
                       max_per_doc: int = 60) -> tuple[list[dict], list[str]]:
    """秒级图表占位：截图 + 标题锚 + local_id，**不做 OCR**（后台 worker 转写）。

    返回 (placeholders, warnings)；placeholder = {local_id, page, title, image}。
    """
    import io

    import pymupdf
    from PIL import Image

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    out: list[dict] = []

    doc = pymupdf.open(pdf_path)
    n_done = 0
    for pno in range(len(doc)):
        if n_done >= max_per_doc:
            warnings.append(f"图表数达到上限 {max_per_doc}，截断")
            break
        page = doc[pno]
        page_text = desurrogate(page.get_text())
        title_m = _TITLE_RE.search(page_text)

        seen_xrefs = set()
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                px = pymupdf.Pixmap(doc, xref)
                if min(px.width, px.height) < _MIN_SIDE:
                    continue
                if px.width * px.height < 20000:
                    continue
                page_area = page.rect.width * page.rect.height * 4
                if px.width * px.height > page_area * 1.2:
                    continue
                pix_bytes = px.tobytes("png")
            except Exception:  # noqa: BLE001
                continue

            n_done += 1
            local_id = f"c_{pno+1:03d}_{n_done:02d}"
            img_path = images_dir / f"{local_id}.png"
            try:
                Image.open(io.BytesIO(pix_bytes)).save(str(img_path))
            except Exception:  # noqa: BLE001
                continue
            out.append({
                "local_id": local_id, "page": pno + 1,
                "title": title_m.group(1).strip() if title_m else "",
                "image": str(img_path),
            })
    doc.close()
    return out, warnings


def build_chart_records(pdf_path: str | Path, images_dir: str | Path,
                        max_per_doc: int = 60) -> tuple[list[ChartRecord], list[str]]:
    """PDF → ChartRecord（截图 + OCR 文字）。OCR 引擎懒加载。"""
    import io

    import pymupdf
    from PIL import Image

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    records: list[ChartRecord] = []

    try:
        from rapidocr import RapidOCR
        ocr = RapidOCR()
    except Exception as e:  # noqa: BLE001
        return [], [f"rapidocr 不可用: {type(e).__name__}"]

    doc = pymupdf.open(pdf_path)
    n_done = 0
    for pno in range(len(doc)):
        if n_done >= max_per_doc:
            warnings.append(f"图表数达到上限 {max_per_doc}，截断")
            break
        page = doc[pno]
        page_text = desurrogate(page.get_text())
        title_m = _TITLE_RE.search(page_text)
        source_m = _SOURCE_RE.search(page_text)

        seen_xrefs = set()
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                px = pymupdf.Pixmap(doc, xref)
                if min(px.width, px.height) < _MIN_SIDE:
                    continue
                if px.width * px.height < 20000:   # 太小不计
                    continue
                # 整页背景图跳过（图表通常是局部嵌入）
                page_area = page.rect.width * page.rect.height * 4
                if px.width * px.height > page_area * 1.2:
                    continue
                pix_bytes = px.tobytes("png")
            except Exception:  # noqa: BLE001
                continue

            n_done += 1
            local_id = f"c_{pno+1:03d}_{n_done:02d}"
            img_path = images_dir / f"{local_id}.png"
            try:
                Image.open(io.BytesIO(pix_bytes)).save(str(img_path))
            except Exception:  # noqa: BLE001
                continue

            try:
                out = ocr(str(img_path))
                texts = [desurrogate(t) for t in (list(out.txts) if out and out.txts is not None else [])]
            except Exception as e:  # noqa: BLE001
                texts = []
                warnings.append(f"{local_id} OCR 失败: {type(e).__name__}")

            ctype, conf = classify_chart(texts)
            records.append(ChartRecord(
                local_id=local_id, page=pno + 1,
                title=title_m.group(1).strip() if title_m else "",
                chart_type=ctype,
                ocr_texts=texts[:80],
                data_source=source_m.group(1).strip() if source_m else "",
                image_path=str(img_path),
                extracted_by="rapidocr", confidence=conf,
                meta={"n_numeric": n_num},
            ))
    doc.close()
    return records, warnings
