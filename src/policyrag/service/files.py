"""文件接收层：GUI 上传 / 路径引用 / MCP base64 三种入口归一。

落临时文件 → ingest_file → 清理。支持 pdf/docx/md/txt。
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from ..models.ids import desurrogate

__all__ = ["ingest_bytes", "ingest_path_ref", "SUPPORTED_EXTS"]

SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}

MAX_BYTES = 64 * 1024 * 1024  # 64MB 上限


def ingest_bytes(svc: Any, data: bytes, filename: str, **meta: Any) -> dict[str, Any]:
    """二进制内容入库（GUI 上传 / MCP base64）。"""
    # 浏览器 multipart 文件名可能带孤立代理字符（surrogate）——先清理再 strip
    filename = desurrogate(filename or "upload.bin").strip()
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        return {"status": "rejected",
                "reason": f"不支持的文件类型 {ext!r}，支持: {sorted(SUPPORTED_EXTS)}"}
    if len(data) > MAX_BYTES:
        return {"status": "rejected", "reason": f"文件过大 {len(data)/1e6:.0f}MB > {MAX_BYTES/1e6:.0f}MB"}
    if not data.strip():
        return {"status": "rejected", "reason": "空文件"}

    # 落临时文件（UTF-8 路径安全：用 tempfile 前缀 ASCII + 自建名）
    safe_stem = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:10]
    tmp_dir = Path(tempfile.gettempdir()) / "policyrag_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{safe_stem}{ext}"
    tmp_path.write_bytes(data)
    # PDF 持久留存（公式/图表按需转写需要源文件——临时文件入库后会被删除）
    keep_path = tmp_path
    if ext == ".pdf":
        try:
            keep_dir = Path(os.environ.get("POLICYRAG_DATA", "data")) / "sources"
            keep_dir.mkdir(parents=True, exist_ok=True)
            keep_path = keep_dir / f"{safe_stem}{ext}"
            keep_path.write_bytes(data)
        except Exception:  # noqa: BLE001
            keep_path = tmp_path
    try:
        # G8：内容提取链优先（《》/文号/题名行），原始文件名仅作兜底提示——
        # 不再 setdefault doc_id/title=文件名（曾短路内容提取导致乱码标题入库）
        out = svc.ingest(tmp_path, filename_stem=Path(filename).stem, **meta)
        # 把持久副本路径写进文档 meta（供按需转写用）
        if out.get("status") == "ok" and keep_path != tmp_path:
            try:
                doc = svc.meta.get_document(out["doc_key"])
                import json as _j
                m = _j.loads(doc.get("meta_json") or "{}")
                m["source_file"] = str(keep_path)
                svc.meta.upsert_document(out["doc_id"], out["version"], doc["title"],
                                         effective_date=doc.get("effective_date"),
                                         status=doc.get("status", "active"), meta=m)
                svc.meta.commit()
            except Exception:  # noqa: BLE001
                pass
        return out
    except Exception as e:  # noqa: BLE001 —— 任何解析失败都回结构化结果，不冒 500
        return {"status": "rejected", "reason": f"{type(e).__name__}: {str(e)[:300]}",
                "filename": filename}
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def ingest_path_ref(svc: Any, path: str, **meta: Any) -> dict[str, Any]:
    """路径引用入库（agent 输入框里的文件路径）。支持引号/Windows 反斜杠。"""
    p = Path(desurrogate(path).strip().strip('"').strip("'"))
    if not p.exists():
        return {"status": "rejected", "reason": f"路径不存在: {p}"}
    if p.suffix.lower() not in SUPPORTED_EXTS:
        return {"status": "rejected",
                "reason": f"不支持的文件类型 {p.suffix}，支持: {sorted(SUPPORTED_EXTS)}"}
    try:
        return svc.ingest(p, **meta)
    except Exception as e:  # noqa: BLE001 —— 解析失败回结构化结果，不冒 500
        return {"status": "rejected", "reason": f"{type(e).__name__}: {str(e)[:300]}",
                "path": str(p)}
