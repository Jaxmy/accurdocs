"""离线 ingestion 编排 A1-A9（计划 §四 Pipeline A）。

确定性步骤（解析/归一化/结构/分块/ID/索引）——同输入必同输出。
摘要（A6）与版本（A9）在此层的简化实现，v1 扩展。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..index.stores.sqlite_meta import MetaStore
from ..models.document import DocAST
from ..models.document import BlockKind
from ..models.ids import (ClausePath, desurrogate, make_chunk_id, make_doc_key,
                        make_rich_chunk_id, normalize_text_for_id, sanitize_doc_id)
from .chunker import ChunkConfig, chunk_document
from .normalize import canonical_date, map_symbol_pua
from .structure import build_ast

__all__ = ["ingest_file", "IngestReport", "build_ingest_notice"]

# 文号提取：财办金〔2023〕15号 / 银发〔2024〕15号
_DOC_NO_RE = re.compile(r"([一-鿿]{2,8}〔\d{2,4}〕\d+\s*号)")
_EFFECTIVE_RE = re.compile(
    r"(?:自)?(\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?)(?:起)?(?:施行|生效|实施)"
)


def _marker_summary(mk: dict) -> str:
    bits = [mk.get(k) for k in ("doc_no", "version_marker", "draft_state", "period")
            if mk.get(k)]
    return "、".join(bits) if bits else "未识别到版本标记"


def build_ingest_notice(rep: dict) -> str:
    """入库回执文本（GUI 文本框展示 / MCP 转告用户）。

    内容：版本判断（T0 标题快判 + T1 全文确判）→ 自动开启富化 → 大致时间 →
    扫描中可提问（提问可插队）。
    """
    st = rep.get("status")
    if st == "noop":
        return (f"内容与已有文档《{rep.get('title') or ''}》"
                f"（{rep.get('version') or ''}）一致，已跳过重复入库。")
    if st != "ok":
        return ""
    vj = rep.get("version_judgment") or {}
    mk = vj.get("t0_markers") or {}
    if vj.get("relation") == "same_doc_version":
        vline = (f"版本判断：标题快判「{_marker_summary(mk)}」，全文确判与库内"
                 f"《{vj.get('near_title') or '同族文档'}》"
                 f"（{vj.get('near_version') or ''}）属同一文档家族（版本演进），"
                 f"已归入版本组")
    elif vj.get("relation") == "near_dup":
        vline = (f"版本判断：标题快判「{_marker_summary(mk)}」，全文确判与库内"
                 f"《{vj.get('near_title') or '已有文档'}》高度相似（近似重复），"
                 "已归入同一版本组")
    else:
        vline = f"版本判断：标题快判「{_marker_summary(mk)}」，全文确判为新文档家族"
    n_f, n_c = rep.get("n_formulas") or 0, rep.get("n_charts") or 0
    if n_f or n_c:
        mins = max(1, round((rep.get("eta_seconds") or 0) / 60))
        scan = (f"已自动开启扫描：公式 {n_f} 处、图表 {n_c} 处，"
                f"预计约 {mins} 分钟完成。扫描过程中也可以提问——"
                "若问题命中正在转写的内容，会自动优先插队处理。")
    else:
        scan = "无需公式/图表扫描，入库即可提问。"
    warn = ""
    for w in rep.get("warnings") or []:
        if w.startswith("[vision_"):
            warn = "\n" + w.split("] ", 1)[-1]
    return (f"《{rep.get('title') or ''}》（{rep.get('version') or ''}）已入库。\n"
            f"{vline}。\n{scan}{warn}")


def ingest_file(
    path: Path,
    meta: MetaStore,
    *,
    doc_id: str | None = None,
    version: str | None = None,
    title: str | None = None,
    effective_date: str | None = None,
    filename_stem: str | None = None,
    chunk_cfg: ChunkConfig | None = None,
    enrich_formulas: bool = True,
    enrich_charts: bool = True,
    images_root: Path | None = None,
    deep: bool = False,
    use_vision: bool = True,
) -> dict[str, Any]:
    """文件 → DocAST → Chunks → SQLite registry。

    幂等：同内容重传 → noop 短路；同 key 异内容 → 版本后缀 -r{n}（绝不静默覆盖）。
    """
    suffix = path.suffix.lower()
    warnings: list[str] = []
    vision_info: dict = {}

    if suffix == ".pdf":
        from .parsers.pdf_parser import parse_pdf
        parsed = parse_pdf(str(path))
    elif suffix in (".docx", ".doc"):
        from .parsers.docx_parser import parse_docx
        parsed = parse_docx(str(path))
    elif suffix in (".md", ".markdown", ".txt"):
        from .parsers.md_parser import parse_md
        parsed = parse_md(str(path))
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")

    if getattr(parsed, "ocr_required", False):
        return {"status": "rejected", "reason": "ocr_required",
                "warnings": parsed.warnings}

    warnings += parsed.warnings

    # PDF/DOCX 抽取文本可能带孤立代理字符（无法 encode UTF-8 / 入 SQLite）——边界清理
    parsed.pages = [(map_symbol_pua(desurrogate(t)), p) for t, p in parsed.pages]
    for tb in getattr(parsed, "tables", []) or []:
        tb.caption = desurrogate(tb.caption)
        tb.headers = [desurrogate(h) for h in tb.headers]
        tb.rows = [[desurrogate(c) for c in r] for r in tb.rows]

    # 文档标识：显式参数 > 文号 > 标题提取链（内容优先——WS-E；文件名仅兜底）
    from .identity import (content_fingerprint, decide_identity, extract_title,
                           resolve_doc_key, simhash64)

    raw_text = "\n".join(t for t, _ in parsed.pages)
    stem = filename_stem or path.stem
    m_no = _DOC_NO_RE.search(raw_text[:2000])
    doc_no = m_no.group(1).replace(" ", "") if m_no else ""
    if doc_id is None:
        doc_id = doc_no or stem
    doc_id = sanitize_doc_id(doc_id)  # 真实文件名含（）《》空格等，清洗而非拒收
    title_source = "explicit"
    if title is None:
        title, title_source = extract_title(raw_text, stem)
        if title_source == "filename":
            warnings.append("[title_fallback] 内容未识别出标题，采用文件名")
    if version is None:
        if effective_date is None:
            m = _EFFECTIVE_RE.search(raw_text[:3000])
            if m:
                effective_date = canonical_date(m.group(1))
        version = effective_date or "undated"

    # 身份判定（WS-E）：内容指纹 noop 短路 → 近似并组 → -r{n} 防覆盖
    # 两级版本判定（用户决策）：T0 标题/首页快判（版本号/文号/稿态/财年季度）→
    # T1 全文指纹确判；同基名+版本标记强信号 → 大改版也归入同版本组
    from .identity import extract_version_markers
    chash = content_fingerprint(raw_text)
    t0_markers = extract_version_markers(raw_text[:4000], title)
    for row in meta.list_documents():
        if row.get("content_hash") and row["content_hash"] == chash:
            out = {"status": "noop", "reason": "内容与已有文档一致，跳过重复入库",
                   "doc_key": row["doc_key"], "doc_id": row["doc_id"],
                   "version": row["version"], "title": row["title"],
                   "effective_date": row.get("effective_date"),
                   "n_chunks": 0, "n_formulas": 0, "n_charts": 0,
                   "n_tables": 0, "rich_jobs": [], "warnings": warnings,
                   "version_judgment": {
                       "t0_markers": t0_markers, "relation": "duplicate",
                       "near_dup_of": row["doc_key"], "t1_confirmed": True}}
            out["notice"] = build_ingest_notice(out)
            return out
    ident = decide_identity(doc_id=doc_id, version=version, title=title,
                            doc_no=doc_no, raw_text=raw_text,
                            existing=meta.list_documents())
    if version == "undated":
        version = f"v{ident.version_seq}"
    version, _ = resolve_doc_key(meta, doc_id, version, chash)
    # 版本判语（近邻标题供回执展示）
    near_title = near_version = ""
    if ident.near_dup_of:
        for row in meta.list_documents():
            if row.get("doc_key") == ident.near_dup_of:
                near_title, near_version = row.get("title") or "", row.get("version") or ""
                break
    version_judgment = {
        "t0_markers": t0_markers, "relation": ident.relation,
        "near_dup_of": ident.near_dup_of, "near_title": near_title,
        "near_version": near_version,
        "doc_group": ident.doc_group, "version": version,
        "t1_confirmed": True,
    }

    # A2-A4：结构 + 分块
    nodes, struct_warnings = build_ast(parsed.pages, parsed.tables)
    warnings += [f"[{w.kind}] {w.detail} @line {w.line_no}" for w in struct_warnings]

    ast = DocAST(
        doc_id=doc_id, version=version, title=title,
        nodes=nodes, effective_date=effective_date,
    )
    chunks = chunk_document(ast, chunk_cfg or ChunkConfig())

    # A5/A8：写 registry（幂等：chunk_id 冲突即覆盖）
    # 原子性：先写全部 chunk 再注册文档——中途失败不留下"有名无实"的半截文档
    doc_key = make_doc_key(doc_id, version)
    status = "active" if effective_date else "temporal_unknown"

    # A6b/A6c：富化快路径——占位块（截图）秒级入库，转写入后台队列渐进升级
    # deep=True 时同步跑完（测试/CLI 用）；默认异步（低配设备友好，不阻塞入库）
    formula_count = chart_count = 0
    pending_jobs: list[str] = []
    written_ids: set[str] = set()   # 本次写入全集（含富化占位）——替换语义用
    img_root = images_root or (path.parent / "rich_assets")
    if path.suffix.lower() == ".pdf" and (enrich_formulas or enrich_charts):
        from .enrichment import file_hash_of, get_worker, triage_pdf
        try:
            triage = triage_pdf(path)
        except Exception as e:  # noqa: BLE001
            triage = {"formula_pages": [], "chart_pages": []}
            warnings.append(f"富化漏斗失败: {type(e).__name__}")

        if enrich_formulas and triage["formula_pages"]:
            try:
                import pymupdf
                from .formulas import crop_formula_image
                from ..models.rich import FormulaRecord
                doc = pymupdf.open(path)
                img_dir = Path(img_root) / "formulas"
                img_dir.mkdir(parents=True, exist_ok=True)
                for pg in triage["formula_pages"]:
                    local_id = f"f_{pg:03d}_01"
                    img = crop_formula_image(doc, pg, "计算公式",
                                             img_dir / f"{local_id}.png")
                    # 占位文本带页面锚句——让公式块本身可被检索命中
                    anchor_txt = ""
                    try:
                        for w in doc[pg - 1].get_text("words"):
                            if any(k in w[4] for k in ("计算公式", "计算方式", "公式如下", "式中")):
                                anchor_txt = w[4]
                                break
                    except Exception:  # noqa: BLE001
                        pass
                    fr = FormulaRecord(
                        local_id=local_id, page=pg, latex="",
                        display_text=f"（第{pg}页公式，原文见附图，文本转写进行中）{('锚:' + anchor_txt) if anchor_txt else ''}",
                        image_path=img, transcribed_by="pending", confidence="low")
                    formula_count += 1
                    _cid = make_rich_chunk_id(doc_id, version, "formula", local_id)
                    written_ids.add(_cid)
                    meta.upsert_chunk(
                        chunk_id=_cid,
                        doc_key=doc_key, path_key=f"公式/{local_id}",
                        breadcrumb=f"第{pg}页 公式{local_id}",
                        text=fr.evidence_text(), text_norm=fr.evidence_text(),
                        kind=BlockKind.FORMULA.value, page=pg,
                        meta={"latex": "", "image": fr.image_path, "pending": True,
                              "confidence": "low", "transcribed_by": "pending"})
                doc.close()
            except Exception as e:  # noqa: BLE001
                warnings.append(f"公式占位失败: {type(e).__name__}: {str(e)[:100]}")

        if enrich_charts and triage["chart_pages"]:
            # 占位秒级入库（pending=True），OCR 全部后台渐进——上传不被 RapidOCR 阻塞
            try:
                from .charts import chart_placeholders
                phs, cwarn = chart_placeholders(path, Path(img_root) / "charts")
                warnings += cwarn
                chart_ids = []
                for ph in phs:
                    chart_count += 1
                    chart_ids.append(ph["local_id"])
                    body = f"（第{ph['page']}页图表{('：' + ph['title']) if ph['title'] else ''}，图内文字转写进行中）"
                    _cid = make_rich_chunk_id(doc_id, version, "chart", ph["local_id"])
                    written_ids.add(_cid)
                    meta.upsert_chunk(
                        chunk_id=_cid,
                        doc_key=doc_key, path_key=f"图表/{ph['local_id']}",
                        breadcrumb=f"第{ph['page']}页 图表{ph['local_id']}",
                        text=body, text_norm=body,
                        kind=BlockKind.CHART.value, page=ph["page"],
                        meta={"title": ph["title"], "chart_type": "",
                              "image": ph["image"], "confidence": "low",
                              "ocr_texts": [], "pending": True,
                              "page_no": ph["page"]})
            except Exception as e:  # noqa: BLE001
                chart_ids = []
                warnings.append(f"图表占位失败: {type(e).__name__}: {str(e)[:100]}")
        else:
            chart_ids = []

        # 入队（或 deep 同步执行）
        if triage["formula_pages"] or chart_ids:
            worker = get_worker(meta)
            if triage["formula_pages"]:
                jid = worker.submit(doc_key, doc_id, version, path, "formula",
                                    pages=triage["formula_pages"])
                pending_jobs.append(jid)
            if chart_ids:
                # 视觉能力验证（MCP 无 GUI 可提示 → 结果随响应回报调用方）；
                # 能力探测为准（宿主共享键未必有视觉能力）
                if use_vision:
                    from ..llm.vision import probe_vision
                    vision_info = probe_vision()
                    if not vision_info.get("capable"):
                        warnings.append(
                            "[vision_unavailable] 检测到复杂图表 "
                            f"{len(chart_ids)} 个，但{vision_info.get('reason', '')}——"
                            "图内数值不转写，图表数值类问题将拒答。请配置带视觉能力的 API Key"
                            "（POLICYRAG_VISION_API_KEY 或更换共享键）后重跑图表转写。")
                else:
                    vision_info = {"capable": False, "checked": False,
                                   "reason": "调用方选择不使用视觉 API"}
                    warnings.append(
                        "[vision_skipped] 按调用方选择不使用视觉 API——"
                        "图内数值不转写，图表数值类问题将拒答。")
                jid = worker.submit(doc_key, doc_id, version, path, "chart",
                                    chart_local_ids=chart_ids, use_vision=use_vision)
                pending_jobs.append(jid)
            if deep:
                for jid in pending_jobs:
                    worker.run_inline(jid)

    try:
        for ch in chunks:
            written_ids.add(ch.chunk_id)
            meta.upsert_chunk(
                chunk_id=ch.chunk_id, doc_key=doc_key,
                path_key=ch.path.key, breadcrumb=ch.breadcrumb,
                text=ch.text, text_norm=ch.text_norm, kind=ch.kind.value,
                page=ch.page, summary=ch.summary,
                meta={"parent": ch.parent_chunk_id, "row_index": ch.row_index},
            )
        # 替换语义：清掉同 doc_key 的孤儿旧 chunk（解析结果变化时旧 ID 不残留检索池）
        stale_n = meta.replace_document_chunks(doc_key, written_ids)
        if stale_n:
            warnings.append(f"[reingest_cleanup] 清理 {stale_n} 个旧分块")
        meta.upsert_document(
            doc_id=doc_id, version=version, title=title,
            effective_date=effective_date, status=status,
            meta={"source_file": str(path), "warnings": warnings,
                  "title_source": title_source,
                  "version_markers": t0_markers},
            content_hash=chash, simhash=str(simhash64(raw_text[:20000])),
            doc_group=ident.doc_group, version_seq=ident.version_seq,
            is_latest=1, doc_no=doc_no,
        )
        meta.set_latest_version(ident.doc_group, doc_key)
        meta.commit()
    except Exception:
        # 回滚半截状态：清掉本次已写的 chunk，文档注册不落
        meta.delete_document(doc_key)
        raise

    # ETA：扫描大致时间（回执提醒用）——MinerU ~8s/页，图表 OCR ~5s/图（+VLM ~6s）
    from .enrichment import estimate_rich_seconds
    eta_seconds = (estimate_rich_seconds(len(triage.get("formula_pages") or []),
                                         chart_count, use_vision)
                   if pending_jobs else 0)

    out = {
        "status": "ok",
        "doc_id": doc_id,
        "n_formulas": formula_count,
        "n_charts": chart_count,
        "needs_vision": bool(chart_count >= 5),
        "vision": vision_info,
        "rich_jobs": pending_jobs,
        "eta_seconds": eta_seconds,
        "version": version,
        "doc_key": doc_key,
        "title": title,
        "effective_date": effective_date,
        "n_chunks": len(chunks),
        "n_tables": len(parsed.tables),
        "doc_group": ident.doc_group,
        "version_seq": ident.version_seq,
        "near_dup_of": ident.near_dup_of,
        "content_hash": chash,
        "warnings": warnings,
        "version_judgment": version_judgment,
    }
    out["notice"] = build_ingest_notice(out)
    return out
