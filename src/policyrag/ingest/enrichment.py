"""富化引擎：选择性漏斗 + 异步渐进 + 磁盘缓存 + 按需深转。

设计约束（用户决策）：
- 不并行——照顾内存/缓存弱于本机的设备；单线程串行 worker + 单飞锁；
- 入库只做秒级漏斗（锚行/图像检测+占位块），MinerU/OCR 全部后台渐进；
- (file_hash, page, kind) 磁盘缓存——重入库/版本更新只跑新页；
- 提问命中未转换公式/图表 → 当场提交转换任务（enqueue 立即执行），
  前端进度条轮询 job 状态。
"""

from __future__ import annotations

import hashlib
import queue
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from ..models.ids import desurrogate, make_rich_chunk_id
from ..models.rich import FormulaRecord, FormulaSymbol

__all__ = [
    "file_hash_of", "triage_pdf", "RichWorker", "get_worker",
    "rich_chunk_ids", "estimate_rich_seconds", "merge_symbol_lines",
]

# ---------------------------------------------------------------- 工具

def file_hash_of(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


_MATH_GLYPH_RE = re.compile(r"[∑∫√±×÷≤≥≈∂Δ∇]")
_FORMULA_ANCHOR_RE = re.compile(r"(计算公式|计算方式|计算如下|公式如下|式中|公式中)")
_PUA_RE = re.compile("[-]")


def triage_pdf(pdf_path: str | Path) -> dict[str, list[int]]:
    """秒级漏斗：公式候选页 + 图表候选页（不做任何识别）。"""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    formula_pages: list[int] = []
    chart_pages: list[int] = []
    for pno in range(len(doc)):
        page = doc[pno]
        t = desurrogate(page.get_text())
        n_pua = len(_PUA_RE.findall(t))
        if (_MATH_GLYPH_RE.search(t) or _FORMULA_ANCHOR_RE.search(t) or n_pua >= 2):
            formula_pages.append(pno + 1)
        for img in page.get_images(full=True):
            try:
                px = pymupdf.Pixmap(doc, img[0])
                if min(px.width, px.height) >= 120 and px.width * px.height >= 20000:
                    chart_pages.append(pno + 1)
                    break
            except Exception:  # noqa: BLE001
                continue
    doc.close()
    return {"formula_pages": formula_pages, "chart_pages": chart_pages}


def rich_chunk_ids(doc_id: str, version: str, kind: str, local_id: str) -> str:
    return make_rich_chunk_id(doc_id, version, kind, local_id)


def estimate_rich_seconds(n_formula_pages: int, n_charts: int,
                          use_vision: bool = False) -> int:
    """扫描大致时间（回执提醒用）：MinerU ~8s/页，图表 OCR ~5s/图（+VLM ~6s/图）。"""
    sec = n_formula_pages * 8 + n_charts * (5 + (6 if use_vision else 0))
    return max(15, sec) if (n_formula_pages or n_charts) else 0


def merge_symbol_lines(text: str, symbols: list[dict]) -> str:
    """meta.symbols → 公式块文本【符号】行（存量块修复/升级共用；幂等）。

    符号表（"式中"说明）落 meta 不落文本时，公式块纯 LaTeX 文本对中文问题
    不可检索、符号含义不可逐字引用——必须并入 text（chunk_id 不变）。
    """
    if not symbols:
        return text or ""
    if "【符号】" in (text or ""):
        return text or ""
    lines = [f"【符号】{s.get('name', '')} 为 {s.get('desc', '')}"
             for s in symbols if s.get("desc")]
    if not lines:
        return text or ""
    return ((text or "").rstrip() + "\n" + "\n".join(lines)).strip()


# ---------------------------------------------------------------- Worker

class RichWorker:
    """串行后台富化 worker（单线程队列，低内存设备友好）。

    优先级插队（用户决策：富化进程中用户提问可插队）：
    - priority 0 = 提问/按需触发；priority 1 = 入库批量；
    - 批量任务在页/图粒度检查插队信号，让出后重排队尾，插队任务先跑；
    - 串行不并行不变——同一时刻仍只跑一个任务。
    """

    def __init__(self, meta, on_index_refresh: Callable[[], None] | None = None):
        self.meta = meta
        self.on_index_refresh = on_index_refresh
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()   # 单飞：全局同一时刻只跑一个富化任务
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._p0_waiting = 0            # 排队中的插队任务数（批量让行信号）
        self._prio: dict[str, int] = {}
        self._done_events: dict[str, threading.Event] = {}

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _enqueue(self, job_id: str, priority: int) -> None:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
            if priority <= 0:
                self._p0_waiting += 1
        self._q.put((priority, seq, job_id))

    def _pop(self) -> str:
        priority, _, job_id = self._q.get()   # 阻塞取最高优先级
        if priority <= 0:
            with self._seq_lock:
                self._p0_waiting = max(0, self._p0_waiting - 1)
        return job_id

    def _should_yield(self, job_id: str) -> bool:
        """批量任务让行信号：有插队任务排队且自身非插队 → 暂停重排。"""
        return self._p0_waiting > 0 and self._prio.get(job_id, 1) > 0

    def _yield_requeue(self, job_id: str) -> bool:
        """让行：本任务降为批量优先级重排（缓存保进度，续跑不重做）。"""
        self._prio[job_id] = 1
        self.meta.job_update(job_id, "running", 0.0, "插队让行：提问优先，稍后续跑")
        self._enqueue(job_id, 1)
        return True

    def _signal_done(self, job_id: str) -> None:
        ev = self._done_events.pop(job_id, None)
        if ev:
            ev.set()

    def submit(self, doc_key: str, doc_id: str, version: str, file_path: str,
               kind: str, pages: list[int] | None = None,
               chart_local_ids: list[str] | None = None,
               use_vision: bool = False, priority: int = 1) -> str:
        """提交富化任务，立即返回 job_id（不阻塞调用方）。

        priority：0=提问/按需触发（插队），1=入库批量（默认）。
        """
        job_id = f"rj_{uuid.uuid4().hex[:12]}"
        payload = {
            "doc_id": doc_id, "version": version,
            "file_path": str(file_path), "file_hash": file_hash_of(file_path),
            "pages": pages or [], "chart_local_ids": chart_local_ids or [],
            "use_vision": bool(use_vision), "priority": int(priority),
        }
        self.meta.job_upsert(job_id, doc_key, kind, payload, status="queued")
        self._prio[job_id] = int(priority)
        self._enqueue(job_id, int(priority))
        self._ensure_thread()
        return job_id

    def run_inline(self, job_id: str) -> None:
        """当场立即转换（按需深转，提问路径）——提到最高优先级并等待完成。

        批量任务在页/图粒度让行，本任务先跑（串行不并行）。
        """
        self._prio[job_id] = 0
        ev = self._done_events.setdefault(job_id, threading.Event())
        self._enqueue(job_id, 0)
        self._ensure_thread()
        if not ev.wait(timeout=900):
            # 兜底：worker 卡死时在调用方线程直接跑（仍走单飞锁）
            with self._lock:
                job = self.meta.job_get(job_id)
                if job and job["status"] not in ("done", "failed"):
                    self._process(job_id)

    def resume_pending(self) -> int:
        """进程重启续跑：扫 rich_jobs 中 queued/running 重新入队（内存队列不丢任务）。"""
        jobs = self.meta.job_list(status="running") + self.meta.job_list(status="queued")
        n = 0
        for job in jobs:
            if job["status"] == "running":
                self.meta.job_update(job["job_id"], "queued",
                                     job.get("progress") or 0.0,
                                     "resumed after restart")
            prio = int((job.get("payload") or {}).get("priority", 1))
            self._prio[job["job_id"]] = prio
            self._enqueue(job["job_id"], prio)
            n += 1
        if n:
            self._ensure_thread()
        return n

    def _loop(self) -> None:
        while True:
            job_id = self._pop()
            with self._lock:
                self._process(job_id)

    # ---------------------------------------------------------- 处理

    def _process(self, job_id: str) -> None:
        job = self.meta.job_get(job_id)
        if not job or job["status"] == "done":
            self._signal_done(job_id)
            return
        payload = job["payload"]
        kind = job["kind"]
        try:
            if kind == "formula":
                yielded = self._process_formulas(job_id, job["doc_key"], payload)
            else:
                yielded = self._process_charts(job_id, job["doc_key"], payload)
            if yielded:
                return                      # 让行重排——不算完成，不发完成信号
            self.meta.job_update(job_id, "done", 1.0, "完成")
        except Exception as e:  # noqa: BLE001
            self.meta.job_update(job_id, "failed", 1.0,
                                 f"{type(e).__name__}: {str(e)[:120]}")
        if self.on_index_refresh:
            try:
                self.on_index_refresh()
            except Exception:  # noqa: BLE001
                pass
        self._signal_done(job_id)

    def _process_formulas(self, job_id: str, doc_key: str, payload: dict) -> bool:
        from .formulas import (extract_symbols, parse_mineru_markdown,
                               run_mineru, validate_latex)

        fhash = payload["file_hash"]
        pages = payload["pages"]
        doc_id, version = payload["doc_id"], payload["version"]

        # 1) 缓存过滤——空 records 缓存仅在该页已无占位块时可信（否则视为未命中重试，
        #    防"MinerU 一次失败→空结果永久缓存→占位块永不升级"的毒化）
        placeholder_pages = {r.get("page") for r in self.meta.chunks_of_doc(doc_key)
                             if r.get("kind") == "formula"
                             and "转写进行中" in (r.get("text") or "")}
        pending: list[int] = []
        cached: dict[int, list] = {}
        for pg in pages:
            hit = self.meta.enrich_cache_get(fhash, pg, "formula")
            if hit and (hit["records"] or pg not in placeholder_pages):
                cached[pg] = hit["records"]
            else:
                pending.append(pg)
        n_total = max(len(pages), 1)
        self.meta.job_update(job_id, "running", 1 - len(pending) / n_total,
                             f"缓存命中 {len(pages) - len(pending)}/{len(pages)} 页")

        # 2) 未命中页跑 MinerU（小批量串行——批间是插队让行点）
        if pending:
            import tempfile
            from pathlib import Path as _P
            for b0 in range(0, len(pending), 4):
                if self._should_yield(job_id):
                    return self._yield_requeue(job_id)
                batch = pending[b0:b0 + 4]
                with tempfile.TemporaryDirectory(prefix="mineru_") as td:
                    out_file = _P(td) / "out.md"
                    pages_arg = ",".join(str(p) for p in sorted(batch))
                    md = run_mineru(payload["file_path"], out_file, pages=pages_arg)
                if md:
                    import pymupdf
                    doc = pymupdf.open(payload["file_path"])
                    raw_text = desurrogate("\n".join(p.get_text() for p in doc))
                    from ..models.ids import normalize_text_for_id
                    text_norm = normalize_text_for_id(raw_text)
                    raws = parse_mineru_markdown(md)
                    by_page: dict[int, list] = {}
                    for r in raws:
                        fixes: list[str] = []
                        latex = validate_latex(r["latex"], text_norm, fixes)
                        by_page.setdefault(r["page"], []).append({
                            "latex": latex,
                            "symbols": [{"name": s.name, "desc": s.desc}
                                        for s in extract_symbols(r["tail"])],
                            "context_before": r["before"], "fixes": fixes,
                        })
                    for pg in batch:
                        recs = by_page.get(pg, [])
                        self.meta.enrich_cache_put(fhash, pg, "formula", {"records": recs})
                        cached[pg] = recs
                    doc.close()
                else:
                    # MinerU 失败：绝不缓存空结果（毒化缓存会永久阻断重试）——留待重试
                    for pg in batch:
                        hit = self.meta.enrich_cache_get(fhash, pg, "formula")
                        cached[pg] = hit["records"] if hit else []
                    self.meta.job_update(job_id, "running", 0.9,
                                         f"MinerU 失败，{len(batch)} 页留待重试（不缓存空结果）")
                self.meta.job_update(job_id, "running", 0.9,
                                     f"MinerU 完成 {b0 + len(batch)}/{len(pending)} 页")

        # 3) 升级占位块（稳定 ID 覆盖）
        from ..models.document import BlockKind
        from .formulas import crop_formula_image
        import pymupdf

        doc = pymupdf.open(payload["file_path"])
        img_dir = _P = Path(payload["file_path"]).parent / "rich_assets" / "formulas"
        img_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for pg, recs in sorted(cached.items()):
            if self._should_yield(job_id):
                doc.close()
                return self._yield_requeue(job_id)   # 已入缓存，续跑从缓存升级
            for i, rec in enumerate(recs, 1):
                local_id = f"f_{pg:03d}_{i:02d}"
                cid = make_rich_chunk_id(doc_id, version, "formula", local_id)
                img = crop_formula_image(doc, pg, rec.get("context_before", ""),
                                         img_dir / f"{local_id}.png")
                # 符号表必须进文本（【符号】行）——曾只落 meta.symbols，公式块
                # 纯 LaTeX 文本对中文问题不可检索、"式中"说明不可逐字引用
                fr = FormulaRecord(
                    local_id=local_id, page=pg, latex=rec.get("latex", ""),
                    display_text=rec.get("latex", "") or "（公式见附图）",
                    symbols=[FormulaSymbol(name=s.get("name", ""),
                                           desc=s.get("desc", ""))
                             for s in rec.get("symbols", [])],
                    image_path=img,
                    transcribed_by="mineru" if rec.get("latex") else "none",
                    confidence="high" if rec.get("latex") else "low",
                    fixes_applied=rec.get("fixes", []),
                )
                self.meta.upsert_chunk(
                    chunk_id=cid, doc_key=doc_key,
                    path_key=f"公式/{local_id}",
                    breadcrumb=f"第{pg}页 公式{local_id}",
                    text=fr.evidence_text(), text_norm=fr.evidence_text(),
                    kind=BlockKind.FORMULA.value, page=pg,
                    meta={"latex": rec.get("latex", ""), "image": fr.image_path,
                          "symbols": rec.get("symbols", []),
                          "confidence": fr.confidence, "pending": False},
                )
                n += 1
        self.meta.commit()
        doc.close()

        # 4) OCR 兜底：MinerU 解析不出的占位公式块（图片式）跑 RapidOCR，
        #    升级为可检索文本——占位文案"转写进行中"不残留、检索可见
        n_ocr = 0
        placeholders = [r for r in self.meta.chunks_of_doc(doc_key)
                        if r.get("kind") == "formula"
                        and "转写进行中" in (r.get("text") or "")]
        if placeholders:
            from .native_worker import ocr_isolated   # #15 子进程隔离
            for row in placeholders:
                if self._should_yield(job_id):
                    return self._yield_requeue(job_id)
                import json as _j
                try:
                    extra = _j.loads(row.get("meta_json") or "{}")
                except Exception:  # noqa: BLE001
                    extra = {}
                img = extra.get("image")
                texts: list[str] = []
                if img and Path(img).exists():
                    texts = [desurrogate(t) for t in ocr_isolated(img)]
                body = ("【公式·OCR】" + " ".join(texts[:40])) if texts else "（公式原文见附图）"
                extra.update({"transcribed_by": "ocr" if texts else "none",
                              "pending": False, "confidence": "low"})
                self.meta.upsert_chunk(
                    chunk_id=row["chunk_id"], doc_key=doc_key,
                    path_key=row["path_key"], breadcrumb=row["breadcrumb"],
                    text=body, text_norm=body,
                    kind=row["kind"], page=row.get("page"), meta=extra)
                n_ocr += 1
            self.meta.commit()
        note = f"升级 {n} 个公式块"
        if n_ocr:
            note += f"；OCR 兜底 {n_ocr} 个"
        if placeholders and not n_ocr:
            note += "；仍有占位公式未能转写"
        self.meta.job_update(job_id, "running", 1.0, note)
        return False

    def _process_charts(self, job_id: str, doc_key: str, payload: dict) -> bool:
        from ..models.document import BlockKind
        from .native_worker import ocr_isolated

        fhash = payload["file_hash"]
        ids = payload["chart_local_ids"]
        n_total = max(len(ids), 1)

        # 视觉转写（W2/W3）：use_vision 且探测通过才走 VLM；否则退回纯 OCR 并如实标注
        vlm = None
        if payload.get("use_vision"):
            from ..llm.vision import make_vision_llm, probe_vision
            probe = probe_vision()
            if probe.get("capable"):
                vlm = make_vision_llm()
                self.meta.job_update(job_id, "running", 0.0, "视觉 API 探测通过，VLM 转写启用")
            else:
                self.meta.job_update(job_id, "running", 0.0,
                                     f"视觉不可用（{probe.get('reason','')}），退回 OCR（图表数值题将拒答）")

        for i, local_id in enumerate(ids):
            if self._should_yield(job_id):
                return self._yield_requeue(job_id)   # 图粒度让行，缓存保进度
            cid = make_rich_chunk_id(payload["doc_id"], payload["version"], "chart", local_id)
            row = self.meta.get_chunk(cid)
            if not row:
                continue
            import json as _j
            try:
                extra = _j.loads(row.get("meta_json") or "{}")
            except Exception:  # noqa: BLE001
                extra = {}
            img = extra.get("image")
            hit = self.meta.enrich_cache_get(fhash, extra.get("page", 0), f"chart:{local_id}")
            if hit:
                texts = hit.get("ocr_texts", [])
            elif img and Path(img).exists():
                # #15 子进程隔离：onnxruntime 段错误只死子进程，服务不陪葬
                texts = [desurrogate(t) for t in ocr_isolated(img)]
                if texts:
                    self.meta.enrich_cache_put(fhash, extra.get("page", 0),
                                               f"chart:{local_id}", {"ocr_texts": texts})
            else:
                texts = []

            # VLM 转写 + 双源分层（准确度第一）：strong 值进证据文本可进保真比对，
            # weak 值只进展示数据表——防 VLM 误读被"答证一致"自我印证
            vdata: dict = {}
            strong, weak = [], []
            if vlm is not None and img and Path(img).exists():
                vhit = self.meta.enrich_cache_get(fhash, extra.get("page", 0),
                                                  f"chartvlm:{local_id}")
                if vhit:
                    vdata = vhit.get("vdata") or {}
                else:
                    try:
                        from ..llm.vision import transcribe_chart
                        vdata = transcribe_chart(vlm, Path(img).read_bytes()) or {}
                    except Exception:  # noqa: BLE001
                        vdata = {}
                    self.meta.enrich_cache_put(fhash, extra.get("page", 0),
                                               f"chartvlm:{local_id}", {"vdata": vdata})
                from ..llm.vision import tier_chart_values
                strong, weak = tier_chart_values(vdata.get("values") or [], texts)

            extra.update({"ocr_texts": texts, "pending": False})
            if vdata:
                extra["vlm"] = {k: vdata.get(k) for k in ("title", "chart_type", "values", "texts")}
                extra["chart_type"] = extra.get("chart_type") or vdata.get("chart_type")
            if not extra.get("chart_type"):
                ctype, conf = _classify(texts)
                extra["chart_type"] = ctype
                extra["confidence"] = conf
            title = extra.get("title") or (vdata.get("title") or "")
            extra["title"] = title

            parts = [f"【图表】{title}"] if title else []
            if strong:
                # 强证据数据表（双源一致数值，逐字形态）
                rows_t = ["【数据表】"]
                for v in strong:
                    rows_t.append(f"{v.get('series','')} | {v.get('axis','')} | {v.get('value','')}")
                parts.append("\n".join(rows_t))
                extra["transcribed_by"] = "vlm+ocr"
                extra["confidence"] = "high"
            elif vdata:
                parts.append("（图内数值识别存疑，见附图）")
                extra["transcribed_by"] = "vlm_weak"
                extra["confidence"] = "low"
            else:
                extra["transcribed_by"] = "ocr"
            extra["data_table"] = "" if not vdata else "\n".join(
                f"{v.get('series','')} | {v.get('axis','')} | {v.get('value','')}"
                for v in (vdata.get("values") or []))
            if texts:
                parts.append("【图内文字】" + " | ".join(texts[:80]))
            elif not parts:
                parts.append(f"【图表】{title}（图内文字见附图）")
            body = "\n".join(parts)
            self.meta.upsert_chunk(
                chunk_id=cid, doc_key=doc_key,
                path_key=f"图表/{local_id}",
                breadcrumb=f"第{extra.get('page','?')}页 图表{local_id}",
                text=body, text_norm=body,
                kind=BlockKind.CHART.value, page=extra.get("page"),
                meta=extra,
            )
            self.meta.job_update(job_id, "running", (i + 1) / n_total,
                                 f"图表 {i + 1}/{len(ids)}")
        self.meta.commit()
        return False


# ---------------------------------------------------------------- 全局单例

_worker: RichWorker | None = None
_wlock = threading.Lock()
_resumed = False


def _classify(texts: list[str]) -> tuple[str, str]:
    from .charts import classify_chart
    return classify_chart(texts)


def get_worker(meta, on_index_refresh: Callable[[], None] | None = None) -> RichWorker:
    """全局串行 worker。每次调用更新 meta/回调注册（不被首调用方冻结——历史 bug），
    首次获取时扫描未完成任务续跑。"""
    global _worker, _resumed
    with _wlock:
        if _worker is None:
            _worker = RichWorker(meta, on_index_refresh)
        else:
            _worker.meta = meta
            if on_index_refresh is not None:
                _worker.on_index_refresh = on_index_refresh
        if not _resumed:
            _resumed = True
            try:
                _worker.resume_pending()
            except Exception:  # noqa: BLE001 - 续跑失败不阻塞调用方
                pass
        return _worker
