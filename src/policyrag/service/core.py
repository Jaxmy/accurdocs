"""CoreService：REST/MCP 双 adapter 的唯一业务入口（计划 §三 service/core）。

adapter 只做 schema 转换 → 调这里 → 包装 envelope → 写审计。业务逻辑零重复。

多问题拆分：批量提问（10-20 问）自动拆为独立子问题逐题跑完整链路，
避免单次检索 top-8 证据被挤占（批量正确率 10%→逐题水平的根因修复）。
流式：ask_stream 逐题/逐 token 推送事件（SSE 用）。
"""

from __future__ import annotations

import hashlib
import queue
import threading
import numpy as np
import uuid
from pathlib import Path
from typing import Any, Iterator

from ..index.bm25 import BM25Index
from ..index.hybrid import HybridIndex
from ..index.numpy_store import NumpyVectorStore
from ..index.stores.sqlite_meta import MetaStore
from ..index.vectors import get_embedder
from ..llm.base import LLMClient
from ..models.answer import AnswerEnvelope, AnswerStatus, Refusal, Usage
from ..models.document import Chunk
from ..models.retrieval import QueryRequest
from ..query.pipeline import PipelineDeps, QueryPipeline
from ..ingest.pipeline import ingest_file
from ..verify.citations import verify_citations
from ..verify.deterministic import verify_numbers
from ..models.answer import Citation, VerifyReport
from . import reader_view

__all__ = ["CoreService", "ServiceConfig"]


class ServiceConfig:
    def __init__(
        self,
        data_dir: str | Path = "data",
        trace_dir: str | Path = "traces",
        embed_backend: str = "fastembed",
    ):
        self.data_dir = Path(data_dir)
        self.trace_dir = Path(trace_dir)
        self.embed_backend = embed_backend
        self.data_dir.mkdir(parents=True, exist_ok=True)


class CoreService:
    def __init__(self, llm: LLMClient, cfg: ServiceConfig | None = None):
        self.cfg = cfg or ServiceConfig()
        self.meta = MetaStore(self.cfg.data_dir / "meta.sqlite3")
        self.llm = llm
        self._embedder = get_embedder(self.cfg.embed_backend)
        self._bm25 = BM25Index(self.cfg.data_dir / "bm25")
        self._vectors = NumpyVectorStore(self.cfg.data_dir / "vectors", dim=self._embedder.dim)
        self._index: HybridIndex | None = None
        # 富化 worker：注册索引刷新回调 + 进程启动即续跑未完成任务（M0）
        try:
            from ..ingest.enrichment import get_worker
            get_worker(self.meta, on_index_refresh=self.rebuild_index)
        except Exception:  # noqa: BLE001
            pass

    # -- 索引构建 -----------------------------------------------------------

    def rebuild_index(self) -> dict[str, int]:
        """重建索引（ingest 后调用）。向量**增量**：只 embed 新 chunk，旧向量复用。

        之前每次上传全量重嵌（2000+ chunk × 40 chunks/s ≈ 50s），
        是"上传按钮假死"的性能根因。

        向量指纹守卫：embedding 后端/维度变化后旧向量**不可复用**（同维不同
        模型混搭是无声灾难——实测 hash/bge 混搭令 dense 侧整体失真），
        指纹不匹配 → 丢弃全部旧向量全量重嵌。
        """
        metas = self.meta.all_chunk_metas()
        ids = list(metas.keys())
        # BM25 语料 = 正文 + breadcrumb：条款号只存在于 breadcrumb（正文是条款体），
        # 不拼进去则"第四十条"类查询在稀疏侧永远打不中（WS-A 实测）。
        # #10：breadcrumb 清洗（去句中碎片噪声段）+ 数学符号归一（PUA→Unicode，
        # 数学模型块的表达式可被检索）——仅语料侧，存储与 chunk_id 不动。
        from ..models.clause_norm import clean_breadcrumb
        from ..ingest.normalize import map_symbol_pua
        bm25_texts = [
            f"{map_symbol_pua(metas[cid]['text'] or '')} "
            f"{clean_breadcrumb(metas[cid].get('breadcrumb') or '')}".strip()
            for cid in ids
        ]
        texts = [metas[cid]["text"] for cid in ids]   # 向量侧保持纯正文（缓存一致性）
        self._bm25.build(ids, bm25_texts)

        # embedding 指纹：身份变化 → 旧向量作废（防混搭）
        fp_path = self._fingerprint_path()
        embed_key = self._embed_key()
        full_rebuild = self._fingerprint_key() != embed_key

        if ids:
            if full_rebuild:
                old_ids, old_vecs = [], np.zeros((0,), dtype=np.float32)
            else:
                old_ids, old_vecs = self._vectors.snapshot()
            cached = dict(zip(old_ids, old_vecs))
            missing_idx = [i for i, cid in enumerate(ids) if cid not in cached]
            if missing_idx:
                new_vecs = self._embedder.embed([texts[i] for i in missing_idx])
                for i, vec in zip(missing_idx, new_vecs):
                    cached[ids[i]] = np.asarray(vec, dtype=np.float32)
            self._vectors.build(ids, np.stack([cached[cid] for cid in ids]))
            n_new = len(missing_idx)
        else:
            n_new = 0
        self._save_fingerprint()
        self._index = HybridIndex(self._bm25, self._vectors, self._embedder, metas)
        return {"chunks": len(ids), "embedded_new": n_new,
                "full_reembed": full_rebuild}

    # -- embedding 指纹（机制守卫：任何后端/模型变更都不得留下混搭向量库） ------

    def _fingerprint_path(self) -> Path:
        return self.cfg.data_dir / "vectors" / "fingerprint.json"

    def _embed_key(self) -> str:
        """向量身份键：backend:模型名:dim（同后端换模型同维度也必须能识别）。"""
        ident = getattr(self._embedder, "identity", None)
        return ident or f"{self.cfg.embed_backend}:{self._embedder.dim}"

    def _fingerprint_key(self) -> str:
        try:
            import json as _j
            return _j.loads(self._fingerprint_path().read_text(
                encoding="utf-8")).get("embed_key", "")
        except Exception:  # noqa: BLE001
            return ""

    def _save_fingerprint(self) -> None:
        import json as _j
        self._fingerprint_path().write_text(
            _j.dumps({"embed_key": self._embed_key()}), encoding="utf-8")

    def _ensure_index(self) -> HybridIndex:
        if self._index is None:
            self._bm25._open()   # 触发 corpus 分词版本检查
            if self._bm25.stale:
                # 旧版 corpus 无 texts 无法自愈重分词 → 全量重建（M0）
                self.rebuild_index()
                return self._index
            # 向量指纹守卫（机制洞修补）：换 embedding 后端/模型后即使不上传
            # 任何文档、直接提问——首查前也必须全量重嵌，绝不混搭
            if self._fingerprint_key() != self._embed_key():
                self.rebuild_index()
                return self._index
            metas = self.meta.all_chunk_metas()
            self._index = HybridIndex(self._bm25, self._vectors, self._embedder, metas)
        return self._index

    # -- 摄取 / 文档库管理 ----------------------------------------------------

    def ingest(self, path: str | Path, *, doc_id: str | None = None,
               version: str | None = None, title: str | None = None,
               effective_date: str | None = None,
               filename_stem: str | None = None,
               use_vision: bool = True) -> dict[str, Any]:
        report = ingest_file(
            Path(path), self.meta,
            doc_id=doc_id, version=version, title=title,
            effective_date=effective_date, filename_stem=filename_stem,
            use_vision=use_vision,
        )
        if report.get("status") in ("ok", "noop"):
            self.rebuild_index()
        return report

    def vision_check(self, force: bool = False) -> dict[str, Any]:
        """视觉能力验证（探测为准——MCP 共享键未必有视觉能力，无 GUI 可提示）。"""
        from ..llm.vision import probe_vision
        return probe_vision(force=force)

    def requeue_charts(self, doc_key: str, use_vision: bool) -> dict[str, Any]:
        """重跑文档图表转写（用户改选视觉策略 / 补配视觉 Key 后重试）。"""
        from ..ingest.enrichment import get_worker
        doc = self.meta.get_document(doc_key)
        if doc is None:
            return {"status": "rejected", "reason": f"doc_key 不存在: {doc_key}"}
        ids = sorted({r["path_key"].split("/")[-1]
                      for r in self.meta.chunks_of_doc(doc_key)
                      if r.get("kind") == "chart" and r.get("path_key")})
        if not ids:
            return {"status": "rejected", "reason": "该文档无图表块"}
        meta0 = doc.get("meta_json") or "{}"
        import json as _j
        try:
            src = _j.loads(meta0).get("source_file") or ""
        except Exception:  # noqa: BLE001
            src = ""
        if not src:
            return {"status": "rejected", "reason": "文档缺少源文件路径，无法重跑转写"}
        worker = get_worker(self.meta)
        jid = worker.submit(doc_key, doc.get("doc_id"), doc.get("version"), src,
                            "chart", chart_local_ids=ids, use_vision=use_vision,
                            priority=0)
        return {"status": "ok", "job_id": jid, "n_charts": len(ids),
                "use_vision": use_vision}

    def delete_document(self, doc_key: str) -> dict[str, Any]:
        """删文档 → 重建索引（向量/BM25 同步剔除）。"""
        if self.meta.get_document(doc_key) is None:
            return {"status": "rejected", "reason": f"doc_key 不存在: {doc_key}"}
        n_chunks = self.meta.delete_document(doc_key)
        self.rebuild_index()
        return {"status": "ok", "doc_key": doc_key, "deleted_chunks": n_chunks}

    def list_chunks(self, doc_key: str | None = None, q: str | None = None,
                    limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        """分块清单（GUI 勾选浏览）。"""
        return self.meta.list_chunks_filtered(doc_key=doc_key, q=q, limit=limit, offset=offset)

    # -- 富化任务（按需深转） ---------------------------------------------------

    def rich_job(self, job_id: str) -> dict[str, Any] | None:
        return self.meta.job_get(job_id)

    def on_demand_enrich(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """被引中出现未转换公式/图表 → 当场提交转换任务，返回 job 描述。

        串行执行（低配设备友好）；提交即刻排队，前端轮询进度。
        """
        from ..ingest.enrichment import get_worker
        pending_formula_pages: dict[str, list[int]] = {}
        pending_charts: dict[str, list[str]] = {}
        for cid in chunk_ids:
            row = self.meta.get_chunk(cid)
            if not row:
                continue
            import json as _j
            try:
                extra = _j.loads(row.get("meta_json") or "{}")
            except Exception:  # noqa: BLE001
                extra = {}
            if not extra.get("pending"):
                continue
            doc_key = row["doc_key"]
            doc_id, _, version = doc_key.partition("@")
            if row["kind"] == "formula" and row.get("page"):
                pending_formula_pages.setdefault(doc_key, []).append(row["page"])
            elif row["kind"] == "chart":
                pending_charts.setdefault(doc_key, []).append(
                    extra.get("local_id") or row["path_key"].split("/")[-1])

        worker = get_worker(self.meta, on_index_refresh=self.rebuild_index)
        jobs = []
        docs = {d["doc_key"]: d for d in self.meta.list_documents()}
        for doc_key, pages in pending_formula_pages.items():
            d = docs.get(doc_key)
            src = (d or {}).get("meta_json", "{}")
            import json as _j2
            try:
                src_path = _j2.loads(src).get("source_file", "")
            except Exception:  # noqa: BLE001
                src_path = ""
            if not src_path:
                continue
            doc_id, _, version = doc_key.partition("@")
            jid = worker.submit(doc_key, doc_id, version, src_path, "formula",
                                pages=sorted(set(pages)), priority=0)
            jobs.append({"job_id": jid, "kind": "formula",
                         "chunk_ids": [c for c in chunk_ids], "doc_key": doc_key})
        for doc_key, ids in pending_charts.items():
            d = docs.get(doc_key)
            import json as _j3
            try:
                src_path = _j3.loads((d or {}).get("meta_json", "{}")).get("source_file", "")
            except Exception:  # noqa: BLE001
                src_path = ""
            if not src_path:
                continue
            doc_id, _, version = doc_key.partition("@")
            jid = worker.submit(doc_key, doc_id, version, src_path, "chart",
                                chart_local_ids=sorted(set(ids)), priority=0)
            jobs.append({"job_id": jid, "kind": "chart",
                         "chunk_ids": [c for c in chunk_ids], "doc_key": doc_key})
        return jobs

    def on_missing_rich(self, pairs: list[tuple[str, int]]) -> list[dict[str, Any]]:
        """旧文档补建：证据页含乱码公式但无富化块 → 建占位 + 提交转换任务。"""
        from ..ingest.enrichment import get_worker
        from ..models.ids import make_rich_chunk_id
        from ..models.document import BlockKind
        import json as _j
        worker = get_worker(self.meta, on_index_refresh=self.rebuild_index)
        jobs: list[dict[str, Any]] = []
        by_doc: dict[str, list[int]] = {}
        for doc_key, page in pairs:
            by_doc.setdefault(doc_key, []).append(page)
        for doc_key, pages in by_doc.items():
            d = self.meta.get_document(doc_key)
            if not d:
                continue
            try:
                src = _j.loads(d.get("meta_json") or "{}").get("source_file", "")
            except Exception:  # noqa: BLE001
                src = ""
            if not src or not Path(src).exists():
                continue
            doc_id, _, version = doc_key.partition("@")
            for pg in sorted(set(pages)):
                local_id = f"f_{pg:03d}_01"
                cid = make_rich_chunk_id(doc_id, version, "formula", local_id)
                if self.meta.get_chunk(cid):
                    continue
                img = None
                try:
                    import pymupdf
                    from ..ingest.formulas import crop_formula_image
                    doc = pymupdf.open(src)
                    img_dir = Path(src).parent / "rich_assets" / "formulas"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img = crop_formula_image(doc, pg, "计算公式",
                                             img_dir / f"{local_id}.png")
                    doc.close()
                except Exception:  # noqa: BLE001
                    pass
                body = f"（第{pg}页公式，原文见附图，文本转写进行中）"
                self.meta.upsert_chunk(
                    chunk_id=cid, doc_key=doc_key, path_key=f"公式/{local_id}",
                    breadcrumb=f"第{pg}页 公式{local_id}",
                    text=body, text_norm=body, kind=BlockKind.FORMULA.value, page=pg,
                    meta={"latex": "", "image": img, "pending": True,
                          "confidence": "low", "transcribed_by": "pending"})
            self.meta.commit()
            jid = worker.submit(doc_key, doc_id, version, src, "formula",
                                pages=sorted(set(pages)), priority=0)
            jobs.append({"job_id": jid, "kind": "formula",
                         "chunk_ids": [], "doc_key": doc_key})
        return jobs

    # -- 人工复核队列 ----------------------------------------------------------

    def create_review_task(self, env: AnswerEnvelope, query: str) -> str:
        """把一次回答送入人工复核队列，返回 task_id。"""
        task_id = f"rev_{env.request_id}"
        self.meta.create_review_task(
            task_id=task_id, query=query, as_of=env.as_of,
            draft_answer=env.answer,
            citations=[{"chunk_id": c.chunk_id, "quote": c.quote,
                        "clause_path": c.clause_path} for c in env.citations],
            envelope=env.to_dict(),
        )
        return task_id

    def list_review_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.meta.list_review_tasks(status=status)

    def decide_review_task(self, task_id: str, approve: bool,
                           note: str = "", final_answer: str = "") -> dict[str, Any]:
        task = self.meta.get_review_task(task_id)
        if task is None:
            return {"status": "rejected", "reason": f"task 不存在: {task_id}"}
        if task["status"] != "pending":
            return {"status": "rejected", "reason": f"task 已决: {task['status']}"}
        new_status = "approved" if approve else "rejected"
        self.meta.decide_review_task(task_id, new_status, note,
                                     final_answer or task["draft_answer"])
        return {"status": "ok", "task_id": task_id, "decision": new_status}

    # -- 查询 ---------------------------------------------------------------

    def ask(self, req: QueryRequest) -> AnswerEnvelope:
        """同步问答。多问题自动拆分逐题作答并合并（含逐步结果在 meta.per_question）。"""
        return self._ask_impl(req)

    def ask_stream(self, req: QueryRequest, plan: Any = None) -> Iterator[dict]:
        """流式问答事件迭代器（SSE 源）。

        事件：plan / q_start / phase / draft_delta / q_done /
              synthesis_start / synthesis_delta / synthesis_done / done / error。
        多问题逐题完成逐题推送，不等全部完成；plan 非空 = execute 已确认方案。
        """
        q: "queue.Queue[dict | None]" = queue.Queue()

        def emit(ev: dict) -> None:
            q.put(ev)

        def worker() -> None:
            try:
                env = self._ask_impl(req, emit=emit, plan=plan)
                emit({"type": "done", "envelope": env.to_dict()})
            except Exception as e:  # noqa: BLE001 - 流上也要报错，不能静默断流
                emit({"type": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"})
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            ev = q.get()
            if ev is None:
                break
            yield ev

    def build_plan(self, req: QueryRequest) -> dict[str, Any]:
        """生成检索方案（WS-B /v1/plan）：拆分 + 分类 + 指令 + 扩展联想。"""
        import datetime as _dt
        from ..query.decompose import decompose_questions
        from ..query.search_plan import generate_plan, plan_need_confirm

        questions = decompose_questions(req.query) or [req.query]
        plan = generate_plan(
            self.llm, questions,
            as_of=req.as_of or _dt.date.today().isoformat(),
            lib_hints=self._lib_hints(req), mode=req.mode or "quote",
        )
        return {"need_confirm": plan_need_confirm(plan), "plan": plan.to_dict(),
                "questions": questions}

    def _lib_hints(self, req: QueryRequest) -> str:
        """库路由提示（WS-E）：范围内库名拼入 plan 上下文。"""
        ids = (req.filters.library_ids or [])[:4]
        if not ids:
            return ""
        names = []
        for lib in self.meta.list_libraries():
            if lib["id"] in ids:
                names.append(lib["name"])
        return "｜".join(names)

    def _ask_impl(self, req: QueryRequest, emit: Any = None,
                  plan: Any = None) -> AnswerEnvelope:
        """编排：方案 → 阶段1 检索项逐题 → 阶段2 推理项综合（WS-B/C 两阶段）。"""
        import datetime as _dt
        import re as _re
        from ..query.decompose import decompose_questions
        from ..query.search_plan import generate_plan
        from ..query.synthesis import run_synthesis

        require_review = bool((req.meta or {}).get("require_review"))
        if plan is None:
            # 自动路径（MCP / 闸门关 / 旧接口）：plan 自动全选执行，方案落 envelope.meta
            questions = decompose_questions(req.query) or [req.query]
            plan = generate_plan(
                self.llm, questions,
                as_of=req.as_of or _dt.date.today().isoformat(),
                lib_hints=self._lib_hints(req), mode=req.mode or "quote",
            )
        questions = [it.question for it in plan.items] or [req.query]

        if emit is not None:
            emit({"type": "plan", "questions": questions, "total": len(questions),
                  "search_plan": plan.to_dict()})

        base_rid = req.request_id or f"req_{uuid.uuid4().hex[:12]}"

        # --- 工作清单：retrieve 项 + reason 项的隐式事实子问（G5）---
        work: list[tuple[str, Any]] = []
        for it in plan.items:
            if it.kind == "retrieve":
                work.append((it.qid, it))
        fact_qids: dict[str, list[str]] = {}
        for it in plan.items:
            if it.kind != "reason":
                continue
            fids: list[str] = []
            for j, fact in enumerate(it.implicit_facts or [], 1):
                fqid = f"{it.qid}f{j}"
                from ..query.search_plan import PlanItem as _PI
                work.append((fqid, _PI(qid=fqid, kind="retrieve", question=fact,
                                       specificity=it.specificity)))
                fids.append(fqid)
            fact_qids[it.qid] = fids
        q_text = {qid: it.question for qid, it in work}

        # --- 阶段 1：检索项逐题（流式，完成一题推一题）---
        # per_question 题面序归位（#13）：按 plan 项原序收集（reason 项综合后补入），
        # 隐式事实子问是内部依据不进对账单——per_question 与题面问题一一对应
        pos_of = {it.qid: i for i, it in enumerate(plan.items)}
        per_items: list[tuple[int, int, str, AnswerEnvelope]] = []
        envs_by_qid: dict[str, AnswerEnvelope] = {}
        n_work = len(work)
        for idx, (qid, it) in enumerate(work, 1):
            if not it.selected:
                continue
            if emit is not None:
                emit({"type": "q_start", "index": idx, "total": n_work,
                      "question": it.question, "qid": qid,
                      "specificity": it.specificity})
            vq = [it.variants[k].query for k in it.variants_selected
                  if 0 <= k < len(it.variants)]
            for k in it.variants_selected:
                if 0 <= k < len(it.variants):
                    v = it.variants[k]
                    vq += v.terms + v.clause_refs + v.numeric_anchors
            # G1：仅"方案已规划且未降级"的项注入指令跳过 B2；
            # 隐式事实子问/plan 降级项走原 B2 重写（保持召回质量）
            use_directive = (plan.degraded is None
                             and not it.qid[-1:].isalpha())  # 事实子问 qid 以字母结尾
            sub = QueryRequest(
                query=it.question, as_of=req.as_of, filters=req.filters,
                mode=req.mode,
                top_k=(14 if it.specificity == "broad" else req.top_k),
                request_id=f"{base_rid}_{qid}",
                meta={**(req.meta or {}), "_parent_rid": base_rid},
                pre_directive=(it.directive_dict() if use_directive else None),
                variant_queries=vq,
            )
            sub_emit = (lambda ev, _i=idx: emit({**ev, "index": _i})) if emit else None
            env = self._ask_one(sub, emit=sub_emit, require_review=require_review)
            if qid in pos_of:   # 题面主问进对账单；隐式事实子问不进（内部依据）
                per_items.append((pos_of[qid], 0, it.question, env))
            envs_by_qid[qid] = env
            if emit is not None:
                emit({"type": "q_done", "index": idx, "total": n_work,
                      "question": it.question, "qid": qid,
                      "envelope": env.to_dict()})

        # --- 阶段 2：推理项综合（依赖的检索答案齐后；文档锚定推导）---
        syntheses: list[dict] = []
        for it in plan.items:
            if it.kind != "reason" or not it.selected:
                continue
            deps = list(it.depends_on or []) + fact_qids.get(it.qid, [])
            basis_facts = []
            basis_ids: set[str] = set()
            for dq in deps:
                e = envs_by_qid.get(dq)
                if e is None or e.status != AnswerStatus.ANSWERED:
                    continue      # 拒答子答案禁入 basis（G4）
                basis_facts.append({
                    "qid": dq,
                    "question": q_text.get(dq, dq),
                    "answer": _re.sub(r"\[[0-9a-fA-F]{16}\]", "", e.answer),
                    "evidence": [c.quote for c in e.citations if c.verified][:6],
                })
                basis_ids |= {c.chunk_id for c in e.citations}
            if emit is not None:
                emit({"type": "synthesis_start", "question": it.question,
                      "qid": it.qid, "n_basis": len(basis_facts)})

            def _s_delta(txt: str, _qid=it.qid) -> None:
                if emit is not None:
                    emit({"type": "synthesis_delta", "text": txt, "qid": _qid})

            result = run_synthesis(self.llm, it.question, basis_facts,
                                   on_delta=_s_delta)
            result["qid"] = it.qid
            result["question"] = it.question
            syntheses.append(result)
            if emit is not None:
                emit({"type": "synthesis_done", "qid": it.qid,
                      "question": it.question, "synthesis": result})
            # reason 项进对账单（#13）：此前不进 per → 整题从对账单消失（MAB1 售电公司题实测）
            syn_ok = result.get("status") in ("ok", "partial")
            # F5：闭包校验剔除分析句 ≠ 整题拒答——降级部分作答（占位声明），
            # 事实基础（depends_on 子答）仍在对账单；仅零可给内容才拒
            syn_text = (result.get("text") or "").strip()
            if syn_ok:
                syn_status = AnswerStatus.ANSWERED
                syn_answer = result.get("text") or ""
                syn_refusal = None
            elif deps and any(e is not None and e.status == AnswerStatus.ANSWERED
                              for e in (envs_by_qid.get(dq) for dq in deps)):
                syn_status = AnswerStatus.PARTIAL
                syn_answer = "（分析推理未通过校验已省略）"
                syn_refusal = None
            else:
                syn_status = AnswerStatus.REFUSED
                syn_answer = ""
                syn_refusal = Refusal(
                    reason="synthesis_rejected",
                    detail="分析推理未通过校验，已省略。")
            syn_env = AnswerEnvelope(
                request_id=f"{base_rid}_{it.qid}",
                status=syn_status,
                answer=syn_answer,
                confidence=0.7,
                as_of=req.as_of or "",
                corpus_version=self._corpus_version(),
                refusal=syn_refusal,
                synthesis=result,
                meta={"kind": "reason"},
            )
            per_items.append((pos_of.get(it.qid, 0), 0, it.question, syn_env))

        # 题面序归位：按 plan 项原序输出 per_question / 对账单编号
        per_items.sort(key=lambda x: (x[0], x[1]))
        per = [(q, e) for _, _, q, e in per_items]
        merged = merge_envelopes(per, request_id=base_rid,
                                 corpus_version=self._corpus_version())
        merged.meta["search_plan"] = plan.to_dict()
        if syntheses:
            overall = ("ok" if all(s["status"] == "ok" for s in syntheses)
                       else "partial" if any(s["status"] != "rejected" for s in syntheses)
                       else "rejected")
            merged.synthesis = {
                "status": overall,
                "text": "\n\n".join(
                    f"【分析】{s['question']}\n{s.get('text') or '（分析未通过校验已省略）'}"
                    for s in syntheses),
                "parts": syntheses,
            }
        try:
            from .runlog import log_event
            log_event("ask_batch", request_id=base_rid,
                      n_questions=len(questions),
                      n_work=n_work,
                      n_answered=sum(1 for _, e in per
                                     if e.status == AnswerStatus.ANSWERED),
                      n_synthesis=len(syntheses),
                      plan_degraded=plan.degraded,
                      query=req.query[:160])
        except Exception:  # noqa: BLE001
            pass
        return merged

    def _ask_one(self, req: QueryRequest, emit: Any = None,
                 require_review: bool = False) -> AnswerEnvelope:
        # 范围过滤：勾选的库/chunk_ids/doc_keys → 允许参与检索的 chunk 集合
        scope: set[str] | None = None
        if req.filters.has_scope:
            doc_keys = req.filters.doc_keys
            if req.filters.library_ids:
                lib_keys = self.meta.doc_keys_in_libraries(req.filters.library_ids)
                doc_keys = list(dict.fromkeys(list(doc_keys or []) + lib_keys))
                if not doc_keys:
                    doc_keys = [""]   # 空库 → 空范围（走 empty_scope 拒答）
            scope = self.meta.filter_chunk_ids(req.filters.chunk_ids, doc_keys)
            if not scope:
                trace_id = req.request_id or "req_scope_empty"
                return AnswerEnvelope(
                    request_id=trace_id,
                    status=AnswerStatus.REFUSED,
                    as_of=req.as_of or "",
                    corpus_version=self._corpus_version(),
                    refusal=Refusal(
                        reason="empty_scope",
                        detail="勾选范围内的分块不存在（可能已被删除），请重新勾选。",
                    ),
                    trace_id=trace_id,
                )
        # 时效过滤（生效时点真正参与检索范围，不再只是记录字段）
        import datetime as _dt
        from ..ingest.normalize import canonical_date
        as_of_raw = req.as_of or _dt.date.today().isoformat()
        as_of = canonical_date(as_of_raw) or as_of_raw
        temporal = self.meta.temporal_chunk_ids(as_of)
        if temporal is not None:
            scope = temporal if scope is None else (scope & temporal)

        deps = PipelineDeps(
            index=self._ensure_index(),
            llm=self.llm,
            registry=self._chunk_registry(),
            meta_lookup=self.meta.all_chunk_metas(),
            corpus_version=self._corpus_version(),
            scope_chunk_ids=scope,
            on_pending_rich=self.on_demand_enrich,
            on_missing_rich=self.on_missing_rich,
            live_chunk=lambda cid: self.meta.get_chunk(cid),
            live_page_rich=lambda dk, pg: self.meta.rich_blocks_of_page(dk, pg),
            live_family=self.live_family,
        )
        pipeline = QueryPipeline(deps, trace_root=str(self.cfg.trace_dir))
        env = pipeline.run(req, emit=emit)
        if require_review and env.status == AnswerStatus.ANSWERED:
            try:
                env.meta["review_task_id"] = self.create_review_task(env, req.query)
            except Exception:  # noqa: BLE001
                pass
        try:
            from .runlog import log_event
            log_event("ask", request_id=env.request_id, status=env.status.value,
                      query=req.query[:120], mode=req.mode, as_of=as_of,
                      refusal=(env.refusal.reason if env.refusal else None),
                      n_citations=len(env.citations), n_formulas=len(env.formulas),
                      rich_jobs=[j.get("job_id") for j in env.rich_jobs])
        except Exception:  # noqa: BLE001
            pass
        self.meta.write_audit(
            request_id=env.request_id,
            channel="core", tool="ask",
            request_hash=hashlib.sha1(req.query.encode()).hexdigest()[:16],
            status=env.status.value,
            trace_ref=env.trace_id,
            corpus_version=env.corpus_version,
            usage={
                "input_tokens": env.usage.input_tokens,
                "output_tokens": env.usage.output_tokens,
                "cost_usd": env.usage.cost_usd,
            },
            verification={
                "quotes_ok": env.verification.quotes_ok,
                "numbers_ok": env.verification.numbers_ok,
                "review_ok": env.verification.review_ok,
            },
            query_text=req.query,
            answer_preview=(env.answer
                            or (env.refusal.detail if env.refusal else ""))[:200],
            parent_request_id=(req.meta or {}).get("_parent_rid", ""),
        )
        return env

    def _chunk_registry(self) -> dict[str, str]:
        """chunk_id -> 原文（引用校验用）。数百份文档规模全量驻留可接受。"""
        metas = self.meta.all_chunk_metas()
        return {cid: m["text"] for cid, m in metas.items()}

    def _corpus_version(self) -> str:
        docs = self.meta.list_documents()
        if not docs:
            return "empty"
        return hashlib.sha1(
            "|".join(sorted(d["doc_key"] for d in docs)).encode()
        ).hexdigest()[:12]

    # -- 渐进披露 / 取原文 ---------------------------------------------------

    def live_family(self, chunk_id: str) -> list[dict[str, Any]]:
        """标题型条文块的实质子条款块（证据家族扩展用，#9）。

        条文标题（"第七条 发电企业基本条件"）与（一）（二）（三）实质条款分块存储，
        检索只命中标题块时证据空心——按 path_key 前缀补子块。
        """
        row = self.meta.get_chunk(chunk_id)
        if not row:
            return []
        pk = row.get("path_key") or ""
        if not pk:
            return []
        kids = [r for r in self.meta.chunks_of_doc(row["doc_key"])
                if (r.get("path_key") or "").startswith(pk + "/")
                and r.get("kind") in ("item", "paragraph", "text", "table")
                and len(r.get("text") or "") > 15]
        kids.sort(key=lambda r: (r.get("path_key") or ""))
        return [dict(r) for r in kids[:6]]

    def get_chunk(self, chunk_id: str, include_neighbors: int = 0) -> dict[str, Any] | None:
        row = self.meta.get_chunk(chunk_id)
        if row is None:
            return None
        import json as _json
        try:
            extra = _json.loads(row.get("meta_json") or "{}")
        except Exception:  # noqa: BLE001
            extra = {}
        out = {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "clause_path": row["breadcrumb"],
            "doc_key": row["doc_key"],
            "page": row["page"],
            "kind": row["kind"],
            "meta": extra,
            # 展示块下标（拼合后口径）——阅读器跳转定位用；增量字段
            "block_index": reader_view.block_index_of(
                self.meta.chunks_of_doc(row["doc_key"]), chunk_id),
        }
        if include_neighbors > 0:
            siblings = self.meta.chunks_of_doc(row["doc_key"])
            idx = next((i for i, s in enumerate(siblings) if s["chunk_id"] == chunk_id), None)
            if idx is not None:
                lo = max(0, idx - include_neighbors)
                hi = min(len(siblings), idx + include_neighbors + 1)
                out["neighbors"] = [
                    {"chunk_id": s["chunk_id"], "text": s["text"][:200]}
                    for s in siblings[lo:hi] if s["chunk_id"] != chunk_id
                ]
        return out

    def doc_index(self, doc_id: str, version: str | None = None) -> dict[str, Any]:
        docs = [d for d in self.meta.list_documents() if d["doc_id"] == doc_id]
        if version:
            docs = [d for d in docs if d["version"] == version]
        if not docs:
            return {"doc_id": doc_id, "versions": []}
        out = []
        for d in docs:
            chunks = self.meta.chunks_of_doc(d["doc_key"])
            out.append({
                "doc_key": d["doc_key"],
                "title": d["title"],
                "effective_date": d["effective_date"],
                "status": d["status"],
                "clauses": [
                    {"chunk_id": c["chunk_id"], "path": c["breadcrumb"],
                     "summary": c["summary"], "kind": c["kind"]}
                    for c in chunks
                ],
            })
        return {"doc_id": doc_id, "versions": out}

    def auto_archive(self, scope: str = "unarchived",
                     doc_keys: list[str] | None = None) -> dict[str, Any]:
        """AI 自动归档：按标题+主要内容把文档分入库（用户决策：库+路由提示）。"""
        from ..llm.base import ChatMessage, llm_call_json

        docs = self.meta.list_documents()
        want: list[dict] = []
        if doc_keys:
            want = [d for d in docs if d["doc_key"] in set(doc_keys)]
        elif scope == "all":
            want = docs
        else:  # unarchived
            want = [d for d in docs if not self.meta.libraries_of_doc(d["doc_key"])]
        if not want:
            return {"status": "ok", "assigned": 0, "detail": "无待归档文档"}

        libs = self.meta.list_libraries()
        lib_names = [l["name"] for l in libs]
        name_to_id = {l["name"]: l["id"] for l in libs}

        assignments: list[dict] = []
        for start in range(0, len(want), 8):
            batch = want[start:start + 8]
            lines = []
            for i, d in enumerate(batch, 1):
                chunks = self.meta.chunks_of_doc(d["doc_key"])
                lead = " ".join(c["text"] for c in chunks[:3])[:200]
                lines.append(f"{i}. doc_key={d['doc_key']} | 标题={d['title']} | 摘要={lead}")
            prompt = (
                "现有文档库：" + ("、".join(lib_names) if lib_names else "（无）") + "\n"
                "文档清单：\n" + "\n".join(lines) + "\n\n"
                "把每个文档归入最合适的库（可沿用现有库名，或起新库名 2-8 字按领域命名）。"
                '只输出 JSON：{"assignments": [{"doc_key": "...", "library": "库名"}]}'
            )
            r = llm_call_json(self.llm, [
                ChatMessage(role="system", content="你是文档归档员。只输出 JSON。", cache=True),
                ChatMessage(role="user", content=prompt),
            ], max_tokens=1024, effort="low")
            parsed = r.parsed if isinstance(r.parsed, dict) else {}
            if not isinstance(parsed, dict):
                parsed = {}
            for a in parsed.get("assignments") or []:
                if a.get("doc_key") and a.get("library"):
                    assignments.append(a)

        n = 0
        created = []
        for a in assignments:
            name = str(a["library"]).strip()[:24]
            dk = str(a["doc_key"])
            if self.meta.get_document(dk) is None or not name:
                continue
            if name not in name_to_id:
                lib_id = f"lib_{uuid.uuid4().hex[:8]}"
                self.meta.upsert_library(lib_id, name, auto=True)
                name_to_id[name] = lib_id
                created.append(name)
            self.meta.assign_library(dk, name_to_id[name], source="auto")
            n += 1
        return {"status": "ok", "assigned": n, "libraries_created": created,
                "candidates": len(want)}

    def fulltext(self, doc_key: str, offset: int = 0, limit: int = 500) -> dict:
        """重建结构化全文（WS-D 阅读器数据源）。blocks 按页序+path_key 序。

        展示层条款拼合：同一条（含款/项/续块）先拼成一个展示块再分页——
        total/offset/has_more 均以展示块为单位，条款永不跨页撕裂。
        """
        doc = self.meta.get_document(doc_key)
        if doc is None:
            return {"error": f"doc_key 不存在: {doc_key}"}
        # #10 展示侧 breadcrumb 清洗：句中碎片段不进结构树/块标签（存储不动）
        from ..models.clause_norm import clean_breadcrumb
        rows = reader_view.sort_rows(self.meta.chunks_of_doc(doc_key))
        for r in rows:
            r["breadcrumb"] = clean_breadcrumb(r.get("breadcrumb") or "")
        blocks_all = reader_view.display_blocks(rows)
        total = len(blocks_all)
        limit = max(1, min(limit, 1000))
        blocks = blocks_all[offset:offset + limit]

        # 结构树：breadcrumb 层级折叠（章/节/条），一次全量返回；
        # 叶 chunk_id 映射到拼合块主锚点；节点绑定来源物理页区间 [page, page_end]
        # ——树与 PDF 页互跳/高亮的索引（研报阅读走原 PDF 后的定位锚）
        primary = reader_view.chunk_to_primary(rows)
        tree: list[dict] = []
        nodes: dict[str, dict] = {}
        for r in rows:
            # 富化块/表块不进结构树——树是标题/条款结构索引；"第N页 公式f_xxx"
            # 类面包屑曾劈成"第N页"垃圾节点（研报实测）。富化跳转走物理页码。
            if r.get("kind") in ("formula", "chart", "table"):
                continue
            parts = [p for p in (r.get("breadcrumb") or "").split(" ") if p]
            prefix = ""
            for lvl, part in enumerate(parts, 1):
                key = f"{prefix}/{part}"
                prefix = key
                node = nodes.get(key)
                if node is None:
                    node = {"id": key, "label": part, "level": min(lvl, 4),
                            "chunk_id": (primary.get(r["chunk_id"], r["chunk_id"])
                                         if lvl == len(parts) else None),
                            "page": r["page"], "page_end": r["page"]}
                    nodes[key] = node
                    tree.append(node)
                else:
                    if lvl == len(parts) and node.get("chunk_id") is None:
                        node["chunk_id"] = primary.get(r["chunk_id"], r["chunk_id"])
                pg = r.get("page")
                if pg:
                    if not node.get("page"):
                        node["page"] = pg
                    else:
                        node["page"] = min(node["page"], pg)
                    node["page_end"] = max(node.get("page_end") or pg, pg)
        return {
            "doc_key": doc_key, "title": doc["title"], "version": doc["version"],
            "doc_group": doc.get("doc_group") or doc["doc_id"],
            "is_latest": bool(doc.get("is_latest", 1)),
            "total_blocks": total, "offset": offset, "limit": limit,
            "has_more": offset + limit < total,
            "tree": tree, "blocks": blocks,
        }

    def history(self, limit: int = 20, offset: int = 0,
                kind: str = "all", q: str = "") -> dict:
        """检索历史列表（WS-F）：audit_log 时间线（批量按 parent 聚合）。"""
        rows = self.meta.list_audit(limit=limit, offset=offset, kind=kind, q=q)
        return {"total": self.meta.count_audit(kind=kind, q=q),
                "items": rows, "offset": offset, "limit": limit}

    def history_detail(self, request_id: str) -> dict:
        """单次问答详情：audit 行 + trace B9_output 还原（无新持久化）。"""
        row = self.meta.get_audit(request_id)
        if row is None:
            return {"error": f"request_id 不存在: {request_id}"}
        out = dict(row)
        out["detail_available"] = False
        trace_id = row.get("trace_ref") or ""
        try:
            import glob
            import json as _j
            fs = sorted(glob.glob(str(self.cfg.trace_dir / trace_id / "step_*B9*.json")))
            if fs:
                payload = _j.loads(Path(fs[-1]).read_text(encoding="utf-8"))
                env = payload.get("output") or {}
                out.update({
                    "detail_available": True,
                    "answer": env.get("answer", ""),
                    "citations": env.get("citations", []),
                    "verification": env.get("verification", {}),
                    "degradations": env.get("degradations", []),
                    "synthesis": env.get("synthesis"),
                    "formulas": env.get("formulas", []),
                    "charts": env.get("charts", []),
                    "usage": env.get("usage", {}),
                    "meta": env.get("meta", {}),
                    "refusal": env.get("refusal"),
                    "status": env.get("status", out.get("status")),
                })
        except Exception:  # noqa: BLE001
            pass
        return out

    def search(self, query: str, k: int = 10,
               scope_chunk_ids: list[str] | None = None,
               scope_doc_keys: list[str] | None = None) -> list[dict[str, Any]]:
        """纯检索。范围参数：限定在勾选的块/文档内，缺省全局。"""
        allowed: set[str] | None = None
        if scope_chunk_ids or scope_doc_keys:
            allowed = self.meta.filter_chunk_ids(scope_chunk_ids, scope_doc_keys)
        fetch_k = max(k * 5, 50) if allowed is not None else k
        hits = self._ensure_index().search(query, k=fetch_k)
        if allowed is not None:
            hits = [h for h in hits if h.chunk_id in allowed]
        hits = hits[:k]
        return [
            {"chunk_id": h.chunk_id, "score": h.score, "snippet": h.snippet,
             "doc_title": h.doc_title, "clause_path": h.clause_path}
            for h in hits
        ]

    def verify(self, answer_text: str, citations: list[dict[str, str]]) -> dict[str, Any]:
        """独立复核外部草稿（MCP policy_verify）。"""
        cites = [
            Citation(chunk_id=c["chunk_id"], quote=c.get("quote", ""),
                     doc_title="", doc_version="", clause_path=c.get("clause_path", ""))
            for c in citations
        ]
        registry = self._chunk_registry()
        all_ids = set(registry.keys())
        cite_res = verify_citations(cites, registry, all_ids)
        num_res = verify_numbers(answer_text, [registry[c["chunk_id"]] for c in citations if c["chunk_id"] in registry])
        report = VerifyReport(
            quotes_ok=cite_res.ok,
            numbers_ok=num_res.ok,
            failures=[f"[{f.chunk_id}] {f.reason}" for f in cite_res.failures]
            + [f"[{f.span_text}] {f.reason}" for f in num_res.failures],
        )
        return {
            "quotes_ok": report.quotes_ok,
            "numbers_ok": report.numbers_ok,
            "failures": report.failures,
        }


def merge_envelopes(per: list[tuple[str, AnswerEnvelope]], *,
                    request_id: str, corpus_version: str = "") -> AnswerEnvelope:
    """多子问题 envelope 合并为总 envelope（编号对账单）。

    - answer：逐题编号拼接（拒答题显式标注原因，缺答不可见即事故）；
    - status：全拒=refused / 部分拒=partial / 全答=answered；
    - confidence：已答题置信度均值（拒答不稀释）；
    - citations/formulas/charts/rich_jobs 去重合并；usage 求和；
    - meta.per_question 保留逐题完整 envelope（流式先到先得的原始结果）。
    """
    import datetime as _dt

    lines: list[str] = []
    citations: list[Citation] = []
    seen_cites: set[tuple] = set()
    formulas: list[dict] = []
    seen_formulas: set[str] = set()
    charts: list[dict] = []
    seen_charts: set[str] = set()
    rich_jobs: list[dict] = []
    seen_jobs: set[str] = set()
    degradations: list[str] = []
    failures: list[str] = []
    review_task_ids: list[str] = []
    usage = Usage()
    as_of = ""
    trace_id = request_id

    answered = [e for _, e in per if e.status == AnswerStatus.ANSWERED]
    partials = [e for _, e in per if e.status == AnswerStatus.PARTIAL]
    single = len(per) == 1
    for i, (q, env) in enumerate(per, 1):
        as_of = as_of or env.as_of
        if env.status == AnswerStatus.ANSWERED:
            # 单题不加编号前缀（编号对账单只服务多题）
            lines.append(env.answer if single else f"{i}. {q}\n{env.answer}")
        elif env.status == AnswerStatus.PARTIAL:
            # F5 部分作答：可给部分照登，缺口以占位声明（非"拒答"话术）
            body = env.answer or "（部分可答内容缺失）"
            lines.append(body if single else f"{i}. {q}\n{body}")
        else:
            detail = env.refusal.detail if env.refusal else "未作答"
            reason = env.refusal.reason if env.refusal else "unknown"
            lines.append(detail if single else f"{i}. {q}\n（拒答：{reason}）{detail}")
        for c in env.citations:
            key = (c.chunk_id, c.quote)
            if key not in seen_cites:
                seen_cites.add(key)
                citations.append(c)
        for f in env.formulas:
            if f.get("chunk_id") not in seen_formulas:
                seen_formulas.add(f.get("chunk_id"))
                formulas.append(f)
        for ch in env.charts:
            if ch.get("chunk_id") not in seen_charts:
                seen_charts.add(ch.get("chunk_id"))
                charts.append(ch)
        for j in env.rich_jobs:
            if j.get("job_id") not in seen_jobs:
                seen_jobs.add(j.get("job_id"))
                rich_jobs.append(j)
        degradations += [f"[Q{i}] {d}" for d in env.degradations]
        failures += [f"[Q{i}] {f}" for f in env.verification.failures]
        tid = (env.meta or {}).get("review_task_id")
        if tid:
            review_task_ids.append(tid)
        usage.add(env.usage)

    if not per:
        status = AnswerStatus.REFUSED
    elif answered or partials:
        # F5：全 ANSWERED → answered；含部分作答/部分拒 → partial；全拒才 refused
        status = (AnswerStatus.ANSWERED
                  if len(answered) + len(partials) == len(per) and not partials
                  else AnswerStatus.PARTIAL)
    else:
        status = AnswerStatus.REFUSED

    # 内容级去重：跨子问重复引用的同式/同图公式附图只留一个
    from ..query.pipeline import dedupe_rich_lists
    formulas, charts = dedupe_rich_lists(formulas, charts)
    _scored = answered + partials
    confidence = (sum(e.confidence for e in _scored) / len(_scored)) if _scored else 0.0

    meta: dict[str, Any] = {
        "decomposed": len(per) > 1,
        "questions": [q for q, _ in per],
        "per_question": [e.to_dict() for _, e in per],
        "n_questions": len(per),
        "n_answered": len(answered),
    }
    if review_task_ids:
        meta["review_task_ids"] = review_task_ids
        if len(review_task_ids) == 1:
            meta["review_task_id"] = review_task_ids[0]

    return AnswerEnvelope(
        request_id=request_id,
        status=status,
        answer="\n\n".join(lines),
        citations=citations,
        confidence=confidence,
        as_of=as_of or _dt.date.today().isoformat(),
        corpus_version=corpus_version,
        refusal=None if (answered or partials) else (per[0][1].refusal if per else None),
        verification=VerifyReport(
            quotes_ok=all(e.verification.quotes_ok for _, e in per),
            numbers_ok=all(e.verification.numbers_ok for _, e in per),
            review_ok=None,
            failures=failures,
        ),
        degradations=degradations,
        formulas=formulas,
        charts=charts,
        rich_jobs=rich_jobs,
        trace_id=trace_id,
        usage=usage,
        meta=meta,
    )
