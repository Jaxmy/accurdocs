"""REST adapter（FastAPI）+ 图形界面（/）。

薄层——schema 转换 → CoreService → envelope。GUI 静态页在 static/index.html。
启动：python -m uvicorn policyrag.adapters.rest.app:app --workers 1 --limit-concurrency 4
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ...models.retrieval import DocFilter, QueryRequest
from ...service.core import CoreService

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="policyrag", version="0.2.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """任何未捕获异常 → 结构化 JSON（GUI 可显示），不冒 500 白屏。"""
    return JSONResponse(
        status_code=200,
        content={"status": "rejected",
                 "reason": f"{type(exc).__name__}: {str(exc)[:300]}"},
    )

_svc: CoreService | None = None
API_KEY = os.environ.get("POLICYRAG_API_KEY", "")


def _get_svc() -> CoreService:
    global _svc
    if _svc is None:
        from ...settings import make_llm, service_config
        _svc = CoreService(make_llm(), service_config())
    return _svc


def _auth(x_api_key: str) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


# ---------------- GUI ----------------

@app.get("/")
async def gui():
    # no-store：GUI 频繁迭代，防浏览器缓存旧版（曾致新功能不生效）
    return FileResponse(STATIC_DIR / "index.html",
                        headers={"Cache-Control": "no-store"})


@app.get("/healthz")
async def healthz():
    """健康检查（GUI 右上角状态灯）。"""
    return {"ok": True}


@app.get("/favicon.ico")
async def favicon():
    """浏览器自动请求——显式 204，避免挤占并发额度/刷 404。"""
    return JSONResponse(status_code=204, content=None)


# ---------------- 查询 ----------------

def _build_req(body: dict) -> QueryRequest:
    f = body.get("filters") or {}
    return QueryRequest(
        query=body.get("query", ""),
        as_of=body.get("as_of", ""),
        mode=body.get("mode", "quote"),
        filters=DocFilter(
            doc_keys=f.get("doc_keys"),
            chunk_ids=f.get("chunk_ids"),
            doc_ids=f.get("doc_ids"),
            library_ids=f.get("library_ids"),
        ),
        top_k=int(body.get("top_k", 8)),
        request_id=body.get("request_id") or f"api_{uuid.uuid4().hex[:10]}",
        meta={"require_review": bool(body.get("require_review"))},
    )


def _label_review(out: dict) -> dict:
    """复核任务标注（core 统一创建，adapter 只做展示字段映射）。"""
    meta = out.get("meta") or {}
    rids = meta.get("review_task_ids") or (
        [meta["review_task_id"]] if meta.get("review_task_id") else [])
    if rids:
        out["review_task_id"] = rids[0] if len(rids) == 1 else rids
        if out.get("status") == "answered":
            out["status"] = "pending_review"
    return out


def _present(out: dict) -> dict:
    """WS-G 展示分层：原子 ID 从用户文本隐藏（display/ref 分层 + line_refs）。"""
    from ...service.present import present_for_user
    return _label_review(present_for_user(out))


@app.post("/v1/query")
async def query(body: dict, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    svc = _get_svc()
    req = _build_req(body)
    # 同步重活丢线程池——不许阻塞事件循环（曾致整个 GUI 假死）
    env = await asyncio.to_thread(svc.ask, req)
    return _present(env.to_dict())


@app.post("/v1/query/stream")
def query_stream(body: dict, x_api_key: str = Header(default="")):
    """SSE 流式问答：多问题逐题完成逐题推送，草稿 token 渐进输出。

    事件（data: JSON）：plan / q_start / phase / draft_delta / q_done /
    synthesis_start / synthesis_delta / synthesis_done / done / error。
    """
    _auth(x_api_key)
    svc = _get_svc()
    req = _build_req(body)

    def gen():
        try:
            for ev in svc.ask_stream(req):
                if ev.get("type") == "done":
                    ev["envelope"] = _present(ev["envelope"])
                elif ev.get("type") == "q_done":
                    ev["envelope"] = _present(ev["envelope"])
                yield "data: " + json.dumps(ev, ensure_ascii=False, default=str) + "\n\n"
        except Exception as e:  # noqa: BLE001 - 流上也要报错
            yield "data: " + json.dumps(
                {"type": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"},
                ensure_ascii=False) + "\n\n"
        yield "data: " + json.dumps({"type": "end"}) + "\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


# ---------------- 检索方案（WS-B 两段式：plan → execute） ----------------

@app.post("/v1/plan")
async def make_plan(body: dict, x_api_key: str = Header(default="")):
    """生成检索方案（非流式廉价调用）。need_confirm=true 时 GUI 弹确认面板。"""
    _auth(x_api_key)
    req = _build_req(body)
    return await asyncio.to_thread(_get_svc().build_plan, req)


@app.post("/v1/query/execute")
def query_execute(body: dict, x_api_key: str = Header(default="")):
    """SSE 执行已确认方案（无状态红线：body 携带完整 items + directive）。"""
    _auth(x_api_key)
    svc = _get_svc()
    req = _build_req(body)
    from ...query.search_plan import plan_from_dict
    plan = plan_from_dict(body.get("plan") or {"items": body.get("items") or []})

    def gen():
        try:
            for ev in svc.ask_stream(req, plan=plan):
                if ev.get("type") in ("done", "q_done"):
                    ev["envelope"] = _present(ev["envelope"])
                yield "data: " + json.dumps(ev, ensure_ascii=False, default=str) + "\n\n"
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps(
                {"type": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"},
                ensure_ascii=False) + "\n\n"
        yield "data: " + json.dumps({"type": "end"}) + "\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.post("/v1/search_s")
async def search_scoped(body: dict, x_api_key: str = Header(default="")):
    """范围可限检索（POST 版，支持 chunk_ids/doc_keys 勾选范围）。"""
    _auth(x_api_key)
    return await asyncio.to_thread(
        _get_svc().search, body.get("query", ""), int(body.get("k", 10)),
        body.get("chunk_ids"), body.get("doc_keys"),
    )


@app.get("/v1/search")
async def search(q: str, k: int = 10, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return await asyncio.to_thread(_get_svc().search, q, k)


@app.get("/v1/chunks/{chunk_id}")
async def get_chunk(chunk_id: str, include_neighbors: int = 0,
                    x_api_key: str = Header(default="")):
    _auth(x_api_key)
    out = _get_svc().get_chunk(chunk_id, include_neighbors=include_neighbors)
    if out is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return out


@app.get("/v1/rich/jobs/{job_id}")
async def rich_job(job_id: str, x_api_key: str = Header(default="")):
    """按需深转任务状态（GUI 进度条轮询）。"""
    _auth(x_api_key)
    job = _get_svc().rich_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job["job_id"], "kind": job["kind"],
            "status": job["status"], "progress": job["progress"],
            "note": job["note"], "doc_key": job["doc_key"]}


@app.get("/v1/rich/jobs")
async def rich_jobs_list(doc_key: str | None = None,
                         x_api_key: str = Header(default="")):
    """富化任务列表（文档库行内进度 + 全局倒计时条）。"""
    _auth(x_api_key)
    jobs = _get_svc().meta.job_list(doc_key=doc_key)
    return [{"job_id": j["job_id"], "kind": j["kind"], "status": j["status"],
             "progress": j["progress"], "note": j["note"],
             "doc_key": j["doc_key"], "updated_at": j["updated_at"]}
            for j in jobs]


@app.get("/v1/documents/{doc_key}/fulltext")
async def doc_fulltext(doc_key: str, offset: int = 0, limit: int = 500,
                       x_api_key: str = Header(default="")):
    """重建结构化全文（阅读器数据源）：tree 全量 + blocks 分页。"""
    _auth(x_api_key)
    return await asyncio.to_thread(_get_svc().fulltext, doc_key, offset, limit)


# ---------------- 检索历史（WS-F） ----------------

@app.get("/v1/history")
async def history_list(limit: int = 20, offset: int = 0, kind: str = "all",
                       q: str = "", x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return await asyncio.to_thread(_get_svc().history, limit, offset, kind, q)


@app.get("/v1/history/{request_id}")
async def history_detail(request_id: str, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return await asyncio.to_thread(_get_svc().history_detail, request_id)


@app.get("/v1/chunks/{chunk_id}/image")
async def chunk_image(chunk_id: str, api_key: str = "",
                      x_api_key: str = Header(default="")):
    """公式/图表块的截图（ground truth 通道）。img 标签带不了头 → 兼容 query api_key。"""
    _auth(x_api_key or api_key)
    row = _get_svc().meta.get_chunk(chunk_id)
    if not row:
        raise HTTPException(status_code=404, detail="chunk not found")
    import json as _json
    try:
        extra = _json.loads(row.get("meta_json") or "{}")
    except Exception:  # noqa: BLE001
        extra = {}
    img = extra.get("image")
    if not img or not Path(img).exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(img, media_type="image/png")


@app.get("/v1/documents/{doc_key}/pdf")
async def doc_pdf(doc_key: str, api_key: str = "",
                  x_api_key: str = Header(default="")):
    """原 PDF 回源（原文阅读改走原 PDF 后的视图数据源）。

    iframe 带不了头 → 兼容 query api_key；#page=N 由浏览器 PDF 查看器处理。
    """
    _auth(x_api_key or api_key)
    import json as _json
    doc = _get_svc().meta.get_document(doc_key)
    if not doc:
        raise HTTPException(status_code=404, detail="doc not found")
    try:
        src = _json.loads(doc.get("meta_json") or "{}").get("source_file") or ""
    except Exception:  # noqa: BLE001
        src = ""
    if not src or not Path(src).exists():
        raise HTTPException(status_code=404, detail="PDF 源文件不可用")
    if Path(src).suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="该文档无 PDF 源")
    return FileResponse(src, media_type="application/pdf",
                        headers={"Cache-Control": "private, max-age=60",
                                 "Content-Disposition": "inline"})


@app.get("/v1/chunks_list")
async def chunks_list(doc_key: str | None = None, q: str | None = None,
                      limit: int = 200, offset: int = 0,
                      x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return _get_svc().list_chunks(doc_key=doc_key, q=q, limit=limit, offset=offset)


# ---------------- 文档管理 ----------------

@app.get("/v1/documents")
async def list_documents(x_api_key: str = Header(default="")):
    _auth(x_api_key)
    svc = _get_svc()
    docs = svc.meta.list_documents()
    lib_names = {l["id"]: l["name"] for l in svc.meta.list_libraries()}
    for d in docs:
        d["n_chunks"] = len(svc.meta.chunks_of_doc(d["doc_key"]))
        d["libraries"] = [lib_names.get(x, x)
                          for x in svc.meta.libraries_of_doc(d["doc_key"])]
    return docs


@app.delete("/v1/documents/{doc_key}")
async def delete_document(doc_key: str, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return await asyncio.to_thread(_get_svc().delete_document, doc_key)


@app.get("/v1/documents/{doc_id}/index")
async def doc_index(doc_id: str, version: str | None = None,
                    x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return _get_svc().doc_index(doc_id, version)


@app.get("/v1/documents/{doc_id}/versions")
async def doc_versions(doc_id: str, x_api_key: str = Header(default="")):
    """同身份组多版本列表（WS-E；content_hash 为版本 DIFF 钩子）。"""
    _auth(x_api_key)
    svc = _get_svc()
    rows = [d for d in svc.meta.list_documents()
            if d.get("doc_group") == doc_id or d["doc_id"] == doc_id]
    rows.sort(key=lambda d: (d.get("version_seq") or 1, d.get("created_at") or 0))
    out = []
    for d in rows:
        out.append({
            "doc_key": d["doc_key"], "doc_id": d["doc_id"], "version": d["version"],
            "version_seq": d.get("version_seq"), "is_latest": d.get("is_latest"),
            "title": d["title"], "effective_date": d["effective_date"],
            "status": d["status"], "content_hash": d.get("content_hash"),
            "n_chunks": len(svc.meta.chunks_of_doc(d["doc_key"])),
        })
    return {"doc_id": doc_id, "versions": out}


# ---------------- 分库 ----------------

@app.get("/v1/libraries")
async def list_libraries(x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return _get_svc().meta.list_libraries()


@app.post("/v1/libraries")
async def upsert_library(body: dict, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="缺少库名")
    lib_id = body.get("id") or f"lib_{uuid.uuid4().hex[:8]}"
    _get_svc().meta.upsert_library(lib_id, name, auto=bool(body.get("auto", False)))
    return {"id": lib_id, "name": name}


@app.delete("/v1/libraries/{lib_id}")
async def delete_library(lib_id: str, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    n = _get_svc().meta.delete_library(lib_id)
    return {"id": lib_id, "unassigned_docs": n}


@app.post("/v1/libraries/auto_archive")
async def auto_archive(body: dict, x_api_key: str = Header(default="")):
    """AI 自动归档：按标题+内容把文档分入库。scope: unarchived|all；或指定 doc_keys。"""
    _auth(x_api_key)
    return await asyncio.to_thread(
        _get_svc().auto_archive,
        scope=body.get("scope") or "unarchived",
        doc_keys=body.get("doc_keys"),
    )


@app.post("/v1/libraries/assign")
async def assign_library(body: dict, x_api_key: str = Header(default="")):
    """归档勾选文档：library_name 一键命名归档（同名复用/新建），或 library_id 精确指派。"""
    _auth(x_api_key)
    svc = _get_svc()
    lib_id = body.get("library_id")
    name = (body.get("library_name") or "").strip()[:24]
    if not lib_id and name:
        for l in svc.meta.list_libraries():
            if l["name"] == name:
                lib_id = l["id"]
                break
        if not lib_id:
            lib_id = f"lib_{uuid.uuid4().hex[:8]}"
            svc.meta.upsert_library(lib_id, name, auto=False)
    if not lib_id:
        raise HTTPException(status_code=400, detail="缺少 library_id / library_name")
    for dk in body.get("doc_keys") or []:
        if body.get("remove"):
            svc.meta.unassign_library(dk, lib_id)
        else:
            svc.meta.assign_library(dk, lib_id, source="manual")
    return {"status": "ok", "library_id": lib_id, "library_name": name,
            "n_docs": len(body.get("doc_keys") or [])}


@app.post("/v1/ingest")
async def ingest_path(body: dict, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    path = body.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="缺少 path")
    from ...service.files import ingest_path_ref
    return await asyncio.to_thread(
        ingest_path_ref, _get_svc(), path, doc_id=body.get("doc_id"),
        version=body.get("version"), effective_date=body.get("effective_date"),
        use_vision=body.get("use_vision", True),
    )


@app.post("/v1/vision_check")
async def vision_check(body: dict | None = None, x_api_key: str = Header(default="")):
    """视觉能力验证（探测为准——共享键未必有视觉能力）。force=true 重新探测。"""
    _auth(x_api_key)
    return await asyncio.to_thread(_get_svc().vision_check, bool((body or {}).get("force")))


@app.post("/v1/rich/requeue")
async def rich_requeue(body: dict, x_api_key: str = Header(default="")):
    """重跑文档图表转写（改选视觉策略 / 补配视觉 Key 后重试）。"""
    _auth(x_api_key)
    doc_key = body.get("doc_key")
    if not doc_key:
        raise HTTPException(status_code=400, detail="缺少 doc_key")
    return await asyncio.to_thread(
        _get_svc().requeue_charts, doc_key, bool(body.get("use_vision", True)))


@app.post("/v1/upload")
async def ingest_upload(file: UploadFile = File(...),
                        effective_date: str = Form(default=""),
                        use_vision: str = Form(default="1"),
                        x_api_key: str = Header(default="")):
    _auth(x_api_key)
    from ...service.files import ingest_bytes
    data = await file.read()
    meta = {}
    if effective_date.strip():
        meta["effective_date"] = effective_date.strip()
    meta["use_vision"] = use_vision.strip().lower() not in ("0", "false", "no")
    return await asyncio.to_thread(ingest_bytes, _get_svc(), data,
                                   file.filename or "upload.bin", **meta)


# ---------------- 人工复核 ----------------

@app.get("/v1/review")
async def review_list(status: str | None = None, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return _get_svc().list_review_tasks(status=status)


@app.post("/v1/review/{task_id}/decision")
async def review_decision(task_id: str, body: dict, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return _get_svc().decide_review_task(
        task_id, approve=bool(body.get("approve")),
        note=body.get("note", ""), final_answer=body.get("final_answer", ""),
    )


@app.post("/v1/verify")
async def verify(body: dict, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    return _get_svc().verify(body.get("answer", ""), body.get("citations", []))


# ---------------- 配置 ----------------

@app.get("/v1/config")
async def get_config(x_api_key: str = Header(default="")):
    _auth(x_api_key)
    from ...settings import read_config
    return read_config()


@app.post("/v1/config")
async def save_config(body: dict, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    from ...settings import write_config
    write_config(body)
    global _svc
    _svc = None  # 重载 LLM
    return {"status": "ok"}


@app.post("/v1/config/test")
async def test_config(x_api_key: str = Header(default="")):
    _auth(x_api_key)
    from ...settings import make_llm
    from ...llm.base import ChatMessage
    try:
        llm = make_llm()
        r = llm.complete([ChatMessage(role="user", content="请只回复两个字母：OK")],
                         max_tokens=16, effort="low")
        return {"ok": r.text.strip() != "", "model": r.model, "reply": r.text.strip()[:20]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}
