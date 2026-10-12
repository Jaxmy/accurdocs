r"""MCP adapter（官方 mcp SDK）——薄层，业务逻辑在 CoreService。

stdio 启动（Claude Code / Cursor 本地挂载）：
  start_mcp.bat（项目根）
  （或 python -m policyrag.adapters.mcp.server）

Streamable HTTP：
  start_mcp.bat --http --port 8100   →   http://127.0.0.1:8100/mcp

Agent 集成约定：
- 用户消息里出现文件/文件路径时，调用 policy_ingest（path 或 file_base64）直接入库；
- 文档管理（删/列表/勾选范围）与人工复核均可通过本工具面完成；
- policy_ask 的 scope 参数支持限定在指定文档/分块内检索（缺省全局）。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

try:
    # mcp 2.x：FastMCP 更名 MCPServer
    from mcp.server.mcpserver import MCPServer as _MCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _MCP

from ...models.retrieval import DocFilter, QueryRequest
from ...service.core import CoreService

mcp = _MCP("accurdocs")

_svc: CoreService | None = None


def _get_svc() -> CoreService:
    global _svc
    if _svc is None:
        from ...settings import make_llm, service_config
        _svc = CoreService(make_llm(), service_config())
    return _svc


# ---------------- 问答 / 检索 ----------------

@mcp.tool()
def policy_ask(query: str, as_of: str | None = None, mode: str = "quote",
                     scope_doc_keys: list[str] | None = None,
                     scope_chunk_ids: list[str] | None = None,
                     require_review: bool = False,
                     confirm_plan: bool = False,
                     expose_ids: bool = False) -> dict[str, Any]:
    """政策/金融文档精准问答。返回 AnswerEnvelope（逐字引用 + 校验结果）。

    query: 自然语言问题。可含多个问题（1. 2. 3. 或多个问句）——自动拆分后
        逐题独立检索推理合并输出，meta.per_question 含逐步结果；
        含分析推理要求时自动两阶段（检索答案 + meta.synthesis 综合分析）。
    as_of: 生效时点（YYYY-MM-DD），默认今天。
    mode: "quote"（条款事实）或 "analyze"（综合分析）。
    scope_doc_keys: 可选，限定在这些文档内检索（["doc_id@version"]），缺省全库。
    scope_chunk_ids: 可选，限定在这些分块内检索，缺省全库。
    require_review: true 时答案先进人工复核队列（GUI 复核页处理），返回 task_id。
    confirm_plan: true 时只返回检索方案不执行（供调用方自行确认裁剪）。
    expose_ids: true 时引用带原子编码 chunk_id（默认隐藏——仅校验/复核场景需要）。
    """
    req = QueryRequest(
        query=query, as_of=as_of or "", mode=mode,
        filters=DocFilter(doc_keys=scope_doc_keys, chunk_ids=scope_chunk_ids),
        request_id=f"mcp_{os.urandom(4).hex()}",
        meta={"require_review": bool(require_review)},
    )
    svc = _get_svc()
    if confirm_plan:
        return {"status": "plan_only", **svc.build_plan(req)}
    env = svc.ask(req)   # 自动 plan→全选→执行（meta.search_plan 落档可追溯）
    out = env.to_dict()
    meta = out.get("meta") or {}
    rids = meta.get("review_task_ids") or (
        [meta["review_task_id"]] if meta.get("review_task_id") else [])
    if rids:
        out["review_task_id"] = rids[0] if len(rids) == 1 else rids
        if out.get("status") == "answered":
            out["status"] = "pending_review"
    if not expose_ids:
        from ...service.present import present_for_user
        out = present_for_user(out)
    return out


@mcp.tool()
def policy_search(query: str, k: int = 10,
                        scope_doc_keys: list[str] | None = None,
                        scope_chunk_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """纯检索：返回相关条款片段（不作答）。可限定范围（scope_doc_keys/scope_chunk_ids）。"""
    return _get_svc().search(query, k=k, scope_chunk_ids=scope_chunk_ids,
                             scope_doc_keys=scope_doc_keys)


@mcp.tool()
def policy_get_chunks(chunk_ids: list[str], include_neighbors: int = 0) -> list[dict[str, Any]]:
    """按 chunk_id 取原文全文（渐进式披露）。可带相邻块上下文。"""
    out = []
    for cid in chunk_ids[:50]:
        d = _get_svc().get_chunk(cid, include_neighbors=include_neighbors)
        out.append(d if d else {"chunk_id": cid, "error": "not_found"})
    return out


@mcp.tool()
def policy_doc_index(doc_id: str, version: str | None = None) -> dict[str, Any]:
    """文档分级索引树（L1 导航层）。"""
    return _get_svc().doc_index(doc_id, version)


# ---------------- 文档库管理 ----------------

@mcp.tool()
def policy_list_documents() -> list[dict[str, Any]]:
    """列出文档库全部文档（doc_key/title/版本/生效日期/状态/块数）。"""
    svc = _get_svc()
    docs = svc.meta.list_documents()
    for d in docs:
        d["n_chunks"] = len(svc.meta.chunks_of_doc(d["doc_key"]))
    return docs


@mcp.tool()
def policy_list_chunks(doc_key: str | None = None, q: str | None = None,
                             limit: int = 100) -> list[dict[str, Any]]:
    """浏览分块清单（可按文档/文本过滤），返回 chunk_id + 位置 + 预览。"""
    return _get_svc().list_chunks(doc_key=doc_key, q=q, limit=limit)


@mcp.tool()
def policy_delete_document(doc_key: str, confirm: bool = False) -> dict[str, Any]:
    """删除文档及其全部分块/向量（不可恢复）。confirm 必须为 true 才执行。"""
    if not confirm:
        return {"status": "rejected",
                "reason": "删除不可恢复。确认后请以 confirm=true 再次调用。",
                "doc_key": doc_key}
    return _get_svc().delete_document(doc_key)


@mcp.tool()
def policy_ingest(path: str | None = None, file_base64: str | None = None,
                        filename: str | None = None, doc_id: str | None = None,
                        version: str | None = None,
                        effective_date: str | None = None,
                        use_vision: bool = True) -> dict[str, Any]:
    """入库一份政策/金融文档（pdf/docx/md/txt）。两种方式二选一：

    1) path: 服务器本地文件路径（用户消息里引用了文件路径时用这个）；
    2) file_base64 + filename: 文件内容 base64 编码（用户直接给了文件内容时用）。

    use_vision: 复杂图表（研报/论文类）是否调用视觉 API 转写图内数值（默认开）。

    入库即自动开启公式/图表扫描（富化）。响应含 notice 回执文本——请**原样转告用户**，
    内含：版本判断（标题快判+全文确判）、扫描大致时间、"扫描过程中也可以提问"提示。
    响应中的 needs_vision/vision 字段与 [vision_*] 警示也须转告用户：
    视觉能力以探测为准（宿主共享 API Key 未必有视觉能力）；不转写则图表数值类问题将拒答。
    """
    from ...service.files import ingest_bytes, ingest_path_ref
    svc = _get_svc()
    meta = {"doc_id": doc_id, "version": version, "effective_date": effective_date}
    if file_base64 and filename:
        import base64
        try:
            data = base64.b64decode(file_base64)
        except Exception as e:  # noqa: BLE001
            return {"status": "rejected", "reason": f"base64 解码失败: {e}"}
        return ingest_bytes(svc, data, filename, use_vision=use_vision,
                            **{k: v for k, v in meta.items() if v})
    if path:
        return ingest_path_ref(svc, path, use_vision=use_vision,
                               **{k: v for k, v in meta.items() if v})
    return {"status": "rejected", "reason": "需要 path 或 file_base64+filename 之一"}


@mcp.tool()
def policy_vision_check(force: bool = False) -> dict[str, Any]:
    """视觉能力验证（发一张已知数字的测试图探测，模型名不可信、探测为准）。

    处理研报/论文等复杂图表文档前可先调用；capable=false 时请告知用户
    配置带视觉能力的 API Key（POLICYRAG_VISION_API_KEY 或更换共享键），
    否则图内数值不转写、图表数值类问题将拒答。
    """
    return _get_svc().vision_check(force=force)


# ---------------- 人工复核 ----------------

@mcp.tool()
def policy_review_list(status: str | None = "pending") -> list[dict[str, Any]]:
    """人工复核队列。status: pending|approved|rejected|空=全部。"""
    return _get_svc().list_review_tasks(status=status or None)


@mcp.tool()
def policy_review_decide(task_id: str, approve: bool,
                               note: str = "", final_answer: str = "") -> dict[str, Any]:
    """对复核任务给出人工决定。approve=true 通过（可附修订稿 final_answer），false 否决。"""
    return _get_svc().decide_review_task(task_id, approve=approve, note=note,
                                         final_answer=final_answer)


@mcp.tool()
def policy_verify(answer: str, citations: list[dict[str, str]]) -> dict[str, Any]:
    """独立复核一份带引用的答案草稿（数字保真 + 引用逐字命中校验）。"""
    return _get_svc().verify(answer, citations)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="Streamable HTTP 模式")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    if args.http:
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
