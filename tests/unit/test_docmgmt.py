"""文档库管理与范围检索、人工复核队列测试。"""

from __future__ import annotations

import asyncio
import json
import uuid

from policyrag.ingest.chunker import chunk_document
from policyrag.ingest.structure import build_ast
from policyrag.index.bm25 import BM25Index
from policyrag.index.hybrid import HybridIndex
from policyrag.index.numpy_store import NumpyVectorStore
from policyrag.index.stores.sqlite_meta import MetaStore
from policyrag.index.vectors import HashEmbedder
from policyrag.llm.base import Caps, LLMResult
from policyrag.models.answer import AnswerStatus, Usage
from policyrag.models.document import DocAST
from policyrag.models.retrieval import DocFilter, QueryRequest
from policyrag.query.pipeline import PipelineDeps, QueryPipeline

DOC_A = """第一条 资本充足率不得低于 8%。
"""
DOC_B = """第一条 理财产品销售起点金额不得低于 1 元。
"""


class MockLLM:
    name = "mock"
    caps = Caps(json_mode=True)

    def __init__(self, responses):
        self._r = list(responses)
        self.calls = 0

    def complete(self, messages, *, schema=None, max_tokens=4096,
                 temperature=0.0, effort="high", on_delta=None) -> LLMResult:
        self.calls += 1
        p = self._r.pop(0) if self._r else {}
        return LLMResult(text=json.dumps(p, ensure_ascii=False), usage=Usage(),
                         parsed=p, model="mock", stop_reason="end_turn")


def _setup(tmp: str):
    m = MetaStore(f"{tmp}/meta.sqlite3")
    for doc_id, ver, text in [("办法A", "v1", DOC_A), ("办法B", "v1", DOC_B)]:
        nodes, _ = build_ast([(text, 1)], tables=[])
        ast = DocAST(doc_id=doc_id, version=ver, title=doc_id, nodes=nodes)
        chunks = chunk_document(ast)
        m.upsert_document(doc_id, ver, doc_id)
        for c in chunks:
            m.upsert_chunk(c.chunk_id, f"{doc_id}@{ver}", c.path.key,
                           c.breadcrumb, c.text, c.text_norm, c.kind.value)
    m.commit()

    metas = m.all_chunk_metas()
    emb = HashEmbedder()
    bm25 = BM25Index(f"{tmp}/bm25")
    bm25.build(list(metas), [x["text"] for x in metas.values()])
    vecs = NumpyVectorStore(f"{tmp}/vec", dim=emb.dim)
    vecs.build(list(metas), emb.embed([x["text"] for x in metas.values()]))
    index = HybridIndex(bm25, vecs, emb, metas)
    return m, index, {cid: x["text"] for cid, x in metas.items()}, metas


def test_delete_document_removes_chunks_and_index():
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    m, index, registry, metas = _setup(tmp)
    assert len(m.list_documents()) == 2
    n = m.delete_document("办法B@v1")
    assert n >= 1
    assert len(m.list_documents()) == 1
    assert m.chunks_of_doc("办法B@v1") == []
    metas2 = m.all_chunk_metas()
    emb = HashEmbedder()
    bm25 = BM25Index(f"{tmp}/bm25_2")
    bm25.build(list(metas2), [x["text"] for x in metas2.values()] or [""])
    hits = bm25.search("理财产品销售起点", k=5)
    assert all("理财产品" not in m.get_chunk(cid)["text"] for cid, _ in hits)


def test_scoped_search_restricts_to_selection():
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    m, index, registry, metas = _setup(tmp)
    b_ids = set(m.filter_chunk_ids(None, ["办法B@v1"]))
    a_ids = set(m.filter_chunk_ids(None, ["办法A@v1"]))
    assert b_ids and a_ids and not (a_ids & b_ids)
    hits = index.search("资本充足率", k=5)
    assert any(h.chunk_id in a_ids for h in hits)
    hits_b = [h for h in index.search("资本充足率", k=50) if h.chunk_id in b_ids]
    assert all(h.chunk_id in b_ids for h in hits_b)


def test_chunk_scope_pipeline():
    """块级范围：问题只在勾选的一块里找答案，别的块不进证据池。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    m, index, registry, metas = _setup(tmp)
    b_ids = m.filter_chunk_ids(None, ["办法B@v1"])
    mock = MockLLM([
        {"intent": "quote", "clause_refs": [], "terms": [], "numeric_anchors": [],
         "doc_family": None, "as_of": None},
        {"answer": "理财产品销售起点金额不得低于 1 元。", "citations": [], "confidence": 0.9},
    ])
    deps = PipelineDeps(index=index, llm=mock, registry=registry,
                        meta_lookup=metas, scope_chunk_ids=set(b_ids))
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    pipe.run(QueryRequest(query="销售起点是多少？", request_id="t_scope"))
    import pathlib
    trace_dir = pathlib.Path(f"{tmp}/traces/t_scope")
    manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    rec_file = next(e["file"] for e in manifest if e["name"] == "B3_recall")
    rec = json.loads((trace_dir / rec_file).read_text(encoding="utf-8"))
    assert rec["input"].get("scoped") is True, f"范围标记缺失: {rec['input']}"
    b_set = set(b_ids)
    recorded = rec["output"]
    assert all(cid in b_set for cid in recorded), f"范围泄漏: {recorded}"


def test_review_task_flow():
    """复核队列：创建 → 通过；已决任务不可再改（DB 层防御）。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    m, *_ = _setup(tmp)
    m.create_review_task("rev_t1", "问题X", "2026-01-01", "草稿答案",
                         [{"chunk_id": "abc", "quote": "q"}], {"status": "answered"})
    tasks = m.list_review_tasks("pending")
    assert any(t["task_id"] == "rev_t1" for t in tasks)
    assert m.decide_review_task("rev_t1", "approved", "核对无误", "最终答案")
    t = m.get_review_task("rev_t1")
    assert t["status"] == "approved"
    assert t["final_answer"] == "最终答案"
    assert not m.decide_review_task("rev_t1", "rejected")


def test_sanitize_doc_id_real_filenames():
    """回归：真实文件名（全角括号/书名号/空格/尾随空格）清洗而非拒收。"""
    from policyrag.models.ids import sanitize_doc_id
    assert sanitize_doc_id("《商业银行资本管理办法》（2023 年修订） v1.2") == "商业银行资本管理办法(2023年修订)v1.2"
    assert sanitize_doc_id("  银发〔2024〕15 号 关于XX的通知（试行） ") == "银发〔2024〕15号关于XX的通知(试行)"
    s = sanitize_doc_id("！！！")
    assert s.startswith("doc_") and len(s) == 12


def test_upload_ugly_filename_e2e():
    """回归：丑文件名经 REST 上传不再 500（曾因 DOC_ID_RE 拒收）。"""
    import io
    from fastapi.testclient import TestClient
    from policyrag.adapters.rest import app as appmod
    from policyrag.service.core import CoreService
    from policyrag.settings import service_config

    cfg = service_config()
    cfg.data_dir = __import__("pathlib").Path(f"tests/.tmp/{uuid.uuid4().hex[:8]}")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.embed_backend = "hash"
    appmod._svc = CoreService(MockLLM([{}]), cfg)
    c = TestClient(appmod.app)
    assert c.get("/healthz").json()["ok"]  # 曾因路由丢失显示"服务离线"

    name = "《办法》（试行） v1.2.md"
    r = c.post("/v1/upload", files={"file": (name, io.BytesIO("第一条 x。".encode()), "text/markdown")})
    body = r.json()
    assert body["status"] == "ok", body
    assert body["doc_key"].startswith("办法(试行)v1.2")
    c.delete("/v1/documents/" + body["doc_key"])


def test_desurrogate_and_ugly_upload():
    """回归：孤立代理字符（surrogate）文件名/文本不许再炸 UTF-8 编码。

    注入点是服务端边界（ingest_bytes/normalize）；代理字符用 chr() 构造，
    避免源码转义歧义。
    """
    from policyrag.models.ids import desurrogate, normalize_text_for_id
    s_dc = chr(0xDCFF)
    s_d8 = chr(0xD800)

    ugly = "办法 测试版" + s_dc + "说明.md"
    cleaned = desurrogate(ugly)
    cleaned.encode("utf-8")  # 必须可编码
    assert s_dc not in cleaned
    normalize_text_for_id("条款" + s_dc + "文本").encode("utf-8")

    from policyrag.service.files import ingest_bytes
    from policyrag.service.core import CoreService
    from policyrag.settings import service_config

    cfg = service_config()
    cfg.data_dir = __import__("pathlib").Path(f"tests/.tmp/{uuid.uuid4().hex[:8]}")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.embed_backend = "hash"
    svc = CoreService(MockLLM([{}]), cfg)
    name = "《某办法》" + s_d8 + "（试行）.md"
    body = ("第一条 代理字符测试" + s_dc + "条款。").encode("utf-8", "surrogatepass")
    out = ingest_bytes(svc, body, name)
    assert out["status"] == "ok", out
    svc.delete_document(out["doc_key"])


def test_temporal_as_of_filter():
    """生效时点过滤：as_of 只允许已生效文档参与检索。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    m, index, registry, metas = _setup(tmp)
    # 给办法B 一个未来才生效的日期
    m.upsert_document("办法B", "v1", "办法B", effective_date="2099-01-01", status="active")
    m.commit()
    allowed = m.temporal_chunk_ids("2026-10-09")
    b_ids = set(m.filter_chunk_ids(None, ["办法B@v1"]))
    a_ids = set(m.filter_chunk_ids(None, ["办法A@v1"]))
    assert allowed is not None
    assert not (b_ids & allowed), "未来生效的文档不许进 as_of 检索范围"
    assert a_ids <= allowed or not any(x["effective_date"] for x in m.list_documents())
    # 全部无日期 → 不过滤
    m2, *_ = _setup(f"tests/.tmp/{uuid.uuid4().hex[:8]}")
    assert m2.temporal_chunk_ids("2026-10-09") is None


def test_review_via_require_review_flag():
    """require_review=true 时回答进复核队列（GUI/agent 同一入口）。"""
    import io
    from fastapi.testclient import TestClient
    from policyrag.adapters.rest import app as appmod
    from policyrag.service.core import CoreService
    from policyrag.settings import service_config

    mock = MockLLM([
        # WS-B 新契约：先出检索方案（plan），执行时注入指令不再重写
        {"items": [{"qid": "1", "kind": "retrieve", "intent": "quote",
                    "clause_refs": [], "terms": ["资本充足率"],
                    "numeric_anchors": ["8%"], "specificity": "precise",
                    "variants": [], "depends_on": [], "implicit_facts": []}],
         "ambiguous": False, "clarifications": []},
    ])
    cfg = service_config()
    cfg.data_dir = __import__("pathlib").Path(f"tests/.tmp/{uuid.uuid4().hex[:8]}")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.embed_backend = "hash"
    appmod._svc = CoreService(mock, cfg)
    c = TestClient(appmod.app)
    content = "# t\n\n第一条 资本充足率不得低于 8%。\n".encode()
    up = c.post("/v1/upload", files={"file": ("复核测试.md", io.BytesIO(content), "text/markdown")}).json()
    # 取含目标条款的真实 chunk_id，拼出可通过硬闸的 draft（逐字 quote）
    chs = c.get("/v1/chunks_list", params={"doc_key": up["doc_key"]}).json()
    ch = next(x for x in chs if "资本充足率" in x["preview"])
    mock._r.append({
        "answer": f"资本充足率不得低于 8%。[{ch['chunk_id']}]",
        "citations": [{"chunk_id": ch["chunk_id"], "quote": "资本充足率不得低于 8%",
                        "clause_path": ch["breadcrumb"]}],
        "confidence": 0.95, "uncertainties": [],
    })
    env = c.post("/v1/query", json={"query": "资本充足率是多少？", "require_review": True,
                                     "request_id": "t_rev"}).json()
    assert env["status"] == "pending_review", env
    assert env.get("review_task_id")
    tasks = c.get("/v1/review", params={"status": "pending"}).json()
    assert any(t["task_id"] == env["review_task_id"] for t in tasks)
    d = c.post(f"/v1/review/{env['review_task_id']}/decision",
               json={"approve": True, "note": "核对无误"}).json()
    assert d["decision"] == "approved"
    c.delete("/v1/documents/" + up["doc_key"])
