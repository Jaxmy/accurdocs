"""QueryPipeline 端到端（Mock LLM）：B0-B9 全链、硬闸拦截、拒答。

不花 API 钱——Mock LLM 按调用顺序回放预置 JSON。
注意 SHORT/FULL 路径的 LLM 调用次数不同：
- SHORT（简单事实）：rewrite + draft = 2 次
- FULL（综合）：rewrite + sufficiency + draft = 3 次
"""

from __future__ import annotations

import asyncio
import json
import uuid

from policyrag.ingest.chunker import chunk_document
from policyrag.ingest.structure import build_ast
from policyrag.index.bm25 import BM25Index
from policyrag.index.hybrid import HybridIndex
from policyrag.index.numpy_store import NumpyVectorStore
from policyrag.index.vectors import HashEmbedder
from policyrag.llm.base import Caps, LLMResult
from policyrag.models.answer import AnswerStatus, Usage
from policyrag.models.document import DocAST
from policyrag.models.retrieval import QueryRequest
from policyrag.query.pipeline import PipelineDeps, QueryPipeline

DOC = """第一章 总则

第一条 为加强贷款管理，防范金融风险，制定本办法。

第二条 资本充足率不得低于 8%，杠杆率不得超过 6%。自 2024 年 3 月 1 日起施行。
"""


class MockLLM:
    """按调用顺序回放预置 JSON。"""

    name = "mock"
    caps = Caps(structured_output=False, json_mode=True)

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, schema=None, max_tokens=4096,
                 temperature=0.0, effort="high", on_delta=None) -> LLMResult:
        self.calls += 1
        payload = self._responses.pop(0) if self._responses else {}
        if on_delta is not None:
            on_delta(json.dumps(payload, ensure_ascii=False))
        return LLMResult(
            text=json.dumps(payload, ensure_ascii=False),
            usage=Usage(input_tokens=100, output_tokens=50, cost_usd=0.001),
            parsed=payload, model="mock", stop_reason="end_turn",
        )


def _build_index(tmp: str):
    nodes, _ = build_ast([(DOC, 1)], tables=[])
    ast = DocAST(doc_id="测试办法", version="2024-03-01", title="测试贷款办法", nodes=nodes)
    chunks = chunk_document(ast)
    registry = {c.chunk_id: c.text for c in chunks}
    metas = {
        c.chunk_id: {
            "text": c.text, "text_norm": c.text_norm, "breadcrumb": c.breadcrumb,
            "doc_title": ast.title, "version": ast.version, "kind": c.kind.value,
        }
        for c in chunks
    }
    embedder = HashEmbedder()
    bm25 = BM25Index(f"{tmp}/bm25")
    bm25.build(list(registry.keys()), list(registry.values()))
    vecs = NumpyVectorStore(f"{tmp}/vec", dim=embedder.dim)
    vecs.build(list(registry.keys()), embedder.embed(list(registry.values())))
    index = HybridIndex(bm25, vecs, embedder, metas)
    return index, registry, metas, chunks


def _empty_index(tmp: str):
    import numpy as np
    embedder = HashEmbedder()
    bm25 = BM25Index(f"{tmp}/bm25_empty")
    bm25.build([], [])
    vecs = NumpyVectorStore(f"{tmp}/vec_empty", dim=embedder.dim)
    vecs.build([], np.zeros((0, embedder.dim), dtype=np.float32))
    return HybridIndex(bm25, vecs, embedder, {})


def _find_chunk(chunks, keyword: str) -> str:
    return next(c.chunk_id for c in chunks if keyword in c.text)


def _run(pipe: QueryPipeline, req: QueryRequest) -> AnswerEnvelope:
    return pipe.run(req)


from policyrag.models.answer import AnswerEnvelope  # noqa: E402


def test_full_path_verified_citation():
    """FULL 路径（mode=analyze）：rewrite + sufficiency + draft → answered。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    index, registry, metas, chunks = _build_index(tmp)
    cid = _find_chunk(chunks, "资本充足率")
    mock = MockLLM([
        {"intent": "analyze", "clause_refs": ["第二条"], "terms": ["资本充足率"],
         "numeric_anchors": ["8%"], "doc_family": None, "as_of": None},
        {"verdict": "sufficient", "needed": [], "confidence": 0.9},
        {"answer": f"资本充足率不得低于 8%。[{cid}]",
         "citations": [{"chunk_id": cid, "quote": "资本充足率不得低于 8%", "clause_path": "第二条"}],
         "confidence": 0.95, "uncertainties": []},
    ])
    deps = PipelineDeps(index=index, llm=mock, registry=registry,
                        meta_lookup=metas, corpus_version="test")
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    env = _run(pipe, QueryRequest(query="请分析资本充足率的合规要求和实施时间。",
                                  mode="analyze", request_id="t_full"))
    assert env.status == AnswerStatus.ANSWERED, (env.status, env.refusal, env.verification)
    assert env.citations and env.citations[0].verified
    assert env.verification.quotes_ok and env.verification.numbers_ok


def test_short_path_verified():
    """SHORT 路径（简单数值题）：rewrite + draft = 2 次调用。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    index, registry, metas, chunks = _build_index(tmp)
    cid = _find_chunk(chunks, "资本充足率")
    mock = MockLLM([
        {"intent": "quote", "clause_refs": ["第二条"], "terms": ["资本充足率"],
         "numeric_anchors": ["8%"], "doc_family": None, "as_of": None},
        {"answer": f"资本充足率不得低于 8%。[{cid}]",
         "citations": [{"chunk_id": cid, "quote": "资本充足率不得低于 8%", "clause_path": "第二条"}],
         "confidence": 0.95, "uncertainties": []},
    ])
    deps = PipelineDeps(index=index, llm=mock, registry=registry, meta_lookup=metas)
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    env = _run(pipe, QueryRequest(query="资本充足率的最低要求是多少？", request_id="t_short"))
    assert env.status == AnswerStatus.ANSWERED, (env.status, env.refusal)
    assert mock.calls == 2
    assert env.meta["tier"] == "short"


def test_blocks_fabricated_number():
    """答案编造 12% → 数字硬闸 → 删句后空答案 → 拒答。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    index, registry, metas, chunks = _build_index(tmp)
    cid = _find_chunk(chunks, "资本充足率")
    mock = MockLLM([
        {"intent": "quote", "clause_refs": [], "terms": [], "numeric_anchors": [],
         "doc_family": None, "as_of": None},
        {"answer": f"资本充足率不得低于 12%。[{cid}]",
         "citations": [{"chunk_id": cid, "quote": "资本充足率不得低于 8%", "clause_path": "第二条"}],
         "confidence": 0.9, "uncertainties": []},
    ])
    deps = PipelineDeps(index=index, llm=mock, registry=registry, meta_lookup=metas)
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    env = _run(pipe, QueryRequest(query="资本充足率是多少？", request_id="t_num"))
    assert env.status == AnswerStatus.REFUSED or "12%" not in env.answer
    if env.status == AnswerStatus.ANSWERED:
        assert env.verification.numbers_ok
    else:
        assert env.refusal.reason == "verification_failed"


def test_blocks_fabricated_quote():
    """引用 quote 编造 → 引用硬闸 → 拒答（verification_failed）。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    index, registry, metas, chunks = _build_index(tmp)
    cid = _find_chunk(chunks, "资本充足率")
    mock = MockLLM([
        {"intent": "quote", "clause_refs": [], "terms": [], "numeric_anchors": [],
         "doc_family": None, "as_of": None},
        {"verdict": "sufficient", "needed": [], "confidence": 0.9},
        {"answer": f"相关规定见第二条。[{cid}]",
         "citations": [{"chunk_id": cid, "quote": "本办法规定资本充足率不得低于 15%",
                        "clause_path": "第二条"}],
         "confidence": 0.9, "uncertainties": []},
    ])
    deps = PipelineDeps(index=index, llm=mock, registry=registry, meta_lookup=metas)
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    env = _run(pipe, QueryRequest(query="请综合分析相关规定的内容与适用范围。", mode="analyze",
                                  request_id="t_quote"))
    assert env.status == AnswerStatus.REFUSED
    assert env.refusal and env.refusal.reason == "verification_failed"
    assert not env.verification.quotes_ok


def test_refuses_when_no_evidence():
    """空索引 → 检索空 → 直接拒答 insufficient_evidence（不进入生成）。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    empty = _empty_index(tmp)
    mock = MockLLM([
        {"intent": "quote", "clause_refs": [], "terms": [], "numeric_anchors": [],
         "doc_family": None, "as_of": None},
    ])
    deps = PipelineDeps(index=empty, llm=mock, registry={}, meta_lookup={})
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    env = _run(pipe, QueryRequest(query="zzzzz-not-in-corpus", request_id="t_empty"))
    assert env.status == AnswerStatus.REFUSED
    assert env.refusal.reason == "insufficient_evidence"
    assert mock.calls == 1  # 只跑了 rewrite，没进生成


def test_expand_clause_siblings_pulls_same_clause_items():
    """R2-06：同条枚举跨块（1./2./3. 分块）→ 兄弟块补入证据池；他条块不拉。"""
    from policyrag.models.retrieval import Evidence
    a, b, c = "a" * 16, "b" * 16, "c" * 16
    metas = {
        a: {"text": "电力市场注册遵循原则 1.规范入市。", "breadcrumb": "细则 第二章 第四条",
            "page": 268, "kind": "article", "doc_title": "t", "version": "v"},
        b: {"text": "2.公开透明。…3.全国统一。…", "breadcrumb": "细则 第二章 第四条",
            "page": 269, "kind": "paragraph", "doc_title": "t", "version": "v"},
        c: {"text": "无关条文。", "breadcrumb": "细则 第二章 第五条",
            "page": 270, "kind": "article", "doc_title": "t", "version": "v"},
    }
    deps = PipelineDeps(index=None, llm=None,
                        registry={k: v["text"] for k, v in metas.items()},
                        meta_lookup=metas)
    pipe = QueryPipeline(deps, trace_root="tests/.tmp/t_sib")
    ev = [Evidence(chunk_id=a, text=metas[a]["text"], doc_title="t", doc_version="v",
                   clause_path="细则 第二章 第四条", page=268)]
    out = pipe._expand_clause_siblings(ev)
    ids = {e.chunk_id for e in out}
    assert b in ids, "同条尾项（2./3.）必须补入"
    assert c not in ids, "他条块不得混入"
    # 上限：每条 ≤4
    assert len(out) <= 5


def test_merge_envelopes_partial_semantics():
    """R2-11 F5d：池内全拒才 refused；部分拒/部分作答 → partial；全答 → answered。"""
    from policyrag.models.answer import AnswerEnvelope, AnswerStatus, Refusal
    from policyrag.service.core import merge_envelopes

    def _env(status, answer="", detail=""):
        return AnswerEnvelope(
            request_id="t", status=status, answer=answer, confidence=0.8,
            as_of="2026-10-12", corpus_version="v",
            refusal=Refusal(reason="synthesis_rejected", detail=detail) if detail else None)

    m1 = merge_envelopes([("q1", _env(AnswerStatus.ANSWERED, "甲")),
                          ("q2", _env(AnswerStatus.REFUSED, "", "拒乙"))],
                         request_id="r", corpus_version="v")
    assert m1.status == AnswerStatus.PARTIAL          # 部分拒 ≠ 全拒
    assert "甲" in m1.answer and "拒答" in m1.answer

    m2 = merge_envelopes([("q1", _env(AnswerStatus.REFUSED, "", "拒甲")),
                          ("q2", _env(AnswerStatus.REFUSED, "", "拒乙"))],
                         request_id="r", corpus_version="v")
    assert m2.status == AnswerStatus.REFUSED          # 全拒才拒

    m3 = merge_envelopes([("q1", _env(AnswerStatus.ANSWERED, "甲")),
                          ("q2", _env(AnswerStatus.PARTIAL, "（分析推理未通过校验已省略）"))],
                         request_id="r", corpus_version="v")
    assert m3.status == AnswerStatus.PARTIAL
    assert "分析推理未通过校验已省略" in m3.answer      # 占位照登，不打"拒答"话术
    assert "拒答：" not in m3.answer
