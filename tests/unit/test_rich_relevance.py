"""富化块返回机制回归：去重（同块多次引用只出一条）+ 相关度过滤（同页他节公式不混入）。"""

import json
import uuid

from policyrag.models.answer import AnswerStatus
from policyrag.models.retrieval import QueryRequest
from policyrag.query.pipeline import (
    PipelineDeps,
    QueryPipeline,
    rich_relevant,
    rich_relevant_tokens,
)

FORMULA_TEXT = "【公式】R_{ 省间日前差价电能量电费 ,i} = \\sum Q_{ 省间日前 ,i} \\times P_{ 节点 ,i}"
NOISE_TEXT = "【公式】C_{ 调平用电 ,j} = Q_{ 调平用电 ,j} \\times P_{ 实时月度加权} 电化学储能调平费用"


class TestRichRelevant:
    def test_question_tokens(self):
        toks = rich_relevant_tokens("省间日前市场差价电能量电费怎么计算？")
        assert "省间" in toks or "日前" in toks
        assert "怎么" not in toks and "计算" not in toks

    def test_relevant_formula_passes(self):
        q = rich_relevant_tokens("省间日前市场差价电能量电费怎么计算？")
        assert rich_relevant(q, FORMULA_TEXT)

    def test_same_page_other_section_filtered(self):
        """同页他节公式（电化学储能调平费用）不得混入省间日前差价答案。"""
        q = rich_relevant_tokens("省间日前市场差价电能量电费怎么计算？")
        assert not rich_relevant(q, NOISE_TEXT)


class MockLLM:
    name = "mock"
    caps = type("C", (), {"structured_output": False, "json_mode": True})()

    def __init__(self, responses):
        self._r = list(responses)
        self.calls = 0

    def complete(self, messages, *, schema=None, max_tokens=4096,
                 temperature=0.0, effort="high", on_delta=None):
        from policyrag.llm.base import LLMResult
        from policyrag.models.answer import Usage
        self.calls += 1
        p = self._r.pop(0) if self._r else {}
        return LLMResult(text=json.dumps(p, ensure_ascii=False), usage=Usage(),
                         parsed=p, model="mock", stop_reason="end_turn")


def _build(tmp, include_noise=True):
    from policyrag.index.bm25 import BM25Index
    from policyrag.index.hybrid import HybridIndex
    from policyrag.index.numpy_store import NumpyVectorStore
    from policyrag.index.vectors import HashEmbedder
    from policyrag.index.stores.sqlite_meta import MetaStore

    m = MetaStore(f"{tmp}/meta.sqlite3")
    m.upsert_document("规则", "v1", "测试市场规则")
    m.upsert_chunk("t1", "规则@v1", "第一条", "第一条", "省间日前市场差价电能量电费按以下公式计算。",
                   "省间日前市场差价电能量电费按以下公式计算。", "text", page=10)
    m.upsert_chunk("fm1", "规则@v1", "公式/fm1", "第10页 公式fm1",
                   FORMULA_TEXT, FORMULA_TEXT, "formula", page=10,
                   meta={"latex": "R = \\sum Q \\times P", "image": "/x/fm1.png",
                         "pending": False, "confidence": "high"})
    if include_noise:
        m.upsert_chunk("fm2", "规则@v1", "公式/fm2", "第10页 公式fm2",
                       NOISE_TEXT, NOISE_TEXT, "formula", page=10,
                       meta={"latex": "C = Q \\times P", "image": "/x/fm2.png",
                             "pending": False, "confidence": "high"})
    m.commit()
    metas = m.all_chunk_metas()
    registry = {cid: row["text"] for cid, row in metas.items()}
    embedder = HashEmbedder()
    bm25 = BM25Index(f"{tmp}/bm25")
    bm25.build(list(metas), [metas[c]["text"] for c in metas])
    vecs = NumpyVectorStore(f"{tmp}/vec", dim=embedder.dim)
    vecs.build(list(metas), embedder.embed([metas[c]["text"] for c in metas]))
    index = HybridIndex(bm25, vecs, embedder, metas)
    return index, registry, metas, m


def _run(tmp, draft, question="省间日前市场差价电能量电费怎么计算？"):
    index, registry, metas, _ = _build(tmp)
    rewrite = {"intent": "quote", "clause_refs": [], "terms": ["省间日前"],
               "numeric_anchors": [], "doc_family": None, "as_of": None}
    mock = MockLLM([rewrite, draft])
    deps = PipelineDeps(index=index, llm=mock, registry=registry,
                        meta_lookup=metas, corpus_version="t",
                        live_page_rich=lambda dk, pg: [
                            dict(r) for r in __import__(
                                "policyrag.index.stores.sqlite_meta",
                                fromlist=["MetaStore"]).MetaStore(
                                f"{tmp}/meta.sqlite3").rich_blocks_of_page(dk, pg)])
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    return pipe.run(QueryRequest(query=question, request_id="t_rich"))


def test_duplicate_citations_deduped():
    """同一公式块被引用两次 → formulas 只出一条（q_done 路径不经 merge 去重）。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    draft = {
        "answer": "省间日前差价电费按公式计算。[fm1]其中电量为日前合约电量。[fm1]",
        "citations": [
            {"chunk_id": "fm1", "quote": "R_{ 省间日前差价电能量电费 ,i}", "clause_path": "公式"},
            {"chunk_id": "fm1", "quote": "Q_{ 省间日前 ,i}", "clause_path": "公式"},
        ],
        "confidence": 0.9, "uncertainties": [],
    }
    env = _run(tmp, draft)
    assert env.status == AnswerStatus.ANSWERED, (env.status, env.refusal, env.verification)
    fm_ids = [f["chunk_id"] for f in env.formulas]
    assert fm_ids == ["fm1"], f"同块重复引用必须去重，实测 {fm_ids}"


def test_same_page_noise_filtered():
    """同页无关公式（储能调平）不随引用附带；相关公式正常附带。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    draft = {
        "answer": "省间日前差价电费按公式计算。[t1]",
        "citations": [
            {"chunk_id": "t1", "quote": "省间日前市场差价电能量电费按以下公式计算",
             "clause_path": "第一条"},
        ],
        "confidence": 0.9, "uncertainties": [],
    }
    env = _run(tmp, draft)
    fm_ids = [f["chunk_id"] for f in env.formulas]
    assert "fm1" in fm_ids, "相关公式应随引用附带"
    assert "fm2" not in fm_ids, f"同页无关公式不得混入，实测 {fm_ids}"
