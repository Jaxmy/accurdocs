"""WS-B 检索方案 + G1 execute 注入 + WS-G 展示分层 测试。"""

import json
import uuid

from policyrag.llm.base import Caps, LLMResult
from policyrag.models.answer import Usage
from policyrag.query.search_plan import (
    generate_plan,
    plan_from_dict,
    plan_need_confirm,
)


class MockLLM:
    name = "mock"
    caps = Caps(json_mode=True)

    def __init__(self, responses):
        self._r = list(responses)
        self.calls = 0
        self.seen = []

    def complete(self, messages, *, schema=None, max_tokens=4096,
                 temperature=0.0, effort="high", on_delta=None):
        self.calls += 1
        p = self._r.pop(0) if self._r else {}
        self.seen.append(p)
        return LLMResult(text=json.dumps(p, ensure_ascii=False), usage=Usage(),
                         parsed=p, model="mock", stop_reason="end_turn")


PLAN_RESP = {
    "items": [
        {"qid": "1", "kind": "retrieve", "intent": "quote",
         "clause_refs": ["第十二条"], "terms": ["资本充足率"],
         "numeric_anchors": ["8%"], "specificity": "precise",
         "variants": [
             {"query": "资本充足率最低要求", "terms": ["资本充足率"],
              "clause_refs": [], "numeric_anchors": [], "confidence": 0.8,
              "basis": "同义词"},
             {"query": "资本比率下限", "terms": ["资本比率"],
              "clause_refs": [], "numeric_anchors": [], "confidence": 0.9,
              "basis": "上位词"},
             {"query": "弱改写", "terms": [], "clause_refs": [],
              "numeric_anchors": [], "confidence": 0.3, "basis": "低置信"},
             {"query": "超量候选4", "terms": [], "clause_refs": [],
              "numeric_anchors": [], "confidence": 0.7, "basis": "同义词"},
             {"query": "超量候选5", "terms": [], "clause_refs": [],
              "numeric_anchors": [], "confidence": 0.6, "basis": "同义词"},
         ]},
        {"qid": "2", "kind": "reason", "intent": "analyze",
         "clause_refs": [], "terms": ["影响"], "numeric_anchors": [],
         "specificity": "precise", "variants": [],
         "depends_on": ["1"], "implicit_facts": ["该办法的适用范围是什么？"]},
    ],
    "ambiguous": False, "clarifications": [],
}


class TestGeneratePlan:
    def test_parse_and_variant_rules(self):
        mock = MockLLM([PLAN_RESP])
        plan = generate_plan(mock, ["第十二条的资本充足率是多少？", "分析其影响"], as_of="2026-01-01")
        it1, it2 = plan.items
        assert it1.kind == "retrieve" and it2.kind == "reason"
        # variants：低置信剔除、降序、≤3
        assert len(it1.variants) <= 3
        assert all(v.confidence >= 0.5 for v in it1.variants)
        assert it1.variants[0].confidence >= it1.variants[1].confidence
        assert it2.depends_on == ["1"]
        assert it2.implicit_facts == ["该办法的适用范围是什么？"]
        # 条款规范化去重
        assert it1.clause_refs == ["第十二条"]

    def test_need_confirm_rules(self):
        mock = MockLLM([PLAN_RESP])
        plan = generate_plan(mock, ["q1？", "q2？"])
        assert plan_need_confirm(plan) is True, "含 reason/variants 须确认"

        resp = {"items": [{"qid": "1", "kind": "retrieve", "clause_refs": [],
                           "terms": ["x"], "numeric_anchors": [],
                           "specificity": "precise", "variants": []}],
                "ambiguous": False, "clarifications": []}
        plan2 = generate_plan(MockLLM([resp]), ["单题明确？"])
        assert plan_need_confirm(plan2) is False

        resp3 = {"items": [{"qid": "1", "kind": "retrieve", "clause_refs": [],
                            "terms": [], "numeric_anchors": [],
                            "specificity": "broad", "variants": []}],
                 "ambiguous": False, "clarifications": ["您是指 A 还是 B？"]}
        plan3 = generate_plan(MockLLM([resp3]), ["宽泛模糊？"])
        assert plan_need_confirm(plan3) is True

    def test_missing_wrapper_recovered(self):
        """G 回归：模型漏外层 items 包装（实测网关行为）→ 平铺对象扫描兜底。"""
        flat = ('{"qid": "1", "kind": "retrieve", "clause_refs": [], '
                '"terms": ["a"], "numeric_anchors": []}\n'
                '{"qid": "2", "kind": "reason", "clause_refs": [], '
                '"terms": [], "numeric_anchors": [], "depends_on": ["1"]}')
        mock = MockLLM([{}])
        # 绕过 MockLLM 的 parsed：直接构造带 text 的假结果
        from policyrag.llm.base import LLMResult

        class FlatLLM:
            name = "flat"
            caps = Caps(json_mode=True)

            def complete(self, messages, **kw):
                return LLMResult(text=flat, usage=Usage(), parsed={"qid": "1"},
                                 model="flat", stop_reason="end_turn")

        plan = generate_plan(FlatLLM(), ["问题一？", "问题二？"])
        assert [i.kind for i in plan.items] == ["retrieve", "reason"]
        assert plan.items[1].depends_on == ["1"]

    def test_degraded_fallback(self):
        mock = MockLLM([{"乱": "码"}])   # 无 items
        plan = generate_plan(mock, ["问题一？"])
        assert plan.degraded is not None
        assert plan.items and plan.items[0].kind == "retrieve"
        assert plan_need_confirm(plan) is False, "降级不弹面板，直执行"

    def test_batching_sharded(self):
        mock = MockLLM([
            {"items": [{"qid": str(i), "kind": "retrieve", "clause_refs": [],
                        "terms": [], "numeric_anchors": []} for i in range(1, 9)]},
            {"items": [{"qid": str(i), "kind": "retrieve", "clause_refs": [],
                        "terms": [], "numeric_anchors": []} for i in range(1, 5)]},
        ])
        qs = [f"问题{i}？" for i in range(1, 13)]
        plan = generate_plan(mock, qs)
        assert mock.calls == 2, "12 题应分 2 批（≤8/批）"
        assert len(plan.items) == 12

    def test_plan_from_dict_roundtrip(self):
        mock = MockLLM([PLAN_RESP])
        plan = generate_plan(mock, ["q1？", "q2？"])
        d = plan.to_dict()
        d["items"][0]["variants_selected"] = [0]      # 用户裁剪：只选第 1 个 variant
        d["items"][1]["selected"] = False              # 取消推理项
        p2 = plan_from_dict(d)
        assert p2.items[0].variants_selected == [0]
        assert p2.items[1].selected is False
        assert p2.items[0].directive_dict()["terms"] == ["资本充足率"]


class TestExecuteSkipsRewrite:
    def test_pre_directive_no_rewrite_call(self):
        """G1 回归：execute 注入指令后 pipeline 不得再调用重写 LLM。"""
        import os
        from policyrag.index.bm25 import BM25Index
        from policyrag.index.hybrid import HybridIndex
        from policyrag.index.numpy_store import NumpyVectorStore
        from policyrag.index.vectors import HashEmbedder
        from policyrag.ingest.structure import build_ast
        from policyrag.ingest.chunker import chunk_document
        from policyrag.models.document import DocAST
        from policyrag.models.retrieval import QueryRequest
        from policyrag.query.pipeline import PipelineDeps, QueryPipeline

        tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
        os.makedirs(tmp, exist_ok=True)
        doc = "第一章 总则\n\n第一条 为规范。\n\n第二条 资本充足率不得低于 8%。\n"
        nodes, _ = build_ast([(doc, 1)], tables=[])
        ast = DocAST(doc_id="办法", version="v1", title="测试办法", nodes=nodes)
        chunks = chunk_document(ast)
        registry = {c.chunk_id: c.text for c in chunks}
        metas = {c.chunk_id: {"text": c.text, "text_norm": c.text_norm,
                              "breadcrumb": c.breadcrumb, "doc_title": ast.title,
                              "version": ast.version, "kind": c.kind.value}
                 for c in chunks}
        embedder = HashEmbedder()
        bm25 = BM25Index(f"{tmp}/bm25")
        bm25.build(list(registry), list(registry.values()))
        vecs = NumpyVectorStore(f"{tmp}/vec", dim=embedder.dim)
        vecs.build(list(registry), embedder.embed(list(registry.values())))
        index = HybridIndex(bm25, vecs, embedder, metas)
        cid = next(c.chunk_id for c in chunks if "资本充足率" in c.text)

        draft = {"answer": f"资本充足率不得低于 8%。[{cid}]",
                 "citations": [{"chunk_id": cid, "quote": "资本充足率不得低于 8%",
                                "clause_path": "第二条"}],
                 "confidence": 0.95, "uncertainties": []}
        mock = MockLLM([draft])   # 只备草稿——若触发重写会耗掉它导致断言失败
        deps = PipelineDeps(index=index, llm=mock, registry=registry,
                            meta_lookup=metas, corpus_version="t")
        pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
        env = pipe.run(QueryRequest(
            query="资本充足率不得低于多少？",
            pre_directive={"intent": "quote", "clause_refs": ["第二条"],
                           "terms": ["资本充足率"], "numeric_anchors": ["8%"],
                           "doc_family": None},
            request_id="t_no_rewrite"))
        assert env.status.value == "answered", (env.status, env.refusal)
        assert mock.calls == 1, f"只应有 1 次 LLM 调用（草稿），实测 {mock.calls}——重写被意外触发"


class TestPresent:
    def test_strip_and_layer(self):
        from policyrag.service.present import present_for_user
        out = {
            "answer": f"资本充足率不得低于 8%。[aaaa1111bbbb2222]杠杆率不超过 6%。[cccc3333dddd4444]",
            "citations": [
                {"chunk_id": "aaaa1111bbbb2222", "quote": "资本充足率不得低于 8%",
                 "doc_title": "办法", "doc_version": "v1", "clause_path": "第二条",
                 "page": 3, "verified": True},
                {"chunk_id": "cccc3333dddd4444", "quote": "杠杆率不超过 6%",
                 "doc_title": "办法", "doc_version": "v1", "clause_path": "第二条",
                 "page": 3, "verified": True},
            ],
            "meta": {"per_question": []},
        }
        p = present_for_user(out)
        assert "[" not in p["answer"] or "aaaa" not in p["answer"]
        assert "8%" in p["answer"] and "6%" in p["answer"]
        assert "chunk_id" not in p["citations"][0], "display 层不带编码"
        assert p["citations"][0]["ref"]["chunk_id"] == "aaaa1111bbbb2222"
        assert p["line_refs"], "句↔引用对位存在"
        assert p["line_refs"][0]["cite_indexes"] == [0]

    def test_expose_ids_passthrough(self):
        from policyrag.service.present import present_for_user
        out = {"answer": "x。[aaaa1111bbbb2222]", "citations": [
            {"chunk_id": "aaaa1111bbbb2222", "quote": "x"}], "meta": {}}
        p = present_for_user(out, expose_ids=True)
        assert p is out

    def test_per_question_recursive(self):
        from policyrag.service.present import present_for_user
        out = {"answer": "", "citations": [], "meta": {"per_question": [
            {"answer": "答。[aaaa1111bbbb2222]", "citations": [
                {"chunk_id": "aaaa1111bbbb2222", "quote": "q"}], "meta": {}},
        ]}}
        p = present_for_user(out)
        sub = p["meta"]["per_question"][0]
        assert "aaaa" not in sub["answer"]
        assert sub["citations"][0]["ref"]["chunk_id"] == "aaaa1111bbbb2222"


def test_parse_item_mode_analyze_intent_fallback():
    """R2-08 F3：mode=analyze 的 retrieve 项 intent 默认 analyze（曾默认 quote 致 B5 按引文口径误判）。"""
    from policyrag.query.search_plan import _parse_item
    raw = {"qid": "1", "kind": "retrieve", "terms": [], "clause_refs": [],
           "numeric_anchors": [], "variants": []}
    it_a = _parse_item(dict(raw), "请分析差异。", mode="analyze")
    assert it_a.intent == "analyze"
    it_q = _parse_item(dict(raw), "上限是多少？", mode="quote")
    assert it_q.intent == "quote"
    # 显式 intent 优先于 mode 回落
    raw2 = dict(raw, intent="compare")
    it_c = _parse_item(raw2, "比较两者。", mode="quote")
    assert it_c.intent == "compare"
