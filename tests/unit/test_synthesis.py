"""WS-C 综合推理引擎测试：文档锚定推导 + 推导闭包复算 + 引用禁令。"""

import json

from policyrag.llm.base import Caps, LLMResult
from policyrag.models.answer import Usage
from policyrag.query.synthesis import (
    parse_synthesis_text,
    run_synthesis,
    verify_synthesis,
)


class MockLLM:
    name = "mock"
    caps = Caps(json_mode=True)

    def __init__(self, text):
        self._text = text
        self.calls = 0

    def complete(self, messages, *, schema=None, max_tokens=4096,
                 temperature=0.0, effort="high", on_delta=None):
        self.calls += 1
        if on_delta:
            on_delta(self._text)
        return LLMResult(text=self._text, usage=Usage(output_tokens=10),
                         model="mock", stop_reason="end_turn")


BASIS = [
    {"qid": "1", "question": "上年不良率是多少？",
     "answer": "上年不良率为 9%。",
     "evidence": ["绿色金融的不良率上限是 3%，上年实际为 9%。"]},
    {"qid": "2", "question": "本年不良率是多少？",
     "answer": "本年不良率为 12%。",
     "evidence": ["本年不良率实际为 12%。"]},
]
BASIS_TEXTS = [f["answer"] for f in BASIS] + [e for f in BASIS for e in f["evidence"]]


class TestVerifySynthesis:
    def test_literal_ok(self):
        text = "上年不良率为 9%。（据问题 1）"
        cleaned, rep = verify_synthesis(text, BASIS_TEXTS)
        assert rep.ok and "9%" in cleaned

    def test_derived_pp_closure(self):
        """'提高 3 个百分点' = 12%−9%，可复算 → 通过（G4 防误杀）。"""
        text = "【推导】不良率较上年提高 3 个百分点。[basis:1,2]"
        cleaned, rep = verify_synthesis(text, BASIS_TEXTS)
        assert rep.ok, rep.failures
        assert rep.inference_ok >= 1

    def test_unmarked_derived_dropped(self):
        """可复算但未标【推导】→ 删句。"""
        text = "不良率较上年提高 3 个百分点。"
        cleaned, rep = verify_synthesis(text, BASIS_TEXTS)
        assert not rep.ok and rep.dropped
        assert "3 个百分点" not in cleaned

    def test_fabricated_number_dropped(self):
        text = "【推导】不良率将达到 37.5%。[basis:1]"
        cleaned, rep = verify_synthesis(text, BASIS_TEXTS)
        assert not rep.ok and rep.dropped

    def test_citation_ban(self):
        text = "依据原文。[aaaa1111bbbb2222]（据问题 1）"
        cleaned, rep = verify_synthesis(text, BASIS_TEXTS,
                                        basis_chunk_ids={"eeee9999ffff0000"})
        assert "aaaa1111bbbb2222" not in cleaned
        assert any("禁止新增原子编码" in f for f in rep.failures)

    def test_basis_citation_allowed(self):
        text = "依据原文。[eeee9999ffff0000]（据问题 1）"
        cleaned, rep = verify_synthesis(text, BASIS_TEXTS,
                                        basis_chunk_ids={"eeee9999ffff0000"})
        # 规则要求不新增编码——basis 内编码也被剥离（正文用（据问题 N））
        assert "eeee9999ffff0000" not in cleaned


class TestParse:
    def test_markers(self):
        text = "事实甲。（据问题 1）【推导】结论乙。[basis:1,2]"
        p = parse_synthesis_text(text)
        assert any("【推导】" not in f for f in p["factual_claims"])
        assert p["inferences"] and p["inferences"][0]["based_on"] == ["1", "2"]
        assert p["basis_questions"] == ["1", "2"]


class TestRunSynthesis:
    def test_zero_basis_refuses(self):
        r = run_synthesis(MockLLM("不该被问"), "分析影响", [])
        assert r["status"] == "rejected"
        assert "无已验证事实基础" in r["verification"]["failures"][0]

    def test_happy_path(self):
        text = "不良率呈上升趋势。（据问题 1、2）【推导】较上年提高 3 个百分点。[basis:1,2]"
        r = run_synthesis(MockLLM(text), "分析不良率变化趋势", BASIS)
        assert r["status"] in ("ok", "partial")
        assert r["text"]
        assert r["inferences"]

    def test_fabrication_rejected(self):
        text = "【推导】不良率将达到 99%。[basis:1]"
        r = run_synthesis(MockLLM(text), "分析影响", BASIS)
        assert r["status"] == "rejected" or not r["text"].strip()
