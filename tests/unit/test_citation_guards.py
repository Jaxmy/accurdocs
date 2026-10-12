"""引用标记卫兵回归：伪造 ID 净化、证据包装剥离、公式词元包含层。"""

from policyrag.models.answer import Citation
from policyrag.service.present import present_for_user, strip_id_marks
from policyrag.verify.citations import (sanitize_citations, strip_unknown_cite_marks,
                                        verify_citations)


class TestStripUnknownMarks:
    def test_uuid_and_chunk_wrapper_stripped(self):
        t = ("差价合约电费按96点计算。[10436a2e-7d3c-4f5e-9b21-0f5a6c8e4a11]"
             "[chunk 4581989200ec821c 【公式】R = 96]")
        out = strip_unknown_cite_marks(t, set())
        assert "[" not in out, out
        assert "96点" in out

    def test_real_id_kept(self):
        t = "结论成立。[4581989200ec821c]"
        assert "[4581989200ec821c]" in strip_unknown_cite_marks(t, {"4581989200ec821c"})

    def test_fake_hex_stripped(self):
        t = "见[033f9f524e976bbf]。"
        assert "[" not in strip_unknown_cite_marks(t, {"4581989200ec821c"})


class TestSanitizeCitations:
    def test_fake_entries_dropped(self):
        cs = [Citation(chunk_id="4581989200ec821c", quote="a", doc_title="T", doc_version="v1", clause_path="第一条"),
              Citation(chunk_id="10436a2e-7d3c-4f5e-9b21-0f5a6c8e4a11", quote="b", doc_title="T", doc_version="v1", clause_path="第一条")]
        out = sanitize_citations(cs, {"4581989200ec821c": "a"})
        assert [c.chunk_id for c in out] == ["4581989200ec821c"]


class TestFormulaTokenTier:
    def test_reordered_formula_quote_passes(self):
        """公式块引用允许词元包含（转写形态差/求和序重排不整答拒掉）。"""
        registry = {"f1": "【公式】R阻塞 = ∑d ∑i ∑t [ Q日上网,i,t × ( LMP省内实时,i,t − LMP实时统一,t ) ]"}
        cs = [Citation(chunk_id="f1",
                       quote="R阻塞 = ∑t ∑i ∑d [ Q日上网,i,t × ( LMP省内实时,i,t − LMP实时统一,t ) ]",
                       doc_title="T", doc_version="v1", clause_path="第46条")]
        res = verify_citations(cs, registry, {"f1"})
        assert res.ok, res.failures

    def test_hallucinated_quote_still_rejected(self):
        registry = {"f1": "【公式】R阻塞 = ∑ Q × P"}
        cs = [Citation(chunk_id="f1", quote="完全无关的编造引文XYZ", doc_title="T", doc_version="v1", clause_path="")]
        res = verify_citations(cs, registry, {"f1"})
        assert not res.ok

    def test_unrelated_word_reorder_rejected(self):
        """词元包含层不放行低重合引文。"""
        registry = {"t1": "发电企业月度总电费由电能量电费和辅助服务费用构成。"}
        cs = [Citation(chunk_id="t1", quote="居民阶梯电价第一档电量上限", doc_title="T", doc_version="v1", clause_path="")]
        res = verify_citations(cs, registry, {"t1"})
        assert not res.ok


class TestPresentLayer:
    def test_strip_variants(self):
        t = ("答案。[4581989200ec821c] 又[10436a2e-7d3c-4f5e-9b21-0f5a6c8e4a11]"
             "[chunk ab12cd34ef567890 【公式】x=1]")
        out = strip_id_marks(t)
        assert "[" not in out, out

    def test_present_envelope_no_ids(self):
        env = {"answer": "结论成立。[4581989200ec821c] 见[10436a2e-7d3c-4f5e-9b21-0f5a6c8e4a11]",
               "citations": [{"display": {"doc_title": "T", "clause_path": "第一条", "page": 1},
                              "ref": {"chunk_id": "4581989200ec821c"}}]}
        out = present_for_user(env)
        assert "[" not in out["answer"], out["answer"]
