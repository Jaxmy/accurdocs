"""多问题拆分 + 合并 envelope 单元测试（批量提问链路）。"""

from policyrag.models.answer import (
    AnswerEnvelope,
    AnswerStatus,
    Citation,
    Refusal,
    Usage,
)
from policyrag.query.decompose import decompose_questions
from policyrag.service.core import merge_envelopes


class TestDecompose:
    def test_single_question(self):
        assert decompose_questions("资本充足率不得低于多少？") == ["资本充足率不得低于多少？"]

    def test_numbered_batch(self):
        t = "1. 资本充足率不得低于多少？\n2. 杠杆率不得超过多少？\n3. 信用贷款额度上限是多少？"
        qs = decompose_questions(t)
        assert len(qs) == 3
        assert "资本充足率" in qs[0] and "杠杆率" in qs[1] and "信用贷款" in qs[2]

    def test_numbered_with_preamble(self):
        t = "对比甲乙两文：\n1. 甲的资本充足率要求？\n2. 乙的资本充足率要求？"
        qs = decompose_questions(t)
        assert len(qs) == 2
        assert all(q.startswith("对比甲乙两文") for q in qs)

    def test_numbered_continuation_lines(self):
        t = "1. 必开成本补偿费用公式的\n求和限元怎么写？\n2. 阻塞费用公式是什么？"
        qs = decompose_questions(t)
        assert len(qs) == 2
        assert "求和限元" in qs[0]

    def test_inline_marks(self):
        qs = decompose_questions("资本充足率是多少？杠杆率上限是多少？")
        assert len(qs) == 2

    def test_subjectless_continuation_merged(self):
        """无主语续问并题（实测 bug："如何计算？"独立成题检索漂移）："""
        qs = decompose_questions("电力现货市场全天统一结算点加权均价的上限是怎么规定的？如何计算？")
        assert len(qs) == 1, qs
        assert "如何计算" in qs[0] and "上限" in qs[0]

    def test_subjectless_with_subject_not_merged(self):
        qs = decompose_questions("上限是怎么规定的？第二条如何规定？")
        assert len(qs) == 2, "有主语的（'第二条如何规定'）不许并题"

    def test_numbered_subjectless_gets_context_not_merged(self):
        qs = decompose_questions("1. 上限是多少？\n2. 如何计算？")
        assert len(qs) == 2, "编号列表尊重显式拆分"
        assert "相关前问" in qs[1], "但须附前问上下文"

    def test_anaphora_gets_context(self):
        t = "1. 存款利率上限是多少？\n2. 它的下限是多少？"
        qs = decompose_questions(t)
        assert len(qs) == 2
        assert "相关前问" in qs[1]

    def test_empty(self):
        assert decompose_questions("") == []

    def test_fullwidth_digits(self):
        t = "１、甲内容是什么？\n２、乙内容是什么？"
        qs = decompose_questions(t)
        assert len(qs) == 2


def _env(status, answer="", conf=0.9, request_id="r1", refusal=None, cites=None):
    return AnswerEnvelope(
        request_id=request_id, status=status, answer=answer, confidence=conf,
        citations=cites or [], refusal=refusal, usage=Usage(output_tokens=10),
    )


class TestMerge:
    def test_all_answered(self):
        c = Citation(chunk_id="c1", quote="q", doc_title="t", doc_version="v",
                     clause_path="第一条", verified=True)
        per = [("问一", _env(AnswerStatus.ANSWERED, "答一", 0.9, cites=[c])),
               ("问二", _env(AnswerStatus.ANSWERED, "答二", 0.7))]
        m = merge_envelopes(per, request_id="req_x", corpus_version="v1")
        assert m.status == AnswerStatus.ANSWERED
        assert "1. 问一\n答一" in m.answer and "2. 问二\n答二" in m.answer
        assert abs(m.confidence - 0.8) < 1e-9
        assert len(m.citations) == 1
        assert m.usage.output_tokens == 20
        assert m.meta["decomposed"] and m.meta["n_answered"] == 2

    def test_partial(self):
        per = [("问一", _env(AnswerStatus.ANSWERED, "答一", 0.9)),
               ("问二", _env(AnswerStatus.REFUSED, "",
                             0.0, refusal=Refusal(reason="insufficient_evidence",
                                                  detail="缺条款")))]
        m = merge_envelopes(per, request_id="req_x")
        assert m.status == AnswerStatus.PARTIAL
        assert "（拒答：insufficient_evidence）缺条款" in m.answer
        assert abs(m.confidence - 0.9) < 1e-9  # 拒答不稀释

    def test_all_refused(self):
        per = [("问一", _env(AnswerStatus.REFUSED, "", 0.0,
                            refusal=Refusal(reason="x", detail="y")))]
        m = merge_envelopes(per, request_id="req_x")
        assert m.status == AnswerStatus.REFUSED
        assert m.refusal is not None

    def test_dedupe_citations(self):
        c1 = Citation(chunk_id="c1", quote="q", doc_title="t", doc_version="v",
                      clause_path="p")
        c2 = Citation(chunk_id="c1", quote="q", doc_title="t", doc_version="v",
                      clause_path="p")
        per = [("q1", _env(AnswerStatus.ANSWERED, "a1", cites=[c1])),
               ("q2", _env(AnswerStatus.ANSWERED, "a2", cites=[c2]))]
        m = merge_envelopes(per, request_id="req_x")
        assert len(m.citations) == 1
