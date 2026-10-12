"""per_question 题面序归位回归（#13）：与题面一一对应、reason 项不缺行、事实子问不混入。"""

from policyrag.models.answer import AnswerEnvelope, AnswerStatus
from policyrag.models.retrieval import QueryRequest
from policyrag.service.core import CoreService


class _FakeLLM:
    name = "fake"
    caps = type("C", (), {"structured_output": False, "json_mode": True})()

    def complete(self, *a, **k):
        raise AssertionError("不应调用 LLM（plan 显式注入 + _ask_one/run_synthesis 打桩）")


def _plan():
    from policyrag.query.search_plan import PlanItem, SearchPlan
    items = [
        PlanItem(qid="1", kind="retrieve",
                 question="发电企业月度总电费如何构成？"),
        PlanItem(qid="2", kind="reason",
                 question="请分析其与发电主体的差异。",
                 implicit_facts=["售电公司月度总电费构成？"]),
        PlanItem(qid="3", kind="retrieve",
                 question="B2C综合模式的论据是什么？"),
    ]
    return SearchPlan(plan_id="p1", items=items)


def test_per_question_face_order(monkeypatch):
    svc = CoreService.__new__(CoreService)
    svc.llm = _FakeLLM()
    svc.meta = None
    monkeypatch.setattr(CoreService, "_corpus_version", lambda self: "t")
    monkeypatch.setattr(CoreService, "_lib_hints", lambda self, req: "")

    answered = []

    def fake_ask_one(self, sub, emit=None, require_review=False):
        answered.append(sub.query)
        return AnswerEnvelope(request_id=sub.request_id, status=AnswerStatus.ANSWERED,
                              answer=f"ans::{sub.query[:8]}", confidence=0.9)

    monkeypatch.setattr(CoreService, "_ask_one", fake_ask_one)

    import policyrag.query.synthesis as S

    def fake_synthesis(llm, question, basis_facts, on_delta=None):
        return {"status": "ok", "text": "syn::分析", "basis_questions": [],
                "inferences": [], "verification": {"failures": []}}

    monkeypatch.setattr(S, "run_synthesis", fake_synthesis)

    plan = _plan()
    env = svc._ask_impl(QueryRequest(query="批量问", request_id="t_order"), plan=plan)

    qs = env.meta["questions"]
    pq = env.meta["per_question"]
    assert len(qs) == 3 and len(pq) == 3, (qs, len(pq))
    # 题面序：Q1 retrieve、Q2 reason（此前整题消失）、Q3 retrieve
    assert qs[0].startswith("发电企业") and qs[1].startswith("请分析") and qs[2].startswith("B2C")
    assert pq[0]["answer"].startswith("ans::")
    assert pq[1].get("synthesis") and pq[1]["answer"].startswith("syn::"), pq[1]
    assert pq[2]["answer"].startswith("ans::")
    # 隐式事实子问不进对账单
    assert not any("售电公司" in q for q in qs)
    # 对账单编号按题面序且含 reason 行（缺答不可见即事故）
    assert env.answer.startswith("1. 发电企业")
    assert "2. 请分析" in env.answer and "syn::分析" in env.answer
    assert "3. B2C" in env.answer


def test_reason_rejected_still_visible(monkeypatch):
    """synthesis 被拒也要在对账单留行（拒答原因可见）。"""
    svc = CoreService.__new__(CoreService)
    svc.llm = _FakeLLM()
    svc.meta = None
    monkeypatch.setattr(CoreService, "_corpus_version", lambda self: "t")
    monkeypatch.setattr(CoreService, "_lib_hints", lambda self, req: "")
    monkeypatch.setattr(CoreService, "_ask_one",
                        lambda self, sub, emit=None, require_review=False:
                        AnswerEnvelope(request_id=sub.request_id,
                                       status=AnswerStatus.ANSWERED,
                                       answer="ans", confidence=0.9))
    import policyrag.query.synthesis as S
    monkeypatch.setattr(S, "run_synthesis",
                        lambda llm, q, b, on_delta=None: {"status": "rejected",
                                                           "text": "", "verification": {}})
    from policyrag.query.search_plan import PlanItem, SearchPlan
    plan = SearchPlan(plan_id="p2", items=[
        PlanItem(qid="1", kind="retrieve", question="构成？"),
        PlanItem(qid="2", kind="reason", question="请分析影响。"),
    ])
    env = svc._ask_impl(QueryRequest(query="q", request_id="t_rej"), plan=plan)
    pq = env.meta["per_question"]
    assert len(pq) == 2
    assert pq[1]["status"] == "refused"
    assert "分析推理未通过校验" in env.answer


def test_split_mixed_reason_constructs():
    """#8 护栏的 PlanItem 构造回归（曾用不存在的 directive= 字段，首次命中即崩）。"""
    from policyrag.query.search_plan import PlanItem, _split_mixed_reason
    it = PlanItem(qid="2", kind="reason",
                  question="京东与苏宁仓储布局有何不同？请分析两种供应链路径的优劣。")
    out = _split_mixed_reason([it])
    assert [o.kind for o in out] == ["retrieve", "reason"]
    assert out[0].question.endswith("？")
    assert out[1].depends_on == ["2"] and out[1].qid == "2r"
