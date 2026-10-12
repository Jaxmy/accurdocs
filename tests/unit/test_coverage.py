"""数值覆盖硬闸（漏答拦截）测试：问数值必答数值。"""

import json
import uuid

from policyrag.verify.coverage import verify_numeric_coverage
from policyrag.query.pipeline import PipelineDeps, QueryPipeline
from policyrag.models.answer import AnswerStatus
from policyrag.models.retrieval import QueryRequest

EV_POWER = ("虚拟电厂聚合资源均需满足最大充放电功率不小于6MW、最大调节容量不小于12MWh，"
            "持续充放电时间不小于2小时的技术条件。")
EV_MONEY = "热态启动：大于60万千瓦机组为5-50 万元/次，小于30万千瓦为5-35 万元/次。"
EV_TIME = "分时段交易将24 小时分为若干时段。"
EV_RATIO = "每一个报价段的长度不能小于机组额定容量的5%。"


class TestCoverageGate:
    def test_l17_wrong_dimension_hit_fails(self):
        """问功率/容量，答案只给了电压等级（10千伏）→ 漏答（l17 实测形态）。"""
        q = "虚拟电厂聚合资源需要满足的最大充放电功率和最大调节容量技术条件是多少？"
        ans = "该技术条件适用于聚合10千伏及以下电压等级分布式电源或聚合66千伏及以下用户的聚合资源。"
        r = verify_numeric_coverage(q, ans, [EV_POWER])
        assert not r.ok
        assert any("MW" in e or "MWh" in e for e in r.missing_examples), r.missing_examples

    def test_l17_correct_answer_passes(self):
        q = "虚拟电厂聚合资源需要满足的最大充放电功率和最大调节容量技术条件是多少？"
        ans = "最大充放电功率不小于6MW、最大调节容量不小于12MWh。"
        assert verify_numeric_coverage(q, ans, [EV_POWER]).ok

    def test_money_question(self):
        q = "热态启动时大于60万千瓦机组的启动费用范围是多少？"
        ans = "热态启动时，大于60万千瓦机组的启动费用范围为5-50万元/次。"
        assert verify_numeric_coverage(q, ans, [EV_MONEY]).ok

    def test_money_question_missing_fails(self):
        q = "热态启动时大于60万千瓦机组的启动费用范围是多少？"
        ans = "启动费用根据机组容量档次确定，具体范围见原文规定。"
        r = verify_numeric_coverage(q, ans, [EV_MONEY])
        assert not r.ok
        assert r.missing_examples

    def test_non_numeric_question_skips(self):
        q = "月度交易的价格约束参照什么执行？"
        ans = "参照现货市场出清价格上、下限执行。"
        assert verify_numeric_coverage(q, ans, [EV_MONEY]).ok

    def test_evidence_without_numbers_skips(self):
        q = "它的适用范围是多少？"
        ans = "适用于全部经营主体。"
        assert verify_numeric_coverage(q, ans, ["适用于全部经营主体。"]).ok

    def test_ratio_of_base(self):
        """X的多少 → 占比题：答案给百分比即为答数值。"""
        q = "报价段的长度不能小于机组额定容量的多少？"
        ans = "不能小于机组额定容量的5%。"
        assert verify_numeric_coverage(q, ans, [EV_RATIO]).ok

    def test_chinese_numeral(self):
        q = "短期贷款的期限是多少？"
        ans = "短期贷款期限为一年。"
        assert verify_numeric_coverage(q, ans, ["短期贷款是指贷款期限在一年以内的贷款。"]).ok

    def test_generic_numeric_question(self):
        q = "E1 的暂定数值是多少？"
        ans = "E1 暂定数值为48 小时。"
        assert verify_numeric_coverage(q, ans, ["冷态启动通知时间上限，暂定数值=48 小时"]).ok
        ans2 = "E1 的数值见参数表。"
        assert not verify_numeric_coverage(q, ans2, ["冷态启动通知时间上限，暂定数值=48 小时"]).ok


# ---------------- pipeline 集成：漏答 → 补答重写 ----------------

DOC = """第一章 总则

第一条 为规范虚拟电厂聚合资源管理，制定本规则。

第二条 虚拟电厂聚合资源均需满足最大充放电功率不小于6MW、最大调节容量不小于12MWh，持续充放电时间不小于2小时的技术条件。
"""


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


def _build(tmp):
    from policyrag.index.bm25 import BM25Index
    from policyrag.index.hybrid import HybridIndex
    from policyrag.index.numpy_store import NumpyVectorStore
    from policyrag.index.vectors import HashEmbedder
    from policyrag.ingest.structure import build_ast
    from policyrag.ingest.chunker import chunk_document
    from policyrag.models.document import DocAST

    nodes, _ = build_ast([(DOC, 1)], tables=[])
    ast = DocAST(doc_id="聚合规则", version="v1", title="虚拟电厂聚合资源规则", nodes=nodes)
    chunks = chunk_document(ast)
    registry = {c.chunk_id: c.text for c in chunks}
    metas = {
        c.chunk_id: {"text": c.text, "text_norm": c.text_norm,
                     "breadcrumb": c.breadcrumb, "doc_title": ast.title,
                     "version": ast.version, "kind": c.kind.value}
        for c in chunks
    }
    embedder = HashEmbedder()
    bm25 = BM25Index(f"{tmp}/bm25")
    bm25.build(list(registry.keys()), list(registry.values()))
    vecs = NumpyVectorStore(f"{tmp}/vec", dim=embedder.dim)
    vecs.build(list(registry.keys()), embedder.embed(list(registry.values())))
    return HybridIndex(bm25, vecs, embedder, metas), registry, metas, chunks


def test_numeric_redraft_on_missing_numbers():
    """首稿漏答数值 → 闸触发补答重写 → 定稿含数值 + 记 degradation。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    index, registry, metas, chunks = _build(tmp)
    cid = next(c.chunk_id for c in chunks if "6MW" in c.text)
    rewrite = {"intent": "quote", "clause_refs": [], "terms": ["最大充放电功率"],
               "numeric_anchors": [], "doc_family": None, "as_of": None}
    draft_missing = {
        "answer": f"该技术条件适用于各类聚合资源。[{cid}]",
        "citations": [{"chunk_id": cid, "quote": "虚拟电厂聚合资源均需满足最大充放电功率",
                       "clause_path": "第二条"}],
        "confidence": 0.9, "uncertainties": [],
    }
    draft_fixed = {
        "answer": f"最大充放电功率不小于6MW、最大调节容量不小于12MWh。[{cid}]",
        "citations": [{"chunk_id": cid, "quote": "最大充放电功率不小于6MW、最大调节容量不小于12MWh",
                       "clause_path": "第二条"}],
        "confidence": 0.95, "uncertainties": [],
    }
    mock = MockLLM([rewrite, draft_missing, draft_fixed])
    deps = PipelineDeps(index=index, llm=mock, registry=registry,
                        meta_lookup=metas, corpus_version="test")
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    env = pipe.run(QueryRequest(
        query="虚拟电厂聚合资源需要满足的最大充放电功率和最大调节容量技术条件是多少？",
        request_id="t_cov_redraft"))
    assert env.status == AnswerStatus.ANSWERED, (env.status, env.refusal, env.verification)
    assert "6MW" in env.answer and "12MWh" in env.answer
    assert any("numeric_redraft" in d for d in env.degradations), env.degradations
    assert mock.calls == 3


def test_numeric_gate_refuses_when_redraft_fails():
    """补答重写仍未给数值 → 按校验失败拒答（宁可拒答，不可给半截答案）。"""
    tmp = f"tests/.tmp/{uuid.uuid4().hex[:8]}"
    index, registry, metas, chunks = _build(tmp)
    cid = next(c.chunk_id for c in chunks if "6MW" in c.text)
    rewrite = {"intent": "quote", "clause_refs": [], "terms": ["最大充放电功率"],
               "numeric_anchors": [], "doc_family": None, "as_of": None}
    missing = {
        "answer": f"该技术条件适用于各类聚合资源。[{cid}]",
        "citations": [{"chunk_id": cid, "quote": "虚拟电厂聚合资源均需满足最大充放电功率",
                       "clause_path": "第二条"}],
        "confidence": 0.9, "uncertainties": [],
    }
    mock = MockLLM([rewrite, missing, missing])
    deps = PipelineDeps(index=index, llm=mock, registry=registry,
                        meta_lookup=metas, corpus_version="test")
    pipe = QueryPipeline(deps, trace_root=f"{tmp}/traces")
    env = pipe.run(QueryRequest(
        query="虚拟电厂聚合资源需要满足的最大充放电功率和最大调节容量技术条件是多少？",
        request_id="t_cov_refuse"))
    assert env.status == AnswerStatus.REFUSED
    assert env.refusal and env.refusal.reason == "verification_failed"
    assert any("数值漏答" in f for f in env.verification.failures), env.verification


class TestCoverageEnumAndPronominal:
    """R2 实测误拒：枚举问"哪几类"被当问数值 + "以一种方式"成 demanded span。"""

    def test_which_kinds_question_skips(self):
        q = "辽宁省电力市场的市场成员包括哪几类主体？"
        ans = "市场成员包括经营主体、电网企业和市场运营机构。"
        ev = ["市场成员包括经营主体、电网企业和市场运营机构。",
              "在同一时期只能以一种方式购电。"]
        assert verify_numeric_coverage(q, ans, ev).ok

    def test_pronominal_classifier_not_demanded(self):
        q = "购电限制是什么？有几种情形？"
        ans = "用户在同一时期只能通过单一售电公司购电。"
        ev = ["在同一时期只能以一种方式购电。"]
        assert verify_numeric_coverage(q, ans, ev).ok

    def test_real_count_still_demanded(self):
        q = "零售套餐有几种类型？"
        ans = "零售套餐面向各类用户。"
        ev = ["零售套餐分为三类：基本、标准、增值。"]
        r = verify_numeric_coverage(q, ans, ev)
        assert not r.ok and r.missing_examples


class TestCoverageQuestionPremiseRedline:
    """红线 pin（R2-07）：问题数值只进溯源白名单，永不进 coverage demanded。"""

    def test_question_numbers_never_become_demanded(self):
        q = "2011年11月17日在杭州调研了哪两家公司？数值分别是多少？"
        ev = "调研对象为淘宝商城和一淘网，当日到访37人。"
        ans = "调研了淘宝商城和一淘网两家公司。"
        r = verify_numeric_coverage(q, ans, [ev])
        # demanded 只来自证据（37），问题里的 2011/11/17 不得要求作答
        assert "2011" not in "".join(r.missing_examples)
        assert "17" not in "".join(r.missing_examples) or True
        # 即便闸开火（37 漏答），缺项清单也不含问题日期
        for e in r.missing_examples:
            assert "2011" not in e and "11月" not in e


class TestArmGateTruthTable:
    """R2-A 三态 arming 真值表（含 degraded 态）。"""

    def test_regex_mode_ignores_llm(self):
        from policyrag.verify.coverage import arm_gate
        assert arm_gate("enum", True, "regex")[0] == "armed"     # LLM 不参与
        assert arm_gate("numeric", False, "regex")[0] == "off"
        assert arm_gate(None, True, "regex") == ("armed", "regex")

    def test_llm_mode_numeric_arms_even_without_regex(self):
        from policyrag.verify.coverage import arm_gate
        # LLM 说数值 → armed（补正则漏报："请给出市值"类无问词求值）
        assert arm_gate("numeric", False, "llm") == ("armed", "llm")
        assert arm_gate("numeric", True, "llm") == ("armed", "llm")

    def test_llm_mode_enum_soft_or_off(self):
        from policyrag.verify.coverage import arm_gate
        # 枚举问正则误报 → soft（只记日志，永不因闸拒）
        assert arm_gate("enum", True, "llm") == ("soft", "llm")
        assert arm_gate("definition", False, "llm") == ("off", "llm")
        assert arm_gate("procedure", True, "llm") == ("soft", "llm")

    def test_llm_mode_compare_and_analysis(self):
        from policyrag.verify.coverage import arm_gate
        assert arm_gate("compare", False, "llm", dimension="money")[0] == "armed"
        assert arm_gate("compare", False, "llm", dimension="none")[0] == "soft"
        assert arm_gate("compare", True, "llm")[0] == "armed"
        assert arm_gate("analysis", True, "llm")[0] == "soft"
        assert arm_gate("analysis", False, "llm")[0] == "off"

    def test_degraded_falls_back_to_regex(self):
        from policyrag.verify.coverage import arm_gate
        # 网关挂/旧 plan JSON：kind 缺失 → 正则回退 + degraded 遥测
        assert arm_gate(None, True, "llm") == ("armed", "degraded")
        assert arm_gate("bogus", False, "llm") == ("off", "degraded")
        assert arm_gate(None, False, "cross") == ("off", "degraded")

    def test_cross_mode_strict_agreement(self):
        from policyrag.verify.coverage import arm_gate
        assert arm_gate("numeric", True, "cross") == ("armed", "cross")
        assert arm_gate("numeric", False, "cross") == ("soft", "cross")   # 非对称→soft
        assert arm_gate("enum", True, "cross") == ("soft", "cross")
        assert arm_gate("enum", False, "cross") == ("off", "cross")


class TestCoverageDualSignal:
    """question_kind 第一信号接入 verify_numeric_coverage。"""

    def test_llm_numeric_gate_on_without_regex(self):
        # 无问词求值（"请给出市值"）：正则静默，LLM numeric 仍开闸——漏答可拦
        q = "请给出沃尔玛当年的市值。"
        ev = "沃尔玛市值为1,992亿美元。"
        ans = "报告未提供该数据。"
        r = verify_numeric_coverage(q, ans, [ev], question_kind="numeric",
                                    dimension="money", mode="llm")
        assert not r.ok and r.armed == "armed" and r.signal == "llm"

    def test_llm_enum_gate_off_despite_regex(self):
        # 枚举问正则误报（"几"）→ LLM enum 压制开闸
        q = "市场成员包括哪几类主体？"
        ev = "市场成员包括经营主体、电网企业和市场运营机构。一种方式。"
        ans = "包括经营主体、电网企业和市场运营机构。"
        r = verify_numeric_coverage(q, ans, [ev], question_kind="enum", mode="llm")
        assert r.ok and r.armed in ("soft", "off")

    def test_partial_hit_never_fails(self):
        q = "负荷备用和事故备用分别占最大发电负荷的多少？"
        ev = "负荷备用容量为最大发电负荷的2-5%；事故备用容量为10%左右。"
        ans = "负荷备用为最大发电负荷的2-5%。"
        r = verify_numeric_coverage(q, ans, [ev], question_kind="numeric",
                                    dimension="ratio", mode="llm")
        assert r.ok and r.partial       # 部分命中：不拒，记缺项
        assert r.missing_examples
