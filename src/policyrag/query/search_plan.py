"""检索方案生成（WS-B）：拆分 + 分类 + 结构化指令 + 扩展联想 → 可确认 SearchPlan。

两段式契约（G2）：plan（本模块，非流式廉价调用）→ execute（core 编排 + 流式）。
执行无状态红线（G1）：execute 必须携带完整 directive 注入 pipeline（跳过 B2
重写），否则 LLM 重写漂移让确认闸门形同虚设。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

from ..llm.base import ChatMessage, llm_call_json
from ..models.clause_norm import canon_clause_ref

__all__ = [
    "Variant", "PlanItem", "SearchPlan",
    "generate_plan", "plan_need_confirm", "plan_from_dict", "PLAN_PROMPT_VERSION",
]

PLAN_PROMPT_VERSION = "v2.1"

SYSTEM_PLAN = """你是检索方案规划器。把给定的编号问题逐个规划为可执行的检索/推理指令。只输出 JSON。

输出结构（外层固定为 items 数组——必须覆盖问题清单中的每一个问题，一个都不能少）：
{
  "items": [
    {
      "qid": "1",
      "kind": "retrieve",
      "intent": "quote",
      "question_kind": "numeric",
      "dimension": "money",
      "doc_family": null,
      "clause_refs": ["第X条"],
      "terms": ["规范化专业术语"],
      "numeric_anchors": ["原文形态数值，如 5%、1,000万元"],
      "specificity": "precise",
      "variants": [{"query": "改写/同义表述", "terms": [], "clause_refs": [],
                    "numeric_anchors": [], "confidence": 0.8,
                    "basis": "同义词|上位词|原文可能用语|缩写展开"}],
      "depends_on": [],
      "implicit_facts": []
    }
  ],
  "ambiguous": false,
  "clarifications": []
}

字段规则：
- kind：retrieve＝事实/数值/条款/条件/定义（多少、几、第X条、是什么、哪些、是否）；
  reason＝纯分析/推演/解释（评估影响、为何、如何理解、综合、结合上述）——不需要先从文档取事实的；
  混合问题必须拆成两个 item，reason 的 depends_on 指向对应 retrieve 的 qid。
  警示：含具体对象的事实询问（"A与B布局有何不同""X由哪几项构成""条件是什么"）即便带
  "请分析/评估"尾巴，也必须先拆 retrieve 项取回事实，再设 reason 项分析——整题划 reason
  会导致不检索、无输出。例："京东与苏宁仓储布局有何不同？请分析优劣"→
  item1 retrieve（布局差异事实）+ item2 reason（优劣分析，depends_on=[1]）。
- specificity：问题宽泛/需要面状信息 → "broad"，否则 "precise"。
- variants：只为可能存在同义/近义/缩写表述的问题生成，≤3 条，confidence≥0.5 降序；
  每条必须自带结构化槽位，不允许只给自然语言句子；术语大概率与文档一致时给 []。
- reason 项：depends_on 填显式题号（"综合1和2"）或默认其前面全部 retrieve 的 qid；
  无可依赖前置检索问题时，把回答所需的前置事实问题写进 implicit_facts。
- question_kind：问题的语义类型——enum＝列举/枚举（包括哪几类/哪些）；numeric＝索取具体数值
  （多少/几/给出数值，含无疑问词的求值如"请给出市值"）；compare＝对比两对象异同或数值高低；
  definition＝问定义/含义（是什么/指什么）；procedure＝问流程/步骤/时限/条件；analysis＝综合分析。
- dimension：所询数值的量纲（仅 numeric/compare 带意义）——power/capacity/money/ratio/time/count；
  不问数值或无量纲给 "none"。
- ambiguous：问题指向多义（术语多个可能对象/文档）时 true，clarifications 给 1-3 个澄清问句。

保留用户原话关键词在 terms；不确定的字段给 null/空数组，不要猜。"""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "qid": {"type": "string"},
                    "kind": {"type": "string", "enum": ["retrieve", "reason"]},
                    "intent": {"type": "string",
                               "enum": ["quote", "locate", "compare", "analyze"]},
                    "question_kind": {"type": "string",
                                      "enum": ["enum", "numeric", "compare",
                                               "definition", "procedure", "analysis"]},
                    "dimension": {"type": "string",
                                  "enum": ["power", "capacity", "money",
                                           "ratio", "time", "count", "none"]},
                    "doc_family": {"type": ["string", "null"]},
                    "clause_refs": {"type": "array", "items": {"type": "string"}},
                    "terms": {"type": "array", "items": {"type": "string"}},
                    "numeric_anchors": {"type": "array", "items": {"type": "string"}},
                    "specificity": {"type": "string", "enum": ["precise", "broad"]},
                    "variants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "terms": {"type": "array", "items": {"type": "string"}},
                                "clause_refs": {"type": "array", "items": {"type": "string"}},
                                "numeric_anchors": {"type": "array", "items": {"type": "string"}},
                                "confidence": {"type": "number"},
                                "basis": {"type": "string"},
                            },
                            "required": ["query", "confidence"],
                        },
                    },
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "implicit_facts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["qid", "kind", "clause_refs", "terms", "numeric_anchors"],
            },
        },
        "ambiguous": {"type": "boolean"},
        "clarifications": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["items"],
}

_MAX_VARIANTS = 3
_MIN_CONF = 0.5
_BATCH = 8   # G13：20 题批量一次调用会爆 token，按 ≤8 题/批分片


@dataclass
class Variant:
    query: str
    terms: list[str] = field(default_factory=list)
    clause_refs: list[str] = field(default_factory=list)
    numeric_anchors: list[str] = field(default_factory=list)
    confidence: float = 0.0
    basis: str = ""

    def to_dict(self) -> dict:
        return {"query": self.query, "terms": self.terms,
                "clause_refs": self.clause_refs,
                "numeric_anchors": self.numeric_anchors,
                "confidence": round(self.confidence, 3), "basis": self.basis}


@dataclass
class PlanItem:
    qid: str
    kind: str                       # retrieve | reason
    question: str
    intent: str = "quote"
    doc_family: str | None = None
    clause_refs: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    numeric_anchors: list[str] = field(default_factory=list)
    specificity: str = "precise"
    variants: list[Variant] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    implicit_facts: list[str] = field(default_factory=list)
    selected: bool = True
    variants_selected: list[int] = field(default_factory=list)
    question_kind: str | None = None   # enum|numeric|compare|definition|procedure|analysis
    dimension: str | None = None       # power|capacity|money|ratio|time|count|none

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "kind": self.kind, "question": self.question,
            "intent": self.intent, "doc_family": self.doc_family,
            "clause_refs": self.clause_refs, "terms": self.terms,
            "numeric_anchors": self.numeric_anchors,
            "specificity": self.specificity,
            "variants": [v.to_dict() for v in self.variants],
            "depends_on": self.depends_on,
            "implicit_facts": self.implicit_facts,
            "selected": self.selected,
            "variants_selected": self.variants_selected,
            "question_kind": self.question_kind,
            "dimension": self.dimension,
        }

    def directive_dict(self) -> dict:
        return {
            "intent": self.intent, "doc_family": self.doc_family,
            "clause_refs": self.clause_refs, "terms": self.terms,
            "numeric_anchors": self.numeric_anchors,
            "question_kind": self.question_kind,
            "dimension": self.dimension,
        }


@dataclass
class SearchPlan:
    plan_id: str
    items: list[PlanItem]
    clarifications: list[str] = field(default_factory=list)
    ambiguity: bool = False
    prompt_version: str = PLAN_PROMPT_VERSION
    degraded: str | None = None

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id, "items": [i.to_dict() for i in self.items],
            "clarifications": self.clarifications, "ambiguity": self.ambiguity,
            "prompt_version": self.prompt_version, "degraded": self.degraded,
        }

    def retrieve_items(self) -> list[PlanItem]:
        return [i for i in self.items if i.kind == "retrieve"]

    def reason_items(self) -> list[PlanItem]:
        return [i for i in self.items if i.kind == "reason"]


def plan_need_confirm(plan: SearchPlan) -> bool:
    """智能触发（用户决策）：扩展联想 / 澄清 / 含推理 / 宽泛 → 须确认。"""
    if plan.degraded:
        return False
    if plan.clarifications or plan.ambiguity:
        return True
    if plan.reason_items():
        return True
    for it in plan.items:
        if it.specificity == "broad":
            return True
        if it.variants:
            return True
    return False


def _norm_clause_refs(refs: list) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for r in refs or []:
        r = str(r).strip()
        if not r:
            continue
        canon = canon_clause_ref(r)
        key = canon or r
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _parse_item(raw: dict, question: str, mode: str = "quote") -> PlanItem:
    variants: list[Variant] = []
    for v in (raw.get("variants") or [])[:_MAX_VARIANTS * 2]:
        try:
            conf = float(v.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < _MIN_CONF or not (v.get("query") or "").strip():
            continue
        variants.append(Variant(
            query=str(v["query"]).strip(),
            terms=[str(t) for t in (v.get("terms") or [])],
            clause_refs=_norm_clause_refs(v.get("clause_refs")),
            numeric_anchors=[str(t) for t in (v.get("numeric_anchors") or [])],
            confidence=conf,
            basis=str(v.get("basis") or ""),
        ))
    variants.sort(key=lambda v: -v.confidence)
    variants = variants[:_MAX_VARIANTS]

    kind = raw.get("kind") if raw.get("kind") in ("retrieve", "reason") else "retrieve"
    qkind = raw.get("question_kind")
    if qkind not in ("enum", "numeric", "compare", "definition", "procedure", "analysis"):
        qkind = None                      # 非法/缺省 → None → 覆盖闸走正则回退
    dim = raw.get("dimension")
    if dim not in ("power", "capacity", "money", "ratio", "time", "count", "none"):
        dim = None
    return PlanItem(
        qid=str(raw.get("qid") or ""),
        kind=kind,
        question=question,
        # mode=analyze 的调用方意图优先于 kind 默认（R2-08：A05 曾被默认成 quote）
        intent=(raw.get("intent")
                or ("analyze" if kind == "reason" or mode == "analyze" else "quote")),
        doc_family=raw.get("doc_family"),
        clause_refs=_norm_clause_refs(raw.get("clause_refs")),
        terms=[str(t) for t in (raw.get("terms") or [])],
        numeric_anchors=[str(t) for t in (raw.get("numeric_anchors") or [])],
        specificity="broad" if raw.get("specificity") == "broad" else "precise",
        variants=variants,
        depends_on=[str(d) for d in (raw.get("depends_on") or [])],
        implicit_facts=[str(f) for f in (raw.get("implicit_facts") or [])],
        variants_selected=list(range(len(variants))),   # 默认全选
        question_kind=qkind,
        dimension=dim,
    )


def _scan_objects(text: str) -> list[dict]:
    """花括号平衡扫描：收集文本中所有含 qid 的平铺 item 对象（模型漏外层包装的兜底）。"""
    out: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            for j in range(i, n):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[i:j + 1])
                        except json.JSONDecodeError:
                            obj = None
                        if isinstance(obj, dict) and obj.get("qid"):
                            out.append(obj)
                        i = j
                        break
        i += 1
    return out


def _items_from_result(result) -> tuple[list[dict], dict]:
    """LLM 结果 → item 原始列表（防御：外层包装缺失/对象平铺均可解析）。"""
    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    raw = parsed.get("items")
    if isinstance(raw, list) and raw:
        return raw, parsed
    # 模型漏外层包装：优先全文扫描（可能平铺多个 item），再退 parsed 单对象
    scanned = _scan_objects(result.text or "")
    if scanned:
        return scanned, parsed
    if parsed.get("qid"):
        return [parsed], parsed
    return [], parsed


def plan_from_dict(d: dict) -> SearchPlan:
    """前端裁剪后的方案 JSON → SearchPlan（execute 无状态重建）。"""
    items: list[PlanItem] = []
    for raw in d.get("items") or []:
        it = _parse_item(raw, str(raw.get("question") or ""))
        it.qid = str(raw.get("qid") or it.qid)
        it.selected = bool(raw.get("selected", True))
        it.variants_selected = [int(x) for x in (raw.get("variants_selected") or [])]
        items.append(it)
    return SearchPlan(
        plan_id=str(d.get("plan_id") or f"plan_{uuid.uuid4().hex[:10]}"),
        items=items,
        clarifications=[str(c) for c in (d.get("clarifications") or [])],
        ambiguity=bool(d.get("ambiguity")),
        degraded=d.get("degraded"),
    )


def generate_plan(llm, questions: list[str], as_of: str = "",
                  lib_hints: str = "", on_progress=None,
                  mode: str = "quote") -> SearchPlan:
    """拆分后的子问题 → SearchPlan（≤8 题/批 LLM 调用；失败降级空指令）。"""
    plan_id = f"plan_{uuid.uuid4().hex[:10]}"
    items: list[PlanItem] = []
    clarifications: list[str] = []
    ambiguity = False
    degraded: str | None = None

    for start in range(0, len(questions), _BATCH):
        batch = questions[start:start + _BATCH]
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(batch))
        user = f"as_of={as_of}\n" + (f"文档库提示：{lib_hints}\n" if lib_hints else "") \
            + (f"调用方模式：mode={mode}（analyze 时 intent 倾向 analyze）\n"
               if mode == "analyze" else "") \
            + f"问题清单：\n{numbered}"
        result = llm_call_json(
            llm,
            [ChatMessage(role="system", content=SYSTEM_PLAN, cache=True),
             ChatMessage(role="user", content=user)],
            schema=PLAN_SCHEMA, max_tokens=2048, effort="low",
        )
        parsed = result.parsed if isinstance(result.parsed, dict) else {}
        raw_items, parsed = _items_from_result(result)
        if result.degraded or not raw_items:
            degraded = degraded or f"plan_{result.degraded or 'empty'}"
        by_qid = {str(r.get("qid")): r for r in raw_items if r.get("qid")}
        for i, q in enumerate(batch):
            qid = str(i + 1)
            raw = by_qid.get(qid) or by_qid.get(str(start + i + 1)) or {}
            item = _parse_item(raw, q, mode=mode)
            item.qid = qid
            if not raw:
                item.kind = "retrieve"
                degraded = degraded or "plan_item_missing"
            items.append(item)
        clarification = parsed.get("clarifications") or []
        clarifications += [str(c) for c in clarification]
        ambiguity = ambiguity or bool(parsed.get("ambiguous"))
        if on_progress:
            on_progress(start + len(batch), len(questions))

    # reason 项默认依赖：其前面全部 retrieve（无显式 depends_on 时）
    seen_retrieve: list[str] = []
    for it in items:
        if it.kind == "retrieve":
            seen_retrieve.append(it.qid)
        elif it.kind == "reason" and not it.depends_on:
            it.depends_on = list(seen_retrieve)

    # 确定性护栏（#8）：reason 项内含"事实问句？+ 分析祈使"的混合题——
    # 整题划 reason 会不检索、synthesis 无依据、全无输出。拆回 retrieve+reason。
    items = _split_mixed_reason(items)

    return SearchPlan(plan_id=plan_id, items=items,
                      clarifications=clarifications[:5], ambiguity=ambiguity,
                      degraded=degraded)


_MIXED_CUE_RE = re.compile(r"^(.*?[？?])\s*((?:请|试|并请|分析|评估|对比|说明|解释|结合).+)$")


def _split_mixed_reason(items: list[PlanItem]) -> list[PlanItem]:
    """混合题护栏：reason 项问题 = 事实问句？+ 分析祈使 → 拆 retrieve（前半）+ reason（后半）。"""
    out: list[PlanItem] = []
    for it in items:
        m = _MIXED_CUE_RE.match(it.question.strip()) if it.kind == "reason" else None
        if not m or it.implicit_facts:
            out.append(it)
            continue
        fact_q, anal_q = m.group(1).strip(), m.group(2).strip()
        shared = dict(doc_family=it.doc_family, clause_refs=list(it.clause_refs),
                      terms=list(it.terms), numeric_anchors=list(it.numeric_anchors),
                      specificity=it.specificity, selected=it.selected)
        fact_item = PlanItem(
            qid=it.qid, kind="retrieve", question=fact_q,
            intent=it.intent, variants=it.variants, depends_on=[],
            implicit_facts=[],
            question_kind=it.question_kind, dimension=it.dimension,   # 数值需求归事实半边
            **shared)
        reason_item = PlanItem(
            qid=it.qid + "r", kind="reason", question=anal_q,
            intent="analyze", variants=[], depends_on=[it.qid],
            implicit_facts=[],
            question_kind="analysis", dimension="none",  # 分析半边强制不 armed（R2-A）
            **shared)
        out.append(fact_item)
        out.append(reason_item)
    return out
