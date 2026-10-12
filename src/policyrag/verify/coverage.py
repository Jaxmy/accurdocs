"""数值覆盖硬闸（漏答拦截）：问数值必答数值——零 LLM 成本。

与 verify_numbers（错答拦截：答案数值必须溯源）互补，专治
"引用有数、答案无数"：问题询问数值、证据里有对应数值，但答案只复述
适用范围/条件/定义（l17 实测：引用命中 6MW/12MWh，答案却只写适用对象）。

规则：
1. 问题含数值疑问词（多少/几/数值/多大…）才启用（"几乎"不触发）；
2. 问题量纲（功率/容量/金额/比例/时间/数量…）约束所询数值的单位族——
   问"功率和容量"时答案命中"10千伏"（电压）不算答了数值；
3. 答案正文（chunk_id 引用标记除外）必须 canonical 命中证据数值 span；
   未命中 → 漏答，调用方触发一次补答重写；再失败按校验失败拒答。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingest.normalize import canonical_number, normalize
from .deterministic import _protected_ranges, _is_protected

__all__ = ["CoverageResult", "verify_numeric_coverage", "arm_gate"]

# 数值疑问词（"几"排除"几乎/几何"；"哪几类/哪几种"是枚举问不是问数值）
NUMERIC_Q_RE = re.compile(r"(多少|(?<![哪])几(?!乎|何)|数值|多大|多高|多长|多小|若干)")

# 数值 span：量纲单位尾（长单位优先）+ 中文数字单位 + 时间点/区间 + 日期
_UNIT = (
    r"兆瓦时|千瓦时|万?千瓦时|MWh|KWh|kWh|安时"
    r"|兆瓦|千瓦|万?千瓦|MW|KW|kW|瓦"
    r"|千伏|kV|KV|伏"
    r"|万元/次|亿元?/次|元/(?:兆瓦时|千瓦时|小时|次|千瓦|吨)|万元|亿元|元"
    r"|个百分点|百分点|基点|bp|BP"
    r"|个工作日|自然月|自然日|工作日|小时|分钟|秒|个月|年度|年|月|日|天"
    r"|段|次|档|级|家|倍|个点|点|个|种|类"
    r"|%|％"
)
_CN = r"[零〇一二两三四五六七八九十百千万]+"

_SPAN_RE = re.compile(
    r"\d{1,2}:\d{2}(?:\s*[-–—~至到]\s*\d{1,2}:\d{2})?"            # 4:00-19:00
    r"|\d{4}\s*[-年/]\s*\d{1,2}\s*[-月/]\s*\d{1,2}\s*日?"          # 日期
    r"|\d+\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|百分之" + _CN + r"(?:点" + _CN + r")?%?"                    # 百分之八
    r"|\d[\d,]*\s*\.\s*\d+(?:\s+\d+)*\s*[%％]?"         # 断裂小数 1665 .13（% 不丢）
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:万元|亿元|元)?"           # 1,000 万元
    r"|\d+(?:\.\d+)?\s*[-–—~至到]\s*\d+(?:\.\d+)?\s*(?:" + _UNIT + r")"  # 数值区间
    r"|" + _CN + r"\s*(?:" + _UNIT + r")"                          # 一年 / 五级
    r"|\d+(?:\.\d+)?\s*(?:" + _UNIT + r")"
    r"|\d+(?:\.\d+)?\s*[%％]"
    r"|\d+(?:\.\d+)?"
)

# span 单位尾 → 量纲（按首单位定类：元/兆瓦时 → 金额；MWh 优先于 MW）
_TYPE_RULES: list[tuple[str, str]] = [
    (r"兆瓦时|千瓦时|MWh|KWh|kWh|安时", "capacity"),
    (r"兆瓦|千瓦|MW|KW|kW|瓦", "power"),
    (r"千伏|kV|KV|伏", "voltage"),
    (r"万元|亿元|元", "money"),
    (r"个百分点|百分点|基点|bp|BP|[%％]", "ratio"),
    (r"个工作日|自然月|自然日|工作日|小时|分钟|秒|个月|年度|年|月|日|天|:\d{2}", "time"),
    (r"段|次|档|级|家|倍|个点|点|个|种|类", "count"),
]

# 问题量纲词（有序；多词命中 = 多量纲并集）
_QUESTION_TYPE_RULES: list[tuple[str, str]] = [
    (r"功率|出力|负荷", "power"),
    (r"容量|电量|能耗|用电量|荷电", "capacity"),
    (r"电压", "voltage"),
    (r"价格|费用|电价|金额|成本|经费|报酬|租金|票价|工资", "money"),
    (r"比例|利率|税率|费率|占比|百分|贴息|折扣|成数", "ratio"),
    (r"时间|期限|工期|周期|年限|时段|日期|几点|小时|分钟|天数|工作日|自然月", "time"),
    (r"数量|个数|段数|次数|档数|级数|人数|家数|几段|几档|几级|几家|几次|几倍|几个|个点", "count"),
]


@dataclass
class CoverageResult:
    ok: bool
    reason: str = ""
    missing_examples: list[str] = field(default_factory=list)  # 补答重写提示用
    demanded_types: list[str] = field(default_factory=list)
    armed: str = "armed"        # armed|soft|off（R2-A 三态）
    signal: str = "regex"       # regex|llm|cross|degraded
    partial: bool = False       # 命中部分 demanded（记缺项注记，永不因 partial 拒答）
    hit_count: int = 0
    demanded_count: int = 0


# 三态 arming（R2-A 真值表）——纯函数，单测穷举
QKIND_KINDS = ("enum", "numeric", "compare", "definition", "procedure", "analysis")
QKIND_DIMS = ("power", "capacity", "money", "ratio", "time", "count", "none")
QKIND_MODES = ("regex", "llm", "cross")


def arm_gate(llm_kind: str | None, regex_hit: bool, mode: str = "regex",
             dimension: str | None = None) -> tuple[str, str]:
    """覆盖闸三态 arming → (state, signal)。

    - mode=regex：无视 LLM（A/B 基线；LLM 网关挂了的兜底语义也在此）；
    - mode=llm：LLM 定 arming——numeric→armed（补正则漏报），enum/definition/
      procedure→soft（regex 误报类，永不因闸拒），compare→看 regex 或量纲，
      analysis→off；LLM 缺失 → 正则（degraded）；
    - mode=cross：两侧一致才 armed，非对称格降 soft/off（保守部署档）。

    拒答触发保持确定性：本函数只决定"闸是否启用"，demanded/命中始终是
    canonical span 比对；soft 态只记日志，永不因闸拒答。
    """
    if mode not in QKIND_MODES:
        mode = "regex"
    llm_ok = llm_kind in QKIND_KINDS
    if mode == "regex" or not llm_ok:
        # LLM 缺失（网关挂/旧 plan JSON/解析失败）→ 正则回退；非 regex 档记 degraded
        return ("armed" if regex_hit else "off"), ("regex" if mode == "regex" else "degraded")

    if mode == "llm":
        if llm_kind == "numeric":
            return "armed", "llm"
        if llm_kind in ("enum", "definition", "procedure"):
            return ("soft" if regex_hit else "off"), "llm"
        if llm_kind == "compare":
            if regex_hit or (dimension and dimension != "none"):
                return "armed", "llm"
            return "soft", "llm"
        # analysis
        return ("soft" if regex_hit else "off"), "llm"

    # cross：严格一致才 armed
    if llm_kind == "numeric":
        return ("armed" if regex_hit else "soft"), "cross"
    if llm_kind in ("enum", "definition", "procedure"):
        return ("off" if not regex_hit else "soft"), "cross"
    if llm_kind == "compare":
        return ("armed" if regex_hit else
                ("off" if (not dimension or dimension == "none") else "soft")), "cross"
    return ("soft" if regex_hit else "off"), "cross"


@dataclass
class _Span:
    text: str
    canonical: str
    unit_type: str | None
    context: str = ""   # 证据内 ±40 字语境（问题语境收窄用）


# 代词性"数量"（同一个节点/两个方案的"一个/两个"）不是数值——两侧都不许当命中
_PRONOMINAL_RE = re.compile(r"^[一二两几]?个$|^[这那哪每各某]个$|^一个$")
# 代词性量词短语："以一种方式购电"的"一种"不是所询数值（曾成唯一 demanded
# span → 枚举问被闸成漏答拒答）。判别：中文数字+种/类/款/式 且
# 后接方式/形式/方法/途径类泛指词 或 前邻以/这/那/某/同/每/只 等代词标记。
_CLASSIFIER_RE = re.compile(r"^[一二两几][种种类款式项条]$")
_PRONOMINAL_PRE = re.compile(r"[以这那某同每另仅只该]$")
_PRONOMINAL_POST = re.compile(r"^(方式|形式|方法|途径|模式|渠道)")


def _classify(span_text: str) -> str | None:
    """数值 span → 量纲（取数字后的单位尾按首单位定类；中文数字取其单位）。"""
    t = span_text.strip()
    cn = re.fullmatch(f"({_CN})\\s*({_UNIT})", re.sub(r"\s+", "", t))
    if cn:
        t = cn.group(2)                      # 一年 → 年 → time
    else:
        t = re.sub(r"^[\d,.:\s年月日\-–—~至到]+", "", t).strip() or t
    for pat, typ in _TYPE_RULES:
        if re.match(pat, t):
            return typ
    # 时间区间 4:00-19:00 / 日期残段
    if re.match(r"\d{1,2}:\d{2}", span_text):
        return "time"
    return None


def _extract(text: str) -> list[_Span]:
    nt = normalize(text)
    protected = _protected_ranges(nt.norm)
    out: list[_Span] = []
    for m in _SPAN_RE.finditer(nt.norm):
        if _is_protected(m.start(), m.end(), protected):
            continue
        raw = m.group(0)
        if _PRONOMINAL_RE.match(re.sub(r"\s+", "", raw)):
            continue
        # 代词性量词短语（"以一种方式"的"一种"）——非数值，不进 demanded/hits
        core = re.sub(r"\s+", "", raw)
        if _CLASSIFIER_RE.match(core):
            pre = nt.norm[max(0, m.start() - 6):m.start()]
            post = nt.norm[m.end():m.end() + 6]
            if _PRONOMINAL_PRE.search(pre) or _PRONOMINAL_POST.match(post):
                continue
        # 字母词内数字（B2C→"2"）不是数值
        if m.start() > 0:
            pre_ch = nt.norm[m.start() - 1]
            if pre_ch.isascii() and pre_ch.isalpha():
                continue
        # 中文数字单位（一年 → 1年）与常规 canonical 双轨
        cn = re.fullmatch(f"({_CN})\\s*({_UNIT})", raw)
        if cn:
            canon = canonical_number(cn.group(1)) + re.sub(r"\s+", "", cn.group(2))
        else:
            canon = re.sub(r"\s+", "", canonical_number(raw))
        ctx = nt.norm[max(0, m.start() - 20):m.end() + 20]
        out.append(_Span(text=raw, canonical=canon, unit_type=_classify(raw), context=ctx))
    return out


# 问题语境词：必答集合收窄到"问题点名实体/对象"附近数值（问 4 国对比不要求表内他国数值）
_Q_STOP = {"如何", "什么", "怎么", "怎样", "多少", "计算", "规定", "分别", "对比",
           "相比", "水平", "情况", "给出", "请", "分析", "以及", "还是", "是不是"}
# 功能词/标点切分：整句 Han 贪婪匹配曾把无标点问句收成 1 个整句 token，
# 语境收窄全程空转（S01"哪几类主体"整句成 token，"一种"未被收窄掉）
_Q_SPLIT_RE = re.compile(
    r"[的地是有了和与或及在对从到被把哪几多少什么请问为很更最吗呢吧啊"
    "、，。？！；：“”‘’（）()《》\\[\\]\\s…—\\-]"
    r"|以及|分别|进行|可以|应当|需要|是否|之一|时候|时间|那么|这个|那个|相关"
)


def _q_tokens(question: str) -> set[str]:
    toks: set[str] = set()
    for part in _Q_SPLIT_RE.split(question or ""):
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(r"[A-Za-z]{2,}", part):
            toks.add(part)
            continue
        if 2 <= len(part) <= 12:
            toks.add(part)
        if len(part) > 6:                     # 长段再取首尾短缀（中国/大卖场）
            toks.update({part[:2], part[:3], part[-2:], part[-3:]})
    return {t for t in toks if t not in _Q_STOP}


def verify_numeric_coverage(question: str, answer_text: str,
                            evidence_texts: list[str],
                            question_kind: str | None = None,
                            dimension: str | None = None,
                            mode: str = "regex",
                            pre_spans: list[dict] | None = None) -> CoverageResult:
    """问数值必答数值。详见模块 docstring。

    R2-A：question_kind/dimension（search_plan/B2 LLM 顺带输出）是覆盖闸第一信号，
    正则（NUMERIC_Q_RE/量纲规则）降级为回退与交叉校验；arming 三态见 arm_gate。
    R2-B：pre_spans 为入库预标注数值清单（ingest.numspan）——demanded 取
    **预标注 ∪ 运行时正则**（永远并集，预标注漏标不得缩小可命中池）。
    拒答触发保持确定性：只有 armed + 确定性 span 比对确认漏答（零命中）才可能拒。
    """
    regex_hit = bool(question) and bool(NUMERIC_Q_RE.search(question))
    armed, signal = arm_gate(question_kind, regex_hit, mode, dimension)
    if armed == "off":
        return CoverageResult(ok=True, armed=armed, signal=signal)
    if not question:
        return CoverageResult(ok=True, armed=armed, signal=signal)

    # 问题量纲集合；"X的多少"（问占比/倍数）补 ratio；LLM 量纲并入（交集过滤用）
    q_types = {t for pat, t in _QUESTION_TYPE_RULES if re.search(pat, question)}
    if "的多少" in question:
        q_types.add("ratio")
    llm_dim = dimension if dimension in QKIND_DIMS and dimension != "none" else None
    if llm_dim and not q_types:
        q_types = {llm_dim}

    ev_spans: list[_Span] = []
    for ev in evidence_texts or []:
        ev_spans.extend(_extract(ev))
    # R2-B 预标注并集（LLM 发现的断格/跨行数值，运行时正则可能抽不出）
    for ps in pre_spans or []:
        canon = str(ps.get("canonical") or "")
        txt = str(ps.get("text") or "")
        if not canon or not txt:
            continue
        ut = ps.get("unit_type")
        if ut not in ("power", "capacity", "voltage", "money", "ratio", "time", "count"):
            ut = None
        ev_spans.append(_Span(text=txt, canonical=canon, unit_type=ut,
                              context=str(ps.get("context_hint") or "")))
    if not ev_spans:
        return CoverageResult(ok=True, armed=armed, signal=signal)  # 证据无数值 → 无数值可要求

    # 问题语境收窄：多值表格/长证据只对"问题点名实体"邻近数值做必答要求
    # （问 4 国对比 → 只要求 4 国数值；表内他国数值不进必答清单，防补答提示被无关数值污染）
    q_toks = _q_tokens(question)
    if q_toks:
        scoped = [s for s in ev_spans if any(t in s.context for t in q_toks)]
        if scoped:
            ev_spans = scoped

    demanded = ev_spans
    if q_types:
        # 无型数值（裸数 2.40/25 类）与问题量纲同池——它们常正是所询数值
        # （曾把裸数值踢出 demanded、只留"231个/120个"类带单位干扰项 → 漏答误判）
        typed = [s for s in ev_spans
                 if s.unit_type in q_types or s.unit_type is None]
        if any(s.unit_type in q_types for s in ev_spans):
            demanded = typed
        else:
            q_types = set()             # 证据无对应量纲 → 退回通用规则

    ans_spans = _extract(answer_text)  # 内部已做引用标记保护
    ev_canons = {s.canonical for s in demanded}
    hits = [s for s in ans_spans if s.canonical in ev_canons]

    def _soft_result(partial: bool, missing: list[str]) -> CoverageResult:
        return CoverageResult(
            ok=True, armed=armed, signal=signal, partial=partial,
            missing_examples=missing, demanded_types=sorted(q_types),
            hit_count=len(hits), demanded_count=len(demanded),
        )

    if armed == "soft":
        # SOFT：只记日志/缺项提示，永不因闸拒（regex 误报类的保险档）
        missing = _examples([s for s in demanded if s.canonical
                             not in {h.canonical for h in hits}], q_types) if not hits else []
        return _soft_result(bool(missing), missing)

    if hits:
        if q_types:
            # 命中中至少一个与问题同量纲或无型（裸数）才算答了所询数值
            if any(s.unit_type in q_types or s.unit_type is None for s in hits):
                # 部分命中：仍 ok（仅零命中才可能拒），但记缺项注记（R2-A 阶梯）
                missing = _examples(
                    [s for s in demanded
                     if s.canonical not in {h.canonical for h in hits}
                     and (s.unit_type in q_types or s.unit_type is None)], q_types)
                return _soft_result(bool(missing), missing)
            return CoverageResult(
                ok=False,
                reason=("问题询问的数值与答案给出的数值量纲不符"
                        f"（需求纲 {sorted(q_types)}，答案命中 {sorted({s.unit_type for s in hits})}）"),
                missing_examples=_examples(demanded, q_types),
                demanded_types=sorted(q_types),
                armed=armed, signal=signal,
                hit_count=len(hits), demanded_count=len(demanded),
            )
        missing = _examples(
            [s for s in demanded if s.canonical not in {h.canonical for h in hits}], q_types)
        return _soft_result(bool(missing), missing)

    examples = _examples(demanded, q_types)
    return CoverageResult(
        ok=False,
        reason="问题询问数值但答案未写出证据中的数值（漏答）",
        missing_examples=examples,
        demanded_types=sorted(q_types),
        armed=armed, signal=signal,
        hit_count=0, demanded_count=len(demanded),
    )


def _examples(demanded: list[_Span], q_types: set[str]) -> list[str]:
    """补答提示用例：优先问题同量纲的带单位 span，去重后返回。"""
    typed = [s.text for s in demanded if s.unit_type in q_types and s.unit_type is not None]
    rest = [s.text for s in demanded if s.text not in typed]
    out: list[str] = []
    for t in typed + rest:
        key = re.sub(r"\s+", "", t)
        if key not in {re.sub(r"\s+", "", x) for x in out}:
            out.append(t)
    return out[:5]
