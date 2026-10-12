"""证据侧数值枚举入库预标注（R2-B）：LLM 提议、确定性处置。

形状（计划 §三）：
- LLM 对 table/chart 块枚举数值 span（含断格/跨行数值、行列归属 context_hint）；
- **校验不走正则重抽**（否则 LLM 召回被正则封顶）：合法性 = normalize(text)
  折叠空白后子串命中 + canonical_number 可计算——LLM 编造的 span 直接丢弃；
- 落 chunk meta_json.numeric_spans: [{text, canonical, unit_type, context_hint}]；
- 执法侧（coverage）demanded = 预标注 ∪ 运行时正则，永远并集——预标注漏标
  最坏导致闸漏拦（保守方向），不产生误拒。
"""

from __future__ import annotations

import hashlib
import json
import re

from .normalize import canonical_number, normalize

__all__ = ["annotate_numeric_spans", "validate_spans", "numspan_cache_key",
           "NUMSPAN_PROMPT"]

NUMSPAN_PROMPT = """你是数值清单标注员。列出下列表格/图表文本中出现的**全部具体数值**（含单位、百分比、金额、数量、时间、断格拼合值）。只输出 JSON：
{"spans": [{"text": "原文形态数值（照抄文本片段）", "context_hint": "行/列归属或所属对象，如 亚马逊|市值 或 格力空调|京东价"}]}

规则：
- text 必须是原文的逐字片段（允许把断格/换行拼回，如 "1,992 亿"）；
- 不要解释、不要改写、不要补全原文没有的数值；数值必须在文本中原样出现；
- 表格行首对象名 + 列头拼进 context_hint（如 "格力空调|京东商城价"）。"""

_UNIT_TAIL = re.compile(
    r"(兆瓦时|千瓦时|MWh|kWh|安时|兆瓦|千瓦|MW|kW|千伏|kV"
    r"|万元|亿元|元|%|％|个百分点"
    r"|个工作日|小时|分钟|个月|年|月|日|天|段|次|档|级|家|倍|个|种|类|X|x)$"
)


def _fold(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def numspan_cache_key(file_hash: str, chunk_id: str, text: str) -> str:
    """enrich_cache PK 适配：kind 列 = numspan:{chunk_content_hash}
    （块哈希入键：重分块自动失效、同文重入库免费命中）。"""
    h = hashlib.blake2s((chunk_id + "\n" + (text or "")).encode("utf-8"), digest_size=10)
    return f"numspan:{h.hexdigest()}"


def validate_spans(text: str, spans: list[dict]) -> list[dict]:
    """确定性处置：子串命中 + canonical 可计算才留；返回规范化清单。"""
    norm = normalize(text or "")
    folded = _fold(norm.norm)
    out: list[dict] = []
    seen: set[str] = set()
    for s in spans or []:
        raw = _fold(str(s.get("text") or ""))
        if not raw or raw not in folded:
            continue                      # LLM 编造/改写 → 丢弃（非逐字不收）
        try:
            canon = re.sub(r"\s+", "", canonical_number(raw))
        except Exception:  # noqa: BLE001
            continue
        if not canon or canon in seen:
            continue
        seen.add(canon)
        tail_m = _UNIT_TAIL.search(raw)
        out.append({
            "text": str(s.get("text") or raw),
            "canonical": canon,
            "unit_type": None,            # 执法侧 classify 用 coverage._classify 兜底
            "context_hint": str(s.get("context_hint") or "")[:60],
            "unit_tail": tail_m.group(0) if tail_m else "",
        })
    return out


def annotate_numeric_spans(llm, text: str, kind: str = "table") -> list[dict]:
    """LLM 枚举 + 确定性校验 → numeric_spans 清单（空清单=无需/失败，回退运行时正则）。"""
    if not text or kind not in ("table", "chart", "text"):
        return []
    from ..llm.base import ChatMessage, llm_call_json
    body = text if len(text) <= 3500 else text[:3500] + "\n…（截断）"
    try:
        result = llm_call_json(
            llm,
            [ChatMessage(role="system", content=NUMSPAN_PROMPT, cache=True),
             ChatMessage(role="user", content=body)],
            schema={"type": "object",
                    "properties": {"spans": {"type": "array",
                                             "items": {"type": "object",
                                                       "properties": {
                                                           "text": {"type": "string"},
                                                           "context_hint": {"type": "string"}},
                                                       "required": ["text"]}}},
                    "required": ["spans"]},
            max_tokens=2048, effort="low")
    except Exception:  # noqa: BLE001 - 网关挂 → 回退运行时正则
        return []
    parsed = result.parsed if isinstance(result.parsed, dict) else {}
    return validate_spans(text, parsed.get("spans") or [])
