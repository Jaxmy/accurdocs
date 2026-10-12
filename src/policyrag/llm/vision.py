"""视觉能力：探测（probe）+ 图表转写（VLM）。

设计（准确度第一）：
- 视觉能力以**探测**为准（发一张已知数字的小图问数，回复命中才算具备）——
  模型名列表仅作提示；MCP 挂载到其他 LLM 时宿主环境变量会共享键（process env wins），
  该键未必有视觉能力，且 MCP 通道无 GUI 可提示 → 能力验证必须内建于工具链并回报报文；
- 转写双源交叉：VLM 读数 × OCR 字面，一致才入证据文本（强证据，可进数字保真比对），
  不一致只进展示数据表（弱证据，不进比对集——防 VLM 误读被"答证一致"自我印证）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .base import ChatMessage, LLMResult

__all__ = ["make_vision_llm", "probe_vision", "transcribe_chart",
           "vision_fingerprint", "needs_vision_hint"]

_PROBE_DIGIT = "7"


def vision_fingerprint() -> str:
    """当前视觉配置指纹（探测缓存键）。"""
    key = (os.environ.get("POLICYRAG_VISION_API_KEY")
           or os.environ.get("ANTHROPIC_AUTH_TOKEN")
           or os.environ.get("ANTHROPIC_API_KEY") or "")
    raw = "|".join([
        os.environ.get("POLICYRAG_VISION_MODEL") or os.environ.get("POLICYRAG_MODEL") or "",
        os.environ.get("POLICYRAG_VISION_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL") or "",
        hashlib.sha256(key.encode()).hexdigest()[:8],
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def make_vision_llm():
    """视觉模型客户端：POLICYRAG_VISION_* 三元组优先，缺省复用主 LLM 配置。"""
    from ..settings import load_env, make_llm
    load_env()
    if not (os.environ.get("POLICYRAG_VISION_MODEL") or os.environ.get("POLICYRAG_VISION_API_KEY")):
        return make_llm()   # 复用主配置（键可能继承宿主环境）
    from .anthropic_client import AnthropicClient
    return AnthropicClient(
        model=os.environ.get("POLICYRAG_VISION_MODEL") or None,
        api_key=os.environ.get("POLICYRAG_VISION_API_KEY") or None,
    )


def _probe_image() -> bytes:
    """生成含已知数字的 96x96 PNG（无第三方绘图依赖，pymupdf 渲染）。"""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=96, height=96)
    page.insert_text((30, 70), _PROBE_DIGIT, fontsize=52)
    pix = page.get_pixmap()
    img = pix.tobytes("png")
    doc.close()
    return img


def probe_vision(force: bool = False) -> dict[str, Any]:
    """视觉能力探测（带缓存）。返回 {capable, model, reason, checked}。"""
    from ..settings import PROJECT_ROOT
    cache_file = PROJECT_ROOT / "data" / "vision_probe.json"
    fp = vision_fingerprint()
    if not force:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fp and cached.get("checked"):
                return {k: cached[k] for k in ("capable", "model", "reason", "checked")}
        except Exception:  # noqa: BLE001
            pass
    out: dict[str, Any] = {"capable": False, "model": "", "reason": "", "checked": True}
    try:
        llm = make_vision_llm()
        out["model"] = getattr(llm, "model", getattr(llm, "name", ""))
        # 预算必须覆盖 thinking 块 + 可见文本——16 token 曾被思考耗光致空回复
        # （stop=max_tokens 被误判成"无视觉能力"，实测 mimo-v2.6-flash 全模态正常）
        for budget in (256, 1024):
            res: LLMResult = llm.complete(
                [ChatMessage(role="user",
                             content="图中显示的数字是几？只回答这一个数字。",
                             images=[_probe_image()])],
                max_tokens=budget, temperature=0.0, effort="low",
            )
            m = re.search(r"\d+", res.text or "")
            if m and m.group(0) == _PROBE_DIGIT:
                out["capable"] = True
                out["reason"] = "视觉探测通过"
                break
            if res.stop_reason != "max_tokens":
                out["reason"] = f"视觉探测未通过（回复 {res.text[:40]!r}）"
                break
            out["reason"] = "视觉探测被 max_tokens 截断（thinking 占用预算），已扩预算重试"
        else:
            out["reason"] = "视觉探测多次截断，未能取得可见回复"
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"视觉探测失败：{e}"
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({**out, "fingerprint": fp}, ensure_ascii=False),
                              encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return out


_TRANSCRIBE_PROMPT = """你是图表数据转写员。读取图中全部可读数值，输出 JSON（只输出 JSON）：
{
  "title": "图表标题",
  "chart_type": "table|bar|line|pie|mixed|other",
  "values": [{"series": "系列/行名", "axis": "横轴/列名", "value": "原文形态数值（含千分位/单位，逐字）"}],
  "texts": ["图内其他可读文字"]
}
铁律：数值必须逐字照抄图上印刷形态（含小数点、千分位、单位），禁止换算、估算、补全；
读不清的数值不要输出。"""


def transcribe_chart(llm, image_bytes: bytes) -> dict[str, Any]:
    """VLM 图表转写 → {title, chart_type, values, texts}（失败返回 {}）。"""
    res = llm.complete(
        [ChatMessage(role="user", content=_TRANSCRIBE_PROMPT, images=[image_bytes])],
        max_tokens=2048, temperature=0.0, effort="low",
    )
    parsed = res.parsed if isinstance(res.parsed, dict) else None
    if parsed is None:
        from .base import _extract_json
        parsed = _extract_json(res.text or "")
    return parsed if isinstance(parsed, dict) else {}


def needs_vision_hint(n_charts: int, n_images_no_text: int) -> bool:
    """是否属于需要视觉 API 的复杂图表文档（研报/论文类）。"""
    return n_charts >= 5 or (n_charts >= 2 and n_images_no_text >= 5)


def _canon_num(s: str) -> str:
    from ..ingest.normalize import canonical_number
    return canonical_number(str(s or ""))


def tier_chart_values(vlm_values: list[dict], ocr_texts: list[str]) -> tuple[list[dict], list[dict]]:
    """双源交叉分层：VLM 读数 × OCR 字面命中 → (strong, weak)。

    strong = 数值规范形在 OCR 文本中出现（双源一致，可进数字保真比对集）；
    weak   = 仅 VLM 一家之言（只进展示数据表，不进比对集——防误读自我印证）。
    """
    hay = _canon_num(" ".join(ocr_texts or [])).replace(" ", "")
    strong, weak = [], []
    for v in vlm_values or []:
        c = _canon_num(v.get("value", "")).replace(" ", "")
        hit = bool(c) and bool(re.search(rf"(?<![\d.]){re.escape(c)}(?![\d.])", hay))
        (strong if hit else weak).append(v)
    return strong, weak
