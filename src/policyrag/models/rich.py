"""公式/图表结构化记录（Phase B 产出 schema，通道无关）。

转写通道（MinerU / 视觉 / 确定性）产出统一结构，Phase C 消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["FormulaSymbol", "FormulaRecord", "ChartRecord"]


@dataclass
class FormulaSymbol:
    name: str          # 如 "R_{阻塞}" / "Q_{日上网,i,t}"（保留原文形态）
    desc: str          # "式中"给出的定义


@dataclass
class FormulaRecord:
    """一个公式 = 公式本体 + 符号表 + 截图（ground truth 通道）。"""

    local_id: str                      # 文档内序号，如 f_0238_01
    page: int                          # PDF 物理页码
    latex: str                         # 转写 LaTeX（可能为 None 时降级为 image-only）
    display_text: str                  # 人读形态（latex 无则中文描述）
    symbols: list[FormulaSymbol] = field(default_factory=list)
    context_before: str = ""           # 公式前文（定位锚）
    context_after: str = ""            # 公式后文
    image_path: str | None = None      # 截图（兜底通道）
    transcribed_by: str = "mineru"     # mineru | vision | deterministic | none
    confidence: Literal["high", "medium", "low"] = "medium"
    fixes_applied: list[str] = field(default_factory=list)   # 形近字等修复记录
    meta: dict[str, Any] = field(default_factory=dict)

    def evidence_text(self) -> str:
        """进入检索/证据池的文本（索引与引用校验都用它）。"""
        parts = []
        if self.display_text:
            parts.append(f"【公式】{self.display_text}")
        if self.latex:
            parts.append(f"【LaTeX】{self.latex}")
        for s in self.symbols:
            parts.append(f"【符号】{s.name} 为 {s.desc}")
        return "\n".join(parts)


@dataclass
class ChartRecord:
    """一张统计图 = 标题 + OCR 标签 + （可选）DePlot 数据表 + 截图。"""

    local_id: str
    page: int
    title: str = ""
    chart_type: str = "unknown"        # bar|pie|line|table|mixed|unknown
    ocr_texts: list[str] = field(default_factory=list)
    data_table: str = ""               # DePlot 输出（可选）
    data_source: str = ""              # 资料来源行
    image_path: str | None = None
    extracted_by: str = "rapidocr"     # rapidocr | rapidocr+deplot | vision
    confidence: Literal["high", "medium", "low"] = "medium"
    meta: dict[str, Any] = field(default_factory=dict)

    def evidence_text(self) -> str:
        parts = []
        if self.title:
            parts.append(f"【图表】{self.title}")
        if self.ocr_texts:
            parts.append("【图内文字】" + " | ".join(self.ocr_texts))
        if self.data_table:
            parts.append(f"【数据表】{self.data_table}")
        if self.data_source:
            parts.append(f"【资料来源】{self.data_source}")
        return "\n".join(parts)
