"""公式转写与公式块构建（公式线主入口）。

链路（实测定型）：
  PDF → MinerU（本地 parse-server）→ markdown（$$LaTeX$$ + 页标记 + 式中）
      → 符号表解析 → 形近字/符号名对照原文校验 → 公式截图 → FormulaRecord

降级：MinerU 不可用 → 仅截图 + display_text（image-only，宁缺勿错）。
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from ..models.ids import desurrogate, normalize_text_for_id
from ..models.rich import FormulaRecord, FormulaSymbol

__all__ = ["run_mineru", "parse_mineru_markdown", "build_formula_records"]


# ---------------------------------------------------------------- MinerU 调用

def run_mineru(pdf_path: str | Path, out_path: str | Path,
               pages: str = "all", tier: str = "basic",
               wait_s: int = 1800) -> str | None:
    """调 mineru CLI 解析 PDF，返回输出 markdown 文本（失败返回 None）。

    注意：out_path 是**文件**路径（v4 CLI 直接写文件）——不要预建目录，
    否则 CLI 行为歧义导致解析结果丢失。
    """
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
    env.setdefault("OMP_NUM_THREADS", "2")
    # 引擎自管理：未起则自动拉起（配置自愈 + 闲置释放），调用方零感知
    try:
        from .mineru_engine import ensure_mineru_ready, touch
        if not ensure_mineru_ready():
            return None
        touch()
    except Exception:  # noqa: BLE001
        pass
    cmd = ["mineru", "parse", str(pdf_path), "--pages", pages,
           "--tier", tier, "-o", str(out_path), "--wait", str(wait_s)]
    import shutil
    if shutil.which("mineru") is None:
        cand = Path(__file__).resolve().parents[3] / ".venv312" / "Scripts" / "mineru.exe"
        if cand.exists():
            cmd[0] = str(cand)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=wait_s + 60, env=env)
        if r.returncode != 0:
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out_path.is_file():
        return out_path.read_text(encoding="utf-8", errors="replace")
    # CLI 若把 -o 当目录写：在其中找输出
    if out_path.is_dir():
        for p in sorted(out_path.rglob("*.md")):
            return p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(out_path.iterdir()):
            if p.is_file() and p.stat().st_size > 100:
                return p.read_text(encoding="utf-8", errors="replace")
    return None


# ------------------------------------------------------------- markdown 解析

_PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s+of\s+\d+\s*-->")
_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_SHIZHONG_RE = re.compile(r"(式中[:：]?|公式中[:：]?|其中[:：]?)")
_FIX_LEXICON = [
    # (错, 对, 说明) —— 形近字修复（PoC 实测错误样本）；对侧必须在原文中出现才修
    ("含内实时加权", "省内实时加权", "形近字含→省"),
    ("必开时限", "必开时段", "形近字限→段"),
    ("超超期", "超短期", "衍字"),
]


_CJK = "一-鿿"


def compact_cjk_latex(latex: str) -> str:
    """压缩 LaTeX 中文下标内的空格（MinerU 输出常带散空格，影响检索与显示）。"""
    return re.sub(f"(?<=[{_CJK}])[ 	]+(?=[{_CJK}])", "", latex)


def parse_mineru_markdown(md: str) -> list[dict]:
    """markdown → 原始公式列表：[{page, latex, before, after, tail}]。

    tail = 公式后到下一公式/页尾之间的文本（式中 符号表来源）。
    """
    marks = [(m.start(), int(m.group(1))) for m in _PAGE_RE.finditer(md)]

    def page_of(pos: int) -> int:
        page = marks[0][1] if marks else 1
        for start, pno in marks:
            if start <= pos:
                page = pno
            else:
                break
        return page

    out = []
    for m in _BLOCK_RE.finditer(md):
        latex = m.group(1).strip()
        before = md[max(0, m.start() - 120):m.start()]
        nxt = _BLOCK_RE.search(md, m.end())
        tail_end = min(nxt.start() if nxt else len(md), m.end() + 800)
        tail = _PAGE_RE.sub("", md[m.end():tail_end])
        out.append({
            "page": page_of(m.start()),
            "latex": compact_cjk_latex(re.sub(r"\s+", " ", latex).strip()),
            "before": re.sub(r"\s+", "", before)[-60:],
            "after": re.sub(r"\s+", "", md[m.end():m.end() + 60])[:40],
            "tail": tail,
        })
    return out


# ------------------------------------------------------------------ 符号表

_SYM_RE = re.compile(
    r"[$]?([A-Za-z\\][\\A-Za-z]*(?:[_^]\{[^}]{1,40}\})?(?:[,，][^为；;]{1,20})?)[$]?\s*[为是]\s*([^；;\n。]{2,120})"
)


def extract_symbols(tail: str) -> list[FormulaSymbol]:
    """从式中段落解析符号定义。"""
    symbols = []
    if not _SHIZHONG_RE.search(tail):
        return symbols
    body = _SHIZHONG_RE.split(tail, maxsplit=1)[-1]
    for m in _SYM_RE.finditer(body):
        name = re.sub(r"\s+", "", m.group(1))
        desc = m.group(2).strip()
        if 2 <= len(name) <= 60:
            symbols.append(FormulaSymbol(name=name, desc=desc))
    return symbols


# -------------------------------------------------------- 形近字/符号名校验

def validate_latex(latex: str, doc_text_norm: str, fixes: list[str]) -> str:
    """符号名对照原文校验：形近字修复只在'修正后出现在原文'时执行（宁缺勿错）。"""
    fixed = latex
    for wrong, right, why in _FIX_LEXICON:
        if wrong in fixed and right in doc_text_norm and wrong not in doc_text_norm:
            fixed = fixed.replace(wrong, right)
            fixes.append(f"{why}: {wrong}→{right}")
    return fixed


# ------------------------------------------------------------------ 截图

_FORMULA_ANCHOR_RE = re.compile(r"(计算公式|计算方式|计算如下|公式如下|式中)")


def crop_formula_image(pdf_doc, page_no: int, before: str,
                       out_path: Path) -> str | None:
    """按公式前文定位裁剪截图；定位失败裁锚行区域兜底。"""
    try:
        page = pdf_doc[page_no - 1]
    except Exception:  # noqa: BLE001
        return None
    rect = None
    if before:
        anchor = before[-16:]
        try:
            quads = page.search_for(anchor)
        except Exception:  # noqa: BLE001
            quads = []
        if quads:
            y0 = min(q.y0 for q in quads) - 8
            rect = type(page.rect)(40, max(0, y0), page.rect.width - 40,
                                   min(page.rect.height, y0 + 170))
    if rect is None:
        words = page.get_text("words")
        ys = [w[1] for w in words if _FORMULA_ANCHOR_RE.search(w[4])]
        if ys:
            y0 = min(ys) - 10
            rect = type(page.rect)(40, max(0, y0), page.rect.width - 40,
                                   min(page.rect.height, y0 + 200))
        else:
            return None
    try:
        pix = page.get_pixmap(clip=rect, dpi=170)
        pix.save(str(out_path))
        return str(out_path)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ 总装

def build_formula_records(pdf_path: str | Path, doc_id: str, version: str,
                          images_dir: str | Path,
                          md_text: str | None = None,
                          pages: str = "all") -> tuple[list[FormulaRecord], list[str]]:
    """PDF → FormulaRecord 列表（含截图）。返回 (records, warnings)。"""
    import pymupdf

    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if md_text is None:
        with tempfile.TemporaryDirectory(prefix="mineru_") as td:
            out_file = Path(td) / "out.md"
            md_text = run_mineru(pdf_path, out_file, pages=pages)
        if md_text is None:
            warnings.append("mineru 不可用：公式仅截图，无转写")

    doc = pymupdf.open(pdf_path)
    raw_text = desurrogate("\n".join(p.get_text() for p in doc))
    text_norm = normalize_text_for_id(raw_text)

    raws = parse_mineru_markdown(md_text) if md_text else []
    records: list[FormulaRecord] = []
    per_page_count: dict[int, int] = {}

    if raws:
        for r in raws:
            n = per_page_count.get(r["page"], 0) + 1
            per_page_count[r["page"]] = n
            local_id = f"f_{r['page']:03d}_{n:02d}"
            fixes: list[str] = []
            latex = validate_latex(r["latex"], text_norm, fixes)
            symbols = extract_symbols(r["tail"])
            conf = "high" if not fixes else "medium"
            if "�" in latex:
                conf = "low"
            img = crop_formula_image(doc, r["page"], r["before"],
                                     images_dir / f"{local_id}.png")
            records.append(FormulaRecord(
                local_id=local_id, page=r["page"], latex=latex,
                display_text=latex, symbols=symbols,
                context_before=r["before"], context_after=r["after"],
                image_path=img, transcribed_by="mineru",
                confidence=conf, fixes_applied=fixes,
            ))
    else:
        for pno in range(len(doc)):
            t = doc[pno].get_text()
            if not _FORMULA_ANCHOR_RE.search(t):
                continue
            n = per_page_count.get(pno + 1, 0) + 1
            per_page_count[pno + 1] = n
            local_id = f"f_{pno+1:03d}_{n:02d}"
            img = crop_formula_image(doc, pno + 1, "计算公式",
                                     images_dir / f"{local_id}.png")
            if img:
                records.append(FormulaRecord(
                    local_id=local_id, page=pno + 1, latex="",
                    display_text="（公式见附图，自动转写不可用）",
                    image_path=img, transcribed_by="none", confidence="low",
                ))
    doc.close()
    return records, warnings
