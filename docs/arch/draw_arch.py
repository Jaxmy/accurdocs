# -*- coding: utf-8 -*-
"""accurdocs 组件工作流架构图生成器（论文插图风格，matplotlib）。

产出三张风格变体：
  arch_v1_paper.png   —— 柔和马卡龙色系（贴近参考图）
  arch_v2_flat.png    —— 扁平饱和色 + 深色描边（更现代）
  arch_v3_swimlane.png—— 入库/查询双泳道布局（构图发散）

用法：python draw_arch.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------ 工具
def shadow_rbox(ax, x, y, w, h, fc, ec, lw=1.4, r=0.9, shadow=True, alpha=1.0):
    """圆角矩形 + 柔和投影。"""
    if shadow:
        sh = FancyBboxPatch((x - 0.35, y - 0.45), w, h,
                            boxstyle=f"round,pad=0,rounding_size={r}",
                            fc="#000000", ec="none", alpha=0.16, zorder=1)
        ax.add_patch(sh)
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2)
    ax.add_patch(p)
    return p


def box(ax, x, y, w, h, title, lines, palette, title_size=12.5, body_size=9.2,
        r=0.9, inner=False, shadow=True):
    """标准框：粗体标题 + 居中正文行。"""
    fc, ec, tc, bc = palette["fc"], palette["ec"], palette["tc"], palette["bc"]
    if inner:
        shadow_rbox(ax, x, y, w, h, fc, ec, lw=1.1, r=0.7, shadow=False)
    else:
        shadow_rbox(ax, x, y, w, h, fc, ec, lw=1.5, r=r, shadow=shadow)
    cx = x + w / 2
    if title:
        ty = y + h - (2.1 if not inner else 1.8)
        ax.text(cx, ty, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=tc, zorder=3)
        by0 = ty - 2.7
    else:
        by0 = y + h / 2 + (len(lines) - 1) * 1.35
    for i, ln in enumerate(lines):
        ax.text(cx, by0 - i * 2.55, ln, ha="center", va="center",
                fontsize=body_size, color=bc, zorder=3)


def container(ax, x, y, w, h, title, palette, title_size=12.5):
    """模块容器（大框 + 顶部标题）。"""
    shadow_rbox(ax, x, y, w, h, palette["fc"], palette["ec"], lw=1.6, r=1.1)
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=palette["tc"], zorder=3)


def arrow(ax, xy_from, xy_to, color="#5a5a5a", lw=1.7, style="-|>",
          rad=0.0, ls="-", ms=16, z=4):
    a = FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                        connectionstyle=f"arc3,rad={rad}",
                        color=color, lw=lw, linestyle=ls,
                        mutation_scale=ms, shrinkA=1, shrinkB=1, zorder=z)
    ax.add_patch(a)


def dash(ax, xy_from, xy_to, color="#8a8a8a", rad=0.0):
    arrow(ax, xy_from, xy_to, color=color, lw=1.3, ls=(0, (4, 3)), ms=11, rad=rad, z=1.2)
    return


def new_fig(w=17.2, h=9.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 172)
    ax.set_ylim(0, 96)
    ax.axis("off")
    return fig, ax


def title_block(ax, title, subtitle):
    ax.text(86, 93.0, title, ha="center", va="center",
            fontsize=19, fontweight="bold", color="#2b2b33")
    ax.text(86, 89.0, subtitle, ha="center", va="center",
            fontsize=11.5, color="#6b6b75", style="italic")


# ------------------------------------------------------------------ 配色
PASTEL = dict(
    blue=dict(fc="#d6e6f7", ec="#6b9fc9", tc="#2e6da4", bc="#3a5a78"),
    green=dict(fc="#d7ecd6", ec="#6aa56c", tc="#3c7a3e", bc="#3f6b41"),
    orange=dict(fc="#f8ddba", ec="#d4924e", tc="#b06a22", bc="#7a5a33"),
    yellow=dict(fc="#f7ebb4", ec="#cbb14e", tc="#9a8420", bc="#6f6426"),
    purple=dict(fc="#ded2ef", ec="#8b6cba", tc="#5f4396", bc="#54437a"),
    violet=dict(fc="#e8ddf6", ec="#a184cc", tc="#6a4ca0", bc="#5a4a80"),
    pink=dict(fc="#f7cdd8", ec="#d07090", tc="#b04a6c", bc="#8a4460"),
    rose=dict(fc="#f6c9c1", ec="#d07a6c", tc="#b05248", bc="#8a4a42"),
    gray=dict(fc="#ececf0", ec="#9a9aa8", tc="#4a4a58", bc="#4a4a58"),
)

FLAT = dict(
    blue=dict(fc="#2f6fed", ec="#1d4fc4", tc="#ffffff", bc="#eaf1ff"),
    green=dict(fc="#27a069", ec="#1b7d51", tc="#ffffff", bc="#e8f8ef"),
    orange=dict(fc="#e8862d", ec="#c26a1c", tc="#ffffff", bc="#fff3e6"),
    yellow=dict(fc="#e3b52c", ec="#bd9418", tc="#5c4708", bc="#5c4708"),
    purple=dict(fc="#7a4fd0", ec="#5c38ad", tc="#ffffff", bc="#f0eaff"),
    violet=dict(fc="#9a6fe0", ec="#7a52c0", tc="#ffffff", bc="#f3eeff"),
    pink=dict(fc="#e05580", ec="#c23a66", tc="#ffffff", bc="#ffeef3"),
    rose=dict(fc="#e06a55", ec="#bf4f3c", tc="#ffffff", bc="#ffecea"),
    gray=dict(fc="#5a5f6e", ec="#414655", tc="#ffffff", bc="#eef0f4"),
)

TEAL = dict(
    blue=dict(fc="#cfe8ef", ec="#5a9fb5", tc="#276b80", bc="#2e5f70"),
    green=dict(fc="#d4ece2", ec="#5fa585", tc="#2e7a58", bc="#356b52"),
    orange=dict(fc="#f6d9c0", ec="#d08a55", tc="#a85f28", bc="#7a5230"),
    yellow=dict(fc="#f2e7bd", ec="#c2a84e", tc="#8f7a20", bc="#6b5f22"),
    purple=dict(fc="#d5d2ee", ec="#8480c0", tc="#4f4a96", bc="#474376"),
    violet=dict(fc="#e2ddf2", ec="#9a90c8", tc="#5c54a0", bc="#4e4880"),
    pink=dict(fc="#f0d0d8", ec="#c07888", tc="#a04a5c", bc="#804450"),
    rose=dict(fc="#edd2c8", ec="#c08878", tc="#a05848", bc="#804a3c"),
    gray=dict(fc="#e6e8ea", ec="#98a0a6", tc="#485058", bc="#485058"),
)

# ============================================================ 变体 1：论文风
def build_v1():
    pal = PASTEL
    fig, ax = new_fig()
    title_block(ax, "accurdocs 精准检索 Agent · 组件工作流架构",
                "Precision retrieval pipeline: ingest → hybrid recall → generation → deterministic hard gates → refusal adjudication")

    # ---- 查询主链（左→右）
    box(ax, 3, 53, 16, 18, "输入 · 问答入口",
        ["Web GUI", "REST API", "MCP Server", "三入口一致"], pal["blue"], title_size=11.8)
    box(ax, 23, 53, 18, 18, "B1 检索方案",
        ["LLM 拆解子查询", "问题语义类型", "+ 量纲判定", "（校验闸第一信号）"], pal["green"])
    box(ax, 45, 53, 20, 18, "B3/B4 混合召回",
        ["BM25 + 向量双路", "条款别名词元", "同条枚举/标题家族/", "公式邻域三层扩展"], pal["orange"])

    # 生成容器（2 子框）
    container(ax, 69, 47, 21, 30, "B6 结构化生成", pal["yellow"])
    box(ax, 71.5, 58, 16, 8.2, "", ["逐句引用作答", "缺证据显式声明"],
        dict(fc="#fbf3cf", ec="#d4bc6a", tc="#9a8420", bc="#6f6426"), body_size=8.8, inner=True)
    box(ax, 71.5, 49.5, 16, 7.2, "", ["分析类【推导】标记", "闭包校验"],
        dict(fc="#fbf3cf", ec="#d4bc6a", tc="#9a8420", bc="#6f6426"), body_size=8.8, inner=True)

    # 硬闸容器（3 子框）
    container(ax, 94, 47, 25, 30, "B7 确定性硬闸", pal["purple"])
    inner_g = dict(fc="#e9dff6", ec="#9a7cc8", tc="#5f4396", bc="#54437a")
    box(ax, 96.5, 66.2, 20, 8.2, "", ["数值溯源 · canonical 归一", "方向词绑定 · 前提白名单"],
        inner_g, body_size=8.5, inner=True)
    box(ax, 96.5, 56.4, 20, 8.2, "", ["引文逐字 · 多层比对", "快照吸附 · 降级转述"],
        inner_g, body_size=8.5, inner=True)
    box(ax, 96.5, 46.6, 20, 8.2, "", ["覆盖闸 三态启停", "ARMED / SOFT / OFF"],
        inner_g, body_size=8.5, inner=True)

    box(ax, 123, 55, 17, 14, "B9 拒答裁决",
        ["校验失败 + 补答", "重写失败才拒答", "部分可答 → PARTIAL"], pal["violet"], body_size=8.8)

    box(ax, 144, 63.5, 25, 13.5, "输出 ① 答案信封",
        ["逐句引用 · 附图", "answered / partial"], pal["pink"], body_size=9.0)
    box(ax, 144, 48, 25, 13.5, "输出 ② 拒答 / 缺口声明",
        ["库外必拒 · 宁拒勿幻", "refused + 理由"], pal["rose"], body_size=9.0)

    # ---- 入库链（左→右，汇入索引）
    box(ax, 3, 22, 14, 15, "输入文档",
        ["PDF / DOCX / MD"], pal["blue"], title_size=10.5, body_size=8.5)
    box(ax, 21, 22, 18, 15, "解析 · 结构树",
        ["章/节/条/款/项", "稳定分块 ID"], pal["green"], title_size=10.5, body_size=8.5)
    box(ax, 43, 22, 21, 15, "富化队列",
        ["MinerU 公式 LaTeX", "OCR / VLM 图表双源", "数值枚举预标注"], pal["orange"], title_size=10.5, body_size=8.5)
    box(ax, 68, 22, 23, 15, "混合索引 · 存储",
        ["BM25 + 向量 bge-small-zh", "SQLite 元数据", "enrich_cache 内容哈希"], pal["purple"], title_size=10.5, body_size=8.5)

    # ---- 顶部：LLM 抽象
    box(ax, 72, 80.5, 58, 6.8, "",
        ["LLM 多供应商抽象：Anthropic 兼容 / DeepSeek / GLM · JSON Schema 结构化输出 · 提示词版本化"],
        pal["gray"], body_size=9.2, r=0.8)

    # ---- 底部：评测闭环
    box(ax, 100, 4, 69, 8, "",
        ["端到端评测闭环：100 题 × GUI/MCP 双入口机判 · 金提示正则判分 · 263 单元测试 · A/B 模式标定"],
        pal["gray"], body_size=9.2, r=0.8)

    # ---- 流向箭头
    ymid = 62
    arrow(ax, (19, ymid), (23, ymid))
    arrow(ax, (41, ymid), (45, ymid))
    arrow(ax, (65, ymid), (69, ymid))
    arrow(ax, (90, ymid), (94, ymid))
    arrow(ax, (119, ymid), (123, ymid))
    arrow(ax, (140, ymid), (144, 70.2), rad=-0.12)
    arrow(ax, (140, ymid), (144, 54.8), rad=0.12)

    # 入库链
    yi = 29.5
    arrow(ax, (17, yi), (21, yi))
    arrow(ax, (39, yi), (43, yi))
    arrow(ax, (64, yi), (68, yi))
    # 索引 → 召回
    arrow(ax, (79.5, 37), (55, 53), rad=0.18, lw=2.0, color="#c26a1c")
    ax.text(63.5, 46.8, "索引升级", fontsize=8.5, color="#c26a1c", ha="center")

    # LLM 虚线
    dash(ax, (82, 80.5), (32, 71), rad=0.12)
    dash(ax, (98, 80.5), (79.5, 77), rad=0.0)
    dash(ax, (112, 80.5), (106, 77), rad=0.0)

    # 评测虚线
    dash(ax, (112, 12), (106, 47), rad=0.10)
    dash(ax, (130, 12), (131.5, 55), rad=-0.05)

    fig.savefig("arch_v1_paper.png", dpi=160, bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)


# ============================================================ 变体 2：扁平风
def build_v2():
    pal = FLAT
    fig, ax = new_fig()
    title_block(ax, "accurdocs · Precision Retrieval Agent Architecture",
                "Ingestion & query pipelines with deterministic zero-hallucination hard gates")

    box(ax, 3, 53, 16, 18, "ENTRY",
        ["Web GUI", "REST API", "MCP Server"], pal["blue"], body_size=9.5)
    box(ax, 23, 53, 18, 18, "B1 · PLAN",
        ["LLM sub-queries", "question kind", "+ dimension"], pal["green"], title_size=11.5, body_size=9.5)
    box(ax, 45, 53, 20, 18, "B3/B4 · RECALL",
        ["BM25 + dense vectors", "clause alias tokens", "3-layer expansion"], pal["orange"], title_size=11, body_size=9.2)

    container(ax, 69, 47, 21, 30, "B6 · GENERATION", pal["yellow"], title_size=11.5)
    box(ax, 71.5, 58, 16, 8.2, "", ["per-sentence citations", "gap declared"],
        dict(fc="#f6df8a", ec="#d0a828", tc="#5c4708", bc="#5c4708"), body_size=8.6, inner=True)
    box(ax, 71.5, 49.5, 16, 7.2, "", ["【derivation】marks", "closure-checked"],
        dict(fc="#f6df8a", ec="#d0a828", tc="#5c4708", bc="#5c4708"), body_size=8.6, inner=True)

    container(ax, 94, 47, 25, 30, "B7 · HARD GATES", pal["purple"], title_size=11.5)
    inner_g = dict(fc="#9a72e0", ec="#7a52c0", tc="#ffffff", bc="#f3eeff")
    box(ax, 96.5, 66.2, 20, 8.2, "", ["numeric provenance", "canonical matching"],
        inner_g, body_size=8.6, inner=True)
    box(ax, 96.5, 56.4, 20, 8.2, "", ["verbatim citations", "snapshot snapping"],
        inner_g, body_size=8.6, inner=True)
    box(ax, 96.5, 46.6, 20, 8.2, "", ["coverage gate", "ARMED / SOFT / OFF"],
        inner_g, body_size=8.6, inner=True)

    box(ax, 123, 55, 17, 14, "B9 · VERDICT",
        ["refuse only after", "rewrite fails", "else PARTIAL"], pal["violet"], title_size=11, body_size=8.6)
    box(ax, 144, 63.5, 25, 13.5, "OUT ① ANSWER",
        ["citations + figures", "answered / partial"], pal["pink"], body_size=9.2)
    box(ax, 144, 48, 25, 13.5, "OUT ② REFUSAL",
        ["out-of-corpus → refuse", "refused + reason"], pal["rose"], body_size=9.2)

    box(ax, 3, 22, 14, 15, "DOCS",
        ["PDF / DOCX / MD"], pal["blue"], title_size=10, body_size=8.4)
    box(ax, 21, 22, 18, 15, "STRUCTURE TREE",
        ["chapter→clause→item", "stable chunk IDs"], pal["green"], title_size=9.2, body_size=8.4)
    box(ax, 43, 22, 21, 15, "ENRICHMENT",
        ["MinerU formulas", "OCR / VLM charts", "numeric pre-annotation"], pal["orange"], title_size=10, body_size=8.4)
    box(ax, 68, 22, 23, 15, "HYBRID INDEX",
        ["BM25 + bge-small-zh", "SQLite + enrich cache"], pal["purple"], title_size=10, body_size=8.4)

    box(ax, 68, 78.8, 66, 8.5, "",
        ["LLM abstraction: Anthropic-compatible / DeepSeek / GLM",
         "structured JSON output · versioned prompts"],
        pal["gray"], body_size=8.6, r=0.8, shadow=False)
    box(ax, 100, 2.5, 69, 9.5, "",
        ["Evaluation loop: 100 questions × dual entry, machine-graded",
         "gold-hint regex · 263 unit tests · A/B gate calibration"],
        pal["gray"], body_size=8.6, r=0.8, shadow=False)

    ymid = 62
    for a, b in ((19, 23), (41, 45), (65, 69), (90, 94), (119, 123)):
        arrow(ax, (a, ymid), (b, ymid), color="#2b2b33", lw=2.0)
    arrow(ax, (140, ymid), (144, 70.2), rad=-0.12, color="#2b2b33", lw=2.0)
    arrow(ax, (140, ymid), (144, 54.8), rad=0.12, color="#2b2b33", lw=2.0)

    yi = 29.5
    for a, b in ((17, 21), (39, 43), (64, 68)):
        arrow(ax, (a, yi), (b, yi), color="#2b2b33", lw=2.0)
    arrow(ax, (79.5, 37), (55, 53), rad=0.18, lw=2.2, color="#c26a1c")
    ax.text(70, 45.5, "index upgrade", fontsize=8.5, color="#c26a1c", ha="center")

    dash(ax, (82, 78.8), (32, 71), rad=0.12, color="#5a5f6e")
    dash(ax, (98, 78.8), (79.5, 77), color="#5a5f6e")
    dash(ax, (112, 78.8), (106, 77), color="#5a5f6e")
    dash(ax, (112, 12), (106, 47), rad=0.10, color="#5a5f6e")
    dash(ax, (130, 12), (131.5, 55), rad=-0.05, color="#5a5f6e")

    fig.savefig("arch_v2_flat.png", dpi=160, bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)


# ====================================================== 变体 3：双泳道布局
def build_v3():
    pal = TEAL
    fig, ax = new_fig()
    title_block(ax, "accurdocs 组件工作流 · 双泳道架构",
                "Ingestion lane & query lane over a shared hybrid index spine")

    # 泳道标签
    for y0, h0, name, col in ((48, 34, "查询泳道  QUERY LANE", "#276b80"),
                              (10, 30, "入库泳道  INGEST LANE", "#2e7a58")):
        shadow_rbox(ax, 1.5, y0, 4.2, h0, col, col, lw=0, r=0.6, shadow=False)
        ax.text(3.6, y0 + h0 / 2, name, rotation=90, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white", zorder=3)

    # ---- 查询泳道
    box(ax, 9, 58, 15, 16, "入口",
        ["GUI / REST", "MCP"], pal["blue"], title_size=11, body_size=9)
    box(ax, 28, 58, 19, 16, "B1 检索方案",
        ["子查询拆解", "语义类型 + 量纲"], pal["green"], title_size=11, body_size=8.8)
    box(ax, 51, 58, 21, 16, "B3/B4 混合召回",
        ["BM25 + 向量", "三层证据扩展"], pal["orange"], title_size=10.5, body_size=8.8)
    box(ax, 76, 58, 19, 16, "B6 结构化生成",
        ["逐句引用 · 【推导】", "闭包校验"], pal["yellow"], title_size=10.5, body_size=8.8)

    container(ax, 99, 51, 28, 28, "B7 确定性硬闸", pal["purple"])
    inner_g = dict(fc="#e0dcf0", ec="#9088c8", tc="#4f4a96", bc="#474376")
    box(ax, 101.5, 68.2, 23, 7.2, "", ["数值溯源 · 方向词绑定"], inner_g, body_size=8.6, inner=True)
    box(ax, 101.5, 59.4, 23, 7.2, "", ["引文逐字 · 快照吸附"], inner_g, body_size=8.6, inner=True)
    box(ax, 101.5, 50.6, 23, 7.2, "", ["覆盖闸 · 三态启停"], inner_g, body_size=8.6, inner=True)

    box(ax, 131, 64.5, 18, 12, "B9 裁决",
        ["拒答 / 部分 / 放行"], pal["violet"], title_size=10.5, body_size=8.6)
    box(ax, 153, 64.5, 17, 12, "输出信封",
        ["引用 · 附图 · 理由"], pal["pink"], title_size=10.5, body_size=8.6)

    # ---- 入库泳道
    box(ax, 9, 16, 15, 16, "文档",
        ["PDF / DOCX", "MD"], pal["blue"], title_size=11, body_size=9)
    box(ax, 28, 16, 19, 16, "解析 · 结构树",
        ["章节条款项", "稳定分块 ID"], pal["green"], title_size=11, body_size=8.8)
    box(ax, 51, 16, 21, 16, "富化队列",
        ["公式 / 图表双源", "数值预标注"], pal["orange"], title_size=11, body_size=8.8)

    # 中央索引脊
    shadow_rbox(ax, 76, 27, 44, 16, "#fff4e0", "#d08a55", lw=1.8, r=1.0)
    ax.text(98, 39.2, "混合索引 · 存储脊  HYBRID INDEX SPINE", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#a85f28", zorder=3)
    ax.text(98, 34.4, "BM25 + 向量 bge-small-zh · SQLite 元数据", ha="center", va="center",
            fontsize=8.8, color="#7a5230", zorder=3)
    ax.text(98, 30.6, "enrich_cache 内容哈希缓存 · 幂等入库", ha="center", va="center",
            fontsize=8.8, color="#7a5230", zorder=3)

    # LLM 侧车
    box(ax, 131, 40, 39, 10, "LLM 多供应商抽象",
        ["Anthropic 兼容 / DeepSeek / GLM", "JSON Schema 结构化输出"],
        pal["gray"], title_size=10.5, body_size=8.6)

    # 评测带
    box(ax, 76, 2.5, 94, 7, "",
        ["端到端评测闭环：100 题 × 双入口机判 · 金提示正则判分 · 263 单测 · A/B 模式标定"],
        pal["gray"], body_size=8.8, r=0.7)

    # ---- 箭头：查询泳道
    yq = 66
    for a, b in ((24, 28), (47, 51), (72, 76), (95, 99), (127, 131), (149, 153)):
        arrow(ax, (a, yq), (b, yq))
    # ---- 箭头：入库泳道
    yi = 24
    arrow(ax, (24, yi), (28, yi))
    arrow(ax, (47, yi), (51, yi))
    arrow(ax, (72, yi), (76, 33), rad=-0.15)
    # 索引脊 → 召回
    arrow(ax, (88, 42), (61, 58), rad=0.12, lw=2.0, color="#c26a1c")
    ax.text(78, 51, "索引升级", fontsize=8.5, color="#c26a1c", ha="center")

    # LLM 虚线到 B1/B6
    dash(ax, (131, 47), (37, 74), rad=0.06)
    dash(ax, (140, 40), (85, 58), rad=0.12)
    # 评测虚线到硬闸/裁决
    dash(ax, (100, 9.5), (106, 51), rad=0.08)
    dash(ax, (140, 9.5), (140, 64.5), rad=-0.04)

    fig.savefig("arch_v3_swimlane.png", dpi=160, bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)


# ================================================== 变体 2-中文：扁平风
def build_v2_cn():
    pal = FLAT
    fig, ax = new_fig()
    title_block(ax, "accurdocs · 精准检索 Agent 组件工作流架构",
                "入库与查询双流水线 · 确定性零幻觉硬闸")

    box(ax, 3, 53, 16, 18, "问答入口",
        ["Web GUI", "REST API", "MCP Server"], pal["blue"], body_size=9.5)
    box(ax, 23, 53, 18, 18, "B1 · 检索方案",
        ["LLM 拆解子查询", "问题语义类型", "+ 量纲判定"], pal["green"], title_size=11.5, body_size=9.5)
    box(ax, 45, 53, 20, 18, "B3/B4 · 混合召回",
        ["BM25 + 向量双路", "条款别名词元", "三层证据扩展"], pal["orange"], title_size=11, body_size=9.2)

    container(ax, 69, 47, 21, 30, "B6 · 结构化生成", pal["yellow"], title_size=11.5)
    box(ax, 71.5, 58, 16, 8.2, "", ["逐句引用作答", "缺证据显式声明"],
        dict(fc="#f6df8a", ec="#d0a828", tc="#5c4708", bc="#5c4708"), body_size=8.6, inner=True)
    box(ax, 71.5, 49.5, 16, 7.2, "", ["分析类【推导】标记", "闭包校验"],
        dict(fc="#f6df8a", ec="#d0a828", tc="#5c4708", bc="#5c4708"), body_size=8.6, inner=True)

    container(ax, 94, 47, 25, 30, "B7 · 确定性硬闸", pal["purple"], title_size=11.5)
    inner_g = dict(fc="#9a72e0", ec="#7a52c0", tc="#ffffff", bc="#f3eeff")
    box(ax, 96.5, 66.2, 20, 8.2, "", ["数值溯源 · canonical 归一", "方向词绑定 · 前提白名单"],
        inner_g, body_size=8.6, inner=True)
    box(ax, 96.5, 56.4, 20, 8.2, "", ["引文逐字 · 多层比对", "快照吸附 · 降级转述"],
        inner_g, body_size=8.6, inner=True)
    box(ax, 96.5, 46.6, 20, 8.2, "", ["覆盖闸 · 三态启停", "ARMED / SOFT / OFF"],
        inner_g, body_size=8.6, inner=True)

    box(ax, 123, 55, 17, 14, "B9 · 拒答裁决",
        ["校验失败 + 补答", "重写失败才拒答", "部分可答 → PARTIAL"], pal["violet"], title_size=11, body_size=8.6)
    box(ax, 144, 63.5, 25, 13.5, "输出 ① 答案信封",
        ["逐句引用 · 附图", "answered / partial"], pal["pink"], body_size=9.2)
    box(ax, 144, 48, 25, 13.5, "输出 ② 拒答声明",
        ["库外必拒 · 宁拒勿幻", "refused + 理由"], pal["rose"], body_size=9.2)

    box(ax, 3, 22, 14, 15, "输入文档",
        ["PDF / DOCX / MD"], pal["blue"], title_size=10, body_size=8.4)
    box(ax, 21, 22, 18, 15, "解析 · 结构树",
        ["章/节/条/款/项", "稳定分块 ID"], pal["green"], title_size=10, body_size=8.4)
    box(ax, 43, 22, 21, 15, "富化队列",
        ["MinerU 公式转写", "OCR / VLM 图表双源", "数值枚举预标注"], pal["orange"], title_size=10, body_size=8.4)
    box(ax, 68, 22, 23, 15, "混合索引 · 存储",
        ["BM25 + 向量 bge-small-zh", "SQLite · enrich_cache"], pal["purple"], title_size=10, body_size=8.4)

    box(ax, 68, 78.8, 66, 8.5, "",
        ["LLM 多供应商抽象：Anthropic 兼容 / DeepSeek / GLM",
         "JSON Schema 结构化输出 · 提示词版本化"],
        pal["gray"], body_size=8.6, r=0.8, shadow=False)
    box(ax, 100, 2.5, 69, 9.5, "",
        ["端到端评测闭环：100 题 × 双入口机判 · 金提示正则判分",
         "263 单元测试 · A/B 模式标定"],
        pal["gray"], body_size=8.6, r=0.8, shadow=False)

    ymid = 62
    for a, b in ((19, 23), (41, 45), (65, 69), (90, 94), (119, 123)):
        arrow(ax, (a, ymid), (b, ymid), color="#2b2b33", lw=2.0)
    arrow(ax, (140, ymid), (144, 70.2), rad=-0.12, color="#2b2b33", lw=2.0)
    arrow(ax, (140, ymid), (144, 54.8), rad=0.12, color="#2b2b33", lw=2.0)

    yi = 29.5
    for a, b in ((17, 21), (39, 43), (64, 68)):
        arrow(ax, (a, yi), (b, yi), color="#2b2b33", lw=2.0)
    arrow(ax, (79.5, 37), (55, 53), rad=0.18, lw=2.2, color="#c26a1c")
    ax.text(63.5, 46.8, "索引升级", fontsize=8.5, color="#c26a1c", ha="center")

    dash(ax, (82, 78.8), (32, 71), rad=0.12, color="#5a5f6e")
    dash(ax, (98, 78.8), (79.5, 77), color="#5a5f6e")
    dash(ax, (112, 78.8), (106, 77), color="#5a5f6e")
    dash(ax, (112, 12), (106, 47), rad=0.10, color="#5a5f6e")
    dash(ax, (130, 12), (131.5, 55), rad=-0.05, color="#5a5f6e")

    fig.savefig("arch_v2_flat_cn.png", dpi=160, bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    plt.close(fig)


if __name__ == "__main__":
    build_v1()
    build_v2()
    build_v2_cn()
    build_v3()
    print("done: arch_v1_paper.png / arch_v2_flat.png / arch_v2_flat_cn.png / arch_v3_swimlane.png")
