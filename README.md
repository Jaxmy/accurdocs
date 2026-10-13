<div align="center">

[English](README.en.md) | **简体中文**

# accurdocs

**政策 / 金融长文档精准检索与分析 Agent**

*宁可拒答，不可幻觉 —— 每句结论可追溯到原子级原文片段*

[![Version](https://img.shields.io/badge/version-v0.4.0-blue)](https://github.com/Jaxmy/accurdocs/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-server-orange)](https://modelcontextprotocol.io/)

</div>

---

## 目录

- [简介](#简介)
- [功能特性](#功能特性)
- [组件构成](#组件构成)
- [关键技术](#关键技术)
- [环境依赖](#环境依赖)
- [安装](#安装)
- [使用方法](#使用方法)
- [实现方法](#实现方法)
- [测试结果](#测试结果)
- [当前版本与路线图](#当前版本与路线图)
- [限制](#限制)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 简介

accurdocs 是面向**政策文件、监管规则、券商研报**等长文档的精准检索问答 Agent。它以"引用逐字可溯、数值确定性校验、宁可拒答"为第一设计目标，把幻觉风险压到机制层面而非提示词层面：

- 每条引用必须**逐字命中**原文块（多层归一比对 + 快照吸附对齐），失配即拒答或降级为明确标注的转述；
- 答案中每个数字 / 日期 / 方向词都要过**确定性溯源校验**，无法回溯原文的句子被删除或整题拒答；
- 问题数值（用户前提）与证据数值严格区分，误杀与放行都有回归测试兜底。

同时提供 **Web GUI、REST API、MCP Server** 三入口，可直接挂载进 Claude Code / Cursor / VS Code 等任意支持 MCP 的 AI 客户端。

## 功能特性

| 能力 | 说明 |
|---|---|
| 精准检索问答 | BM25 + 向量混合召回，条款号/数值锚点/同条枚举扩展，逐句引用溯源 |
| 结构化索引树 | 中文章/节/条/款/项自动识别，标题树与 PDF 原文页码双向跳转 |
| 公式与图表富化 | 公式 LaTeX 转写 + 符号表、图表 OCR/VLM 数值双源转写，附图随答案出示 |
| 数值/引文双硬闸 | 数值溯源（含方向词绑定）、覆盖闸防"问数值答无数"、引文逐字校验 |
| 智能问句分类 | 问题语义类型（枚举/数值/比较/定义/流程/分析）+ 量纲自动判定，驱动校验闸启停 |
| 分析推理 | 分析/比较类问题锚定事实作答 + 显式【推导】标记，闭包校验，缺证据部分明确声明 |
| 版本判定 | 同文档多版本（文号/版本号/稿态）自动归族，防新旧版本混答 |
| 双入口一致 | Web GUI 与 MCP 同题双跑对照测试线，分歧即记 bug |

## 组件构成

```
accurdocs/
├── src/policyrag/            # 核心包（Python）
│   ├── ingest/               #   解析（PDF/DOCX/MD）→ 结构树 → 分块 → 富化队列
│   │   └── numspan.py        #   证据数值枚举预标注（LLM 提议 + 确定性校验）
│   ├── index/                #   BM25 / 向量 / 混合索引与元数据存储（SQLite）
│   ├── query/                #   检索方案（search_plan）→ 混合召回 → 生成 → 校验
│   ├── verify/               #   确定性硬闸：数值溯源 / 引文逐字 / 覆盖闸 / 拒答裁决
│   ├── llm/                  #   多供应商 LLM 抽象（Anthropic 兼容 / DeepSeek / GLM）
│   ├── service/              #   编排核心（问答/入库/文档管理/版本归族）
│   └── adapters/
│       ├── rest/             #   FastAPI + Web GUI（accurdocs 界面）
│       └── mcp/              #   MCP Server（stdio / 可选 HTTP）
├── config/                   # 检索参数与提示词版本
├── scripts/                  # 查询 / 冒烟 / 修复工具
├── tests/unit/               # 单元测试
├── start_api.bat             # 启动 Web GUI + REST（端口可用 ACCURDOCS_PORT 覆写）
├── start_mcp.bat             # 启动 MCP Server（stdio）
├── setup_mcp.py              # MCP 一键挂载脚本
└── mcp_config_example.json   # 各客户端 MCP 配置模板
```

## 关键技术

| 层 | 技术 | 用途 |
|---|---|---|
| 服务与接口 | Python 3.12 · FastAPI + Uvicorn · Pydantic | REST API、Web GUI、结构化数据模型 |
| AI 客户端接入 | MCP SDK（stdio / 可选 HTTP 传输） | 12 个工具挂载进 Claude Code / Cursor / VS Code 等 |
| 混合检索 | rank-bm25 + jieba 中文分词 · fastembed（bge-small-zh-v1.5，ONNX）· NumPy | 稀疏+稠密双路召回、条款别名词元（`§条12`）双侧发射、指纹守卫防向量混搭 |
| 文档解析 | PyMuPDF · python-docx · markdown-it-py | PDF 文本/表格/页图抽取、DOCX/MD 入库、原文 PDF 页码跳转 |
| 结构识别 | 自研中文结构树（章/节/条/款/项正则 + 折行归并 + 标题续行/噪声行过滤） | 稳定分块 ID（blake2s）、索引树与来源页双向绑定 |
| 公式/图表富化 | MinerU（公式 LaTeX 转写）· RapidOCR（子进程隔离）· VLM 视觉模型（图表数值双源转写） | 公式转写+符号表、图表 OCR/VLM 强弱分级、富化优先级队列（断点续跑） |
| 数值预标注 | LLM 提议 + 确定性处置（子串命中 + canonical 可计算） | 证据数值枚举（断格/跨行值）入库预标注，与运行时正则并集执法 |
| LLM 抽象 | Anthropic SDK / OpenAI SDK / ZhipuAI 三供应商 · JSON Schema 结构化输出 · 提示词版本化 | 检索方案、问答生成、充分性评审、分析合成分级调用 |
| 确定性校验 | 数值 canonical 归一（全半角/中文数字/日期 ISO 双轨）· 方向词绑定 · difflib 快照吸附 · 三态启停（ARMED/SOFT/OFF） | 数值溯源、引文逐字、覆盖闸、拒答裁决——零幻觉硬闸 |
| 存储 | SQLite（元数据/富化缓存/任务队列）· 内容哈希缓存（enrich_cache） | 幂等入库、重入库免费复用富化结果 |

## 环境依赖

| 项 | 要求 |
|---|---|
| 操作系统 | Windows / macOS / Linux（启动脚本为 Windows bat；POSIX 用 `python -m …` 直起） |
| Python | 3.12+（推荐独立 venv） |
| 内存 | 建议 8GB 以上（向量索引与图表富化时占用较高） |
| LLM API | 任一：Anthropic 兼容网关（API Key 或 AUTH_TOKEN）、DeepSeek、智谱 GLM；**密钥只写本地 `.env`，不入库不提交** |
| 可选 | 视觉能力 API（图表数值转写）；无视觉能力时图表数值题会按设计拒答 |
| 网络 | 首次安装需访问 PyPI；向量模型（bge-small-zh）首次运行自动下载并本地缓存 |

> 所有端口、路径、密钥均通过环境变量 / 本地 `.env` 配置，代码与文档不绑定任何具体部署环境。

## 安装

```bash
# 1. 获取代码
git clone https://github.com/Jaxmy/accurdocs.git
cd accurdocs

# 2. 创建虚拟环境（Python 3.12+）
python -m venv .venv
# Windows:  .venv\Scripts\activate     POSIX:  source .venv/bin/activate

# 3. 安装依赖（fastembed 可选：完整向量检索；不装则退化为哈希向量仅供冒烟）
pip install -e ".[dev]"
pip install -e ".[embed]"     # 可选，推荐

# 4. 配置密钥
cp .env.example .env          # Windows: copy .env.example .env
# 编辑 .env，填入 LLM 供应商与密钥（Anthropic 兼容 / DeepSeek / GLM 三选一）

# 5.（Windows）设置 UTF-8，避免控制台中文乱码
set PYTHONUTF8=1              # POSIX:  export PYTHONUTF8=1
```

## 使用方法

### ① Web GUI + REST API

```bat
start_api.bat
```

启动后浏览器打开脚本输出的地址即可：上传文档（自动入库、自动富化、进度实时可见），在问答页提问、跳转原文 PDF、查看逐句引用。REST 接口文档见 `/docs`（Swagger）。

### ② MCP 一键挂载（推荐给 AI 客户端）

```bash
python setup_mcp.py --claude    # 生成 .mcp.json 并注册到 Claude Code
python setup_mcp.py             # 其他客户端：按输出把 mcpServers 段并入配置
```

各客户端（Claude Code / Cursor / VS Code / 任意 MCP 框架）的完整配置模板见 [`mcp_config_example.json`](mcp_config_example.json)。挂载后即可在对话中直接调用 12 个工具（问答 / 检索 / 入库 / 文档管理 / 校验）：

```
policy_ask（问答，含逐句引用与附图）、policy_search、policy_get_chunks、
policy_ingest、policy_doc_index、policy_list_documents、policy_list_chunks、
policy_delete_document、policy_verify、policy_vision_check、policy_review_list/decide
```

### ③ 命令行入库与提问

```bat
ingest.bat "docs\policy.pdf" my-policy 2024-03-01 2024-03-01
python scripts\query.py ask "月度阻塞费用如何计算？"
```

### ④ 冒烟自检

```bash
python -m pytest tests/unit -q        # 单元测试全绿
python scripts/smoke.py               # 端到端冒烟
```

## 实现方法

问答链路为确定性优先的多级流水线（B0–B9）：

1. **检索方案**（B1）：LLM 顺带输出问题语义类型（枚举/数值/比较/定义/流程/分析）与量纲，作为校验闸第一信号；正则判定降级为回退与交叉校验（LLM 网关不可用时自动回退）。
2. **混合召回**（B3/B4）：BM25（正文+路径面包屑+条款别名词元）与向量双路，条款引用排序加权；同条枚举兄弟块、标题家族子块、公式页邻域三层证据扩展。
3. **生成**（B6）：带逐句引用的结构化作答；分析类问题只对锚定事实做【推导】，缺证据部分显式声明不覆盖。
4. **确定性硬闸**（B7）：数值溯源（问题前提与证据数值分流白名单）、方向词绑定、覆盖闸防"问数值答无数"（三态启停 ARMED/SOFT/OFF）、引文逐字多层比对（归一包含 → 快照吸附 → 宽松层 → 降级转述，编造数值的引文永不放行）。
5. **拒答裁决**（B9）：只有"确定性校验确认失败 + 补救重写仍失败"才拒答；部分可答走部分作答并注明缺口。

入库侧：PDF/DOCX/MD 解析 → 中文结构树（章/节/条/款/项）→ 稳定分块 ID → 公式/图表富化队列（本地 MinerU 公式转写 + OCR/VLM 图表双源转写 + 数值枚举预标注）→ 混合索引渐进升级。

## 测试结果

最近一轮双入口全量测试（100 题 × Web GUI 与 MCP 双端口，题型覆盖单一提问 / 组合提问 / 单一+分析 / 组合+分析 各 25 题，另设 2 道库外拒答探针；判分为金提示正则 + 质量红线机判）：

| 指标 | Web GUI | MCP |
|---|---|---|
| 回答率 | 99% | 98% |
| 拒答率（含误拒） | 1% | 2% |
| 已答准确率（金提示全中 / 已答） | 95.0% | 99.0% |
| 召回率（金提示条目） | 96.2% | 97.2% |
| 库外必拒探针 | 2/2 | 2/2 |
| 双端同题判定一致率 | 93% | |

- 已答准确率与召回率均为**金提示逐条正则核对**的机判结果，非主观评分。
- 拒答探针验证"库外问题必拒、不编造"的产品红线。
- 单元测试 263 例全绿（含数值/引文硬闸、拒答语义、结构树、公式链路回归）。

## 当前版本与路线图

**当前版本：v0.4.0**（2026-10）

已实现：精准问答三入口（GUI/REST/MCP）、结构化索引与原文 PDF 跳转、公式/图表富化、数值与引文双硬闸、问题语义分类驱动的校验闸、部分作答路由、文档版本归族、入库数值枚举预标注。

**未实现 / 规划中：**

| 版本 | 计划 |
|---|---|
| v0.4.x | 库外/预测类问题在拆分前整题拒（防探针被拆分后泄漏作答）；生成端引文纪律提示词进一步收紧 |
| v0.5.0 | 数值预标注接入富化任务队列（增量/断点续跑）；表格行列归属标注全面应用到比价类问答；部分作答的"覆盖度"显式标注 |
| v0.6.0 | 多文档版本差异对比视图增强；多库路由与库级权限；大库增量索引性能优化 |
| v1.0.0 | Linux/macOS 一键安装包；服务化部署（Docker）；检索评测集与自动回归流水线开源 |

## 限制

- **图表数值依赖视觉能力**：无 VLM API Key 时图表内数值不转写，相关数值问题按设计拒答（宁拒勿幻）。
- **公式转写**依赖本地 MinerU 工具链；复杂手写/扫描公式可能仅提供原文附图而无文本转写。
- **扫描版 PDF** 若无文本层需先 OCR（否则入库被拒收）。
- **分析类回答**严格锚定文档证据：文档未覆盖的推论会显式声明"证据未覆盖"，不做无依据外推。
- **知识边界**：仅回答入库文档覆盖的内容；库外问题一律拒答。
- 大规模并发 / 多副本部署为后续版本目标，当前定位为单机个人/小团队知识库。

## 参与贡献

Issue 与 PR 均欢迎。提交前请确保：

1. `python -m pytest tests/unit -q` 全绿；
2. 不提交任何密钥、`.env`、运行时数据（`.gitignore` 已覆盖）；
3. 机制级修复优先于个案补丁（禁按文档特判）；
4. 涉及提示词变更时同步 bump 版本号并附回归数据。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<div align="center">

**accurdocs** —— 让每一条结论都经得起逐字回查 · [Issues](https://github.com/Jaxmy/accurdocs/issues) · [Releases](https://github.com/Jaxmy/accurdocs/releases)

</div>
