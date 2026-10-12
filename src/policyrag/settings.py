r"""启动配置：.env 加载 + LLM 工厂（独立启动用）。

从项目根 .env 读取密钥/URL/模型名——进程环境变量优先于 .env。
所有入口（CLI / REST / MCP）共用本模块，配置只写一处。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_env", "make_llm", "PROJECT_ROOT"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 项目根
ENV_FILE = PROJECT_ROOT / ".env"

_LOADED = False


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """加载 .env（KEY=VALUE 行）。不覆盖已存在的进程环境变量。幂等。"""
    global _LOADED
    env_path = Path(path) if path else ENV_FILE
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    _LOADED = True
    return dict(os.environ)


def make_llm():
    """按 POLICYRAG_LLM 构造 LLM 客户端（anthropic | deepseek | glm）。"""
    load_env()
    provider = os.environ.get("POLICYRAG_LLM", "anthropic").lower()

    if provider == "anthropic":
        from .llm.anthropic_client import AnthropicClient
        return AnthropicClient()  # 读 ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL / POLICYRAG_MODEL
    if provider == "deepseek":
        from .llm.deepseek_client import DeepSeekClient
        return DeepSeekClient(
            model=os.environ.get("POLICYRAG_MODEL") or "deepseek-chat",
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    if provider == "glm":
        from .llm.glm_client import GLMClient
        return GLMClient(model=os.environ.get("POLICYRAG_MODEL") or "glm-4-plus")
    raise ValueError(f"未知 POLICYRAG_LLM: {provider}")


def qkind_mode() -> str:
    """覆盖闸双信号模式（R2-A）：regex=纯正则 / llm=分类优先 / cross=交叉保守。

    默认 llm——A/B 离线回放（data/test_campaign_2026-10/r2/ab_gate_report.json，
    20 题分歧探针 LLM 分类 20/20）：llm 档补上正则漏报的无疑问词求值类
    （"请给出市值"），非数值类误拒 0，加 F5 部分命中阶梯后误开闸代价=PARTIAL。
    LLM 信号缺失（网关挂/旧 plan）自动回退正则——`哪几`/量词正则即兜底。"""
    v = (os.environ.get("POLICYRAG_QKIND_MODE") or "llm").strip().lower()
    return v if v in ("regex", "llm", "cross") else "llm"


def service_config():
    """CoreService 配置（数据目录/嵌入后端）。"""
    from .service.core import ServiceConfig
    load_env()
    return ServiceConfig(
        data_dir=os.environ.get("POLICYRAG_DATA", str(PROJECT_ROOT / "data")),
        trace_dir=os.environ.get("POLICYRAG_TRACES", str(PROJECT_ROOT / "traces")),
        embed_backend=os.environ.get("POLICYRAG_EMBED", "hash"),
    )


# ---------------------------------------------------------------------------
# 配置读写（GUI 用）——.env 语义：进程环境变量优先于 .env
# ---------------------------------------------------------------------------

_KEY_FIELDS = {
    "provider": "POLICYRAG_LLM",
    "model": "POLICYRAG_MODEL",
    "base_url": "ANTHROPIC_BASE_URL",
    "api_key": "ANTHROPIC_AUTH_TOKEN",
    "embed": "POLICYRAG_EMBED",
    "vision_model": "POLICYRAG_VISION_MODEL",
    "vision_base_url": "POLICYRAG_VISION_BASE_URL",
    "vision_api_key": "POLICYRAG_VISION_API_KEY",
}


def read_config() -> dict:
    load_env()
    return {
        "provider": os.environ.get("POLICYRAG_LLM", "anthropic"),
        "model": os.environ.get("POLICYRAG_MODEL", ""),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL", ""),
        "embed": os.environ.get("POLICYRAG_EMBED", "hash"),
        "key_set": bool(os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")),
        "vision_model": os.environ.get("POLICYRAG_VISION_MODEL", ""),
        "vision_base_url": os.environ.get("POLICYRAG_VISION_BASE_URL", ""),
        "vision_key_set": bool(os.environ.get("POLICYRAG_VISION_API_KEY")),
    }


def write_config(updates: dict) -> None:
    """更新 .env 与当前进程环境。api_key 为空则不改动。"""
    load_env()
    mapping = {}
    for field, env_key in _KEY_FIELDS.items():
        if field in updates and updates[field] not in (None, ""):
            mapping[env_key] = str(updates[field])
    if "api_key" in updates and not updates["api_key"]:
        mapping.pop("ANTHROPIC_AUTH_TOKEN", None)
    if "vision_api_key" in updates and not updates["vision_api_key"]:
        mapping.pop("POLICYRAG_VISION_API_KEY", None)

    # 合并进 .env（按行替换或追加）
    lines: list[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    for env_key, val in mapping.items():
        prefix = env_key + "="
        idx = next((i for i, ln in enumerate(lines) if ln.startswith(prefix)), None)
        if idx is not None:
            lines[idx] = f"{env_key}={val}"
        else:
            lines.append(f"{env_key}={val}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ.update(mapping)
    global _LOADED
    _LOADED = True
