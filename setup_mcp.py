#!/usr/bin/env python
"""accurdocs MCP 一键挂载：生成本机 MCP 配置并（可选）注册到 Claude Code。

用法（在 accurdocs 项目根执行）：
    python setup_mcp.py              # 生成 .mcp.json + 打印各客户端配置
    python setup_mcp.py --claude     # 额外执行 claude mcp add 注册
    python setup_mcp.py --port 8100  # 顺带打印 HTTP 传输启动命令

零依赖（纯标准库）；不读取、不打印任何密钥。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def mcp_command() -> dict:
    """按平台给出 MCP stdio 启动命令（相对本机绝对路径，落盘配置用）。"""
    if os.name == "nt":
        bat = os.path.join(ROOT, "start_mcp.bat")
        return {"command": "cmd", "args": ["/c", bat]}
    py = sys.executable or "python"
    return {"command": py, "args": ["-m", "policyrag.adapters.mcp.server"], "cwd": ROOT}


def main() -> int:
    ap = argparse.ArgumentParser(description="accurdocs MCP one-click setup")
    ap.add_argument("--claude", action="store_true",
                    help="同时执行 claude mcp add accurdocs 注册到 Claude Code")
    ap.add_argument("--port", type=int, default=0,
                    help="打印 HTTP 传输启动命令（给不支持 stdio 的客户端）")
    args = ap.parse_args()

    cfg = {"mcpServers": {"accurdocs": mcp_command()}}
    out = os.path.join(ROOT, ".mcp.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[ok] 已生成 {out}")
    print("     （Claude Code 在本项目根运行时会自动读取 .mcp.json）")

    print("\n其他客户端通用配置（mcpServers 段）：")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    print("\n更多客户端模板见 mcp_config_example.json")

    if args.claude:
        claude = shutil.which("claude")
        if not claude:
            print("\n[warn] 未找到 claude 命令行——请手动把上面配置并入 ~/.claude.json 的 mcpServers")
        else:
            cmd = [claude, "mcp", "add", "accurdocs", "--",
                   cfg["command"], *cfg["args"]]
            print(f"\n[run] {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, cwd=ROOT)
                print("[ok] 已注册到 Claude Code（claude mcp list 可验证）")
            except subprocess.CalledProcessError as e:
                print(f"[warn] 注册失败（{e}），请手动并入配置")
    if args.port:
        print("\nHTTP 传输（给不支持 stdio 的客户端）：")
        print(f"  python -m policyrag.adapters.mcp.server --http --port {args.port}")
        print(f"  然后连接 http://127.0.0.1:{args.port}/mcp")
    print("\n注意：MCP 服务读取项目根 .env 的 LLM 配置——先完成 .env 再挂载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
