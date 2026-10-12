"""测试战役 MCP 入口驱动：stdio 挂载 policyrag MCP，逐输入 tools/call policy_ask 落盘。

与聊天界面 mcp__policyrag__* 工具调用同一协议（initialize → tools/call）。
用法：python scripts/campaign_mcp.py [输入ID ...]   # 缺省全部；--spot 跑抽查
"""
import asyncio

from pathlib import Path as _P
_PROJ_ROOT = _P(__file__).resolve().parents[1]   # 项目根（可移植）
import json
import re
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = _PROJ_ROOT / "data" / "test_campaign_2026-10"
INPUTS = json.loads((ROOT / "inputs.json").read_text(encoding="utf-8"))
SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "policyrag.adapters.mcp.server"],
    env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
         "PATH": os.environ.get("PATH", "")},
)


def text_of(result) -> str:
    parts = []
    for c in getattr(result, "content", []) or []:
        if getattr(c, "type", "") == "text":
            parts.append(c.text)
    return "\n".join(parts)


async def run(inputs, spot=False):
    async with stdio_client(SERVER) as (r, w):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            tools = await sess.list_tools()
            names = [t.name for t in tools.tools]
            print("tools:", names)
            if not spot:
                for item in inputs:
                    iid = item["id"]
                    out = ROOT / "mcp_out" / f"{iid}.json"
                    if out.exists():
                        print(f"[{iid}] 已有记录，跳过")
                        continue
                    res = await sess.call_tool("policy_ask", {
                        "query": item["query"], "mode": item["mode"]})
                    txt = text_of(res)
                    # policy_ask 返回 JSON 文本（envelope）；记录原文+解析
                    try:
                        env = json.loads(txt)
                    except Exception:  # noqa: BLE001
                        env = {"raw": txt}
                    rec = {"input_id": iid, "query": item["query"], "mode": item["mode"],
                           "envelope": env, "text_len": len(txt)}
                    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
                    st = env.get("status", "?")
                    ans = (env.get("answer") or "")[:40].replace("\n", " ")
                    n_id = len(re.findall(r"\[[0-9a-fA-F]{16}\]", env.get("answer") or ""))
                    print(f"[{iid}] status={st} ans={ans} id_marks={n_id}")
            else:
                # 抽查：confirm_plan / policy_verify / policy_search / policy_list_chunks
                p = await sess.call_tool("policy_ask", {
                    "query": "月度阻塞费用如何计算？", "mode": "quote",
                    "confirm_plan": True})
                (ROOT / "mcp_out" / "spot_confirm_plan.json").write_text(
                    text_of(p), encoding="utf-8")
                print("spot confirm_plan:", text_of(p)[:150])
                v = await sess.call_tool("policy_verify", {
                    "answer": "测试校验链路", "citations": [{"quote": "测试", "chunk_id": "x"}]})
                print("spot verify:", text_of(v)[:150])
                s = await sess.call_tool("policy_search", {"query": "阻塞费用", "k": 5})
                print("spot search:", text_of(s)[:150])
                l = await sess.call_tool("policy_list_chunks", {"q": "阻塞", "limit": 5})
                print("spot list_chunks:", text_of(l)[:150])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = set(args)
    spot = "--spot" in sys.argv
    sel = [i for i in INPUTS if not only or i["id"] in only]
    asyncio.run(run(sel, spot=spot))
