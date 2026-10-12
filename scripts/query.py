"""CLI：入库与查询。

用法：
  python scripts/query.py ingest <file> [--doc-id X] [--version V] [--effective-date D]
  python scripts/query.py ask "问题" [--as-of 2026-01-01] [--analyze]
  python scripts/query.py search "问题" [-k 10]
  python scripts/query.py verify "答案文本" --citations citations.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from policyrag.llm.router import LLMRouter  # noqa: E402
from policyrag.models.retrieval import QueryRequest  # noqa: E402
from policyrag.service.core import CoreService, ServiceConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("file")
    p_ing.add_argument("--doc-id")
    p_ing.add_argument("--version")
    p_ing.add_argument("--effective-date")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--as-of", dest="as_of")
    p_ask.add_argument("--analyze", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("question")
    p_search.add_argument("-k", type=int, default=10)

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("answer_text")
    p_ver.add_argument("--citations", required=True, help="JSON 文件路径")

    args = ap.parse_args()
    from policyrag.settings import make_llm, service_config
    svc = CoreService(make_llm(), service_config())

    if args.cmd == "ingest":
        report = svc.ingest(args.file, doc_id=args.doc_id, version=args.version,
                            effective_date=args.effective_date)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.cmd == "ask":
        req = QueryRequest(
            query=args.question,
            as_of=args.as_of or "",
            mode="analyze" if args.analyze else "quote",
            request_id=f"cli_{uuid.uuid4().hex[:10]}",
        )
        env = asyncio.run(svc.ask(req))
        print(json.dumps(env.to_dict(), ensure_ascii=False, indent=2))
    elif args.cmd == "search":
        print(json.dumps(svc.search(args.question, k=args.k), ensure_ascii=False, indent=2))
    elif args.cmd == "verify":
        cites = json.loads(open(args.citations, encoding="utf-8").read())
        print(json.dumps(svc.verify(args.answer_text, cites), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
