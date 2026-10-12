"""从辽宁规则文档抽含明确数值/期限的条款片段，供出题（金标准）。"""
import json
import re
import sqlite3

import os
from pathlib import Path
con = sqlite3.connect(str(Path(__file__).resolve().parents[1] / "data" / "meta.sqlite3"))
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT chunk_id, breadcrumb, page, text FROM chunks WHERE doc_key LIKE '辽宁省%'"
).fetchall()
print("total chunks:", len(rows))

NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|元|亿|万|年|月|日|小时|个|次|万千瓦|兆瓦|千瓦时|吨)")
CLUE_RE = re.compile(r"(不得|不应|应当|应为|按照|按其|上限|下限|不低于|不超过|为每|为.{0,6}\d|标准为|限额)")

cands = []
for r in rows:
    t = (r["text"] or "").strip()
    if not (30 <= len(t) <= 400):
        continue
    nums = NUM_RE.findall(t)
    if len(nums) >= 1 and CLUE_RE.search(t):
        cands.append((r["page"] or 0, r["breadcrumb"] or "", t, r["chunk_id"]))

cands.sort(key=lambda x: x[0])
print("candidates:", len(cands))
seen = set()
picked = 0
for page, bc, t, cid in cands:
    key = t[:30]
    if key in seen:
        continue
    seen.add(key)
    picked += 1
    print(f"\n=== p{page} {bc[:40]}")
    print(t[:300])
    if picked >= 60:
        break
