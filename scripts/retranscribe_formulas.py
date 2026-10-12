"""存量占位公式补转写：按文档重跑 _process_formulas（含 OCR 兜底）。

用法：python scripts/retranscribe_formulas.py [doc_key 片段]
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from policyrag.index.stores.sqlite_meta import MetaStore  # noqa: E402
from policyrag.ingest.enrichment import RichWorker  # noqa: E402

ROOT = _PROJ_ROOT


def main(filt: str = "") -> None:
    m = MetaStore(str(ROOT / "data" / "meta.sqlite3"))
    # 占位公式块按 doc_group 聚页
    rows = m._conn().execute(
        "SELECT doc_key, page FROM chunks WHERE kind='formula' "
        "AND text LIKE '%转写进行中%'").fetchall()
    by_doc: dict[str, list[int]] = {}
    for r in rows:
        if filt and filt not in r["doc_key"]:
            continue
        by_doc.setdefault(r["doc_key"], []).append(r["page"])
    w = RichWorker(m)
    for doc_key, pages in sorted(by_doc.items()):
        # 用该文档任一历史 formula job 的 payload（file_hash/file_path 复用）
        job = m._conn().execute(
            "SELECT payload_json FROM rich_jobs WHERE kind='formula' "
            "AND doc_key=? LIMIT 1", (doc_key,)
        ).fetchone()
        if not job:
            print(f"[skip] {doc_key}: 无历史 payload")
            continue
        payload = json.loads(job["payload_json"])
        payload["pages"] = sorted(set(pages))
        jid = f"rj_re{uuid.uuid4().hex[:8]}"
        m.job_create(jid, "formula", doc_key, payload) if hasattr(m, "job_create") else None
        print(f"[run] {doc_key}: {len(payload['pages'])} 页占位待补 → {payload['pages']}")
        w._process_formulas(jid, doc_key, payload)
        left = m._conn().execute(
            "SELECT count(*) n FROM chunks WHERE doc_key=? AND kind='formula' "
            "AND text LIKE '%转写进行中%'", (doc_key,)).fetchone()["n"]
        print(f"[done] {doc_key}: 剩余占位 {left}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
