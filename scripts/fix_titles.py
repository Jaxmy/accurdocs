"""存量垃圾标题批量修复。

红线：只修 documents.title 显示字段——doc_id/doc_key 冻结（改 doc_id 必改
doc_key 进而改 make_chunk_id 材料，违 ID 稳定红线）。

用法：PYTHONUTF8=1 .venv312/Scripts/python.exe scripts/fix_titles.py [--apply]
默认 dry-run 只打印；--apply 才落库。
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("PYTHONUTF8", "1")

from policyrag.ingest.identity import extract_title  # noqa: E402
from policyrag.index.stores.sqlite_meta import MetaStore  # noqa: E402

_GARBAGE_RE = re.compile(
    r"^[—\-–—\s]*\d{1,4}[—\-–—\s]*$"          # — 93 —
    r"|^第\s*\d{1,3}\s*页"
    r"|^\d{4}\s*年\s*\d{0,2}\s*月?$"
    r"|^[\d\s\W_]+$"                           # 纯数字/符号
    r"|(下载|扫描件|仅供|内部学习)"
    r"|^(doc_|upload|untitled)"
)


def is_garbage(title: str) -> bool:
    t = (title or "").strip()
    return (not t) or bool(_GARBAGE_RE.search(t)) or len(t) < 3


def main() -> None:
    apply = "--apply" in sys.argv
    meta = MetaStore("data/meta.sqlite3")
    n_fix = 0
    for row in meta.list_documents():
        title = row["title"] or ""
        if not is_garbage(title):
            continue
        chunks = meta.chunks_of_doc(row["doc_key"])
        text = "\n".join(c["text"] for c in chunks[:80])
        new_title, src = extract_title(text, row["doc_id"])
        print(f"[{row['doc_key']}] {title!r} -> {new_title!r}  (src={src})")
        if apply and new_title and not is_garbage(new_title):
            meta.set_title(row["doc_key"], new_title)
            n_fix += 1
    print(f"{'已修复' if apply else 'dry-run'}: {n_fix} 个标题"
          f"{'' if apply else '（加 --apply 落库）'}")


if __name__ == "__main__":
    main()
