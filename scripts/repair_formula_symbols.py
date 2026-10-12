"""存量公式块符号表修复：meta.symbols → 文本【符号】行 + 重建索引。

背景：公式升级曾把"式中"符号表只写进 meta.symbols、不进 text——
公式块纯 LaTeX 文本对中文问题不可检索、符号含义不可逐字引用（发电侧
省内实时公式 f_227_01 实测漏检）。chunk_id 不变（ID 红线），只补文本。

用法：python scripts/repair_formula_symbols.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyrag.ingest.enrichment import merge_symbol_lines  # noqa: E402


def main() -> None:
    from policyrag.service.core import CoreService
    from policyrag.settings import make_llm, service_config

    svc = CoreService(make_llm(), service_config())
    meta = svc.meta
    n_fixed = 0
    for doc in meta.list_documents():
        for row in meta.chunks_of_doc(doc["doc_key"]):
            if row.get("kind") != "formula":
                continue
            try:
                extra = json.loads(row.get("meta_json") or "{}")
            except Exception:  # noqa: BLE001
                continue
            symbols = extra.get("symbols") or []
            new_text = merge_symbol_lines(row.get("text") or "", symbols)
            if new_text == (row.get("text") or ""):
                continue
            meta.upsert_chunk(
                chunk_id=row["chunk_id"], doc_key=row["doc_key"],
                path_key=row.get("path_key") or "",
                breadcrumb=row.get("breadcrumb") or "",
                text=new_text, text_norm=new_text,
                kind=row.get("kind") or "formula",
                page=row.get("page"), summary=row.get("summary"), meta=extra,
            )
            n_fixed += 1
    meta.commit()
    print(f"repaired {n_fixed} formula chunks")
    stats = svc.rebuild_index()
    print("index rebuilt:", stats)


if __name__ == "__main__":
    main()
