"""R2-B 批量数值预标注：table/chart 块 LLM 枚举 → meta.numeric_spans（幂等可续跑）。

用法：python scripts/r2_annotate_numspan.py [doc_key前缀 ...]   # 缺省两目标文档
输出：逐块 upsert meta（numeric_spans）；enrich_cache 缓存跳过已标块。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(_PROJ_ROOT / "src"))

from policyrag.ingest.numspan import (annotate_numeric_spans,  # noqa: E402
                                      numspan_cache_key, validate_spans)
from policyrag.index.stores.sqlite_meta import MetaStore  # noqa: E402

TARGETS = sys.argv[1:] or ["liaoning%", "%无边界%"]
meta = MetaStore(_PROJ_ROOT / "data" / "meta.sqlite3")
llm = None


def get_llm():
    global llm
    if llm is None:
        from policyrag.settings import make_llm
        llm = make_llm()
    return llm


def file_hash_of_doc(doc_key: str) -> str:
    return doc_key   # 块哈希已含内容；doc 级哈希仅作缓存命名空间


def main():
    n_done = n_skip = n_empty = n_rowskip = 0
    t0 = time.time()
    work = []          # (dk, cid, m, extra, text, key)——LLM 并行、写库串行
    for pat in TARGETS:
        rows = meta.list_documents()
        docs = [r["doc_key"] for r in rows if r["doc_key"].startswith(pat.rstrip("%"))
                or pat.strip("%") in r["doc_key"]]
        for dk in docs:
            metas = meta.all_chunk_metas()
            for cid, m in metas.items():
                if m.get("doc_key") != dk:
                    continue
                if m.get("kind") not in ("table", "chart"):
                    continue
                try:
                    extra = json.loads(m.get("meta_json") or "{}")
                except Exception:  # noqa: BLE001
                    extra = {}
                if extra.get("numeric_spans"):
                    n_skip += 1
                    continue
                # 表行碎块跳过（值在主表块内，demanded 并集仍经运行时正则覆盖）——
                # 逐行 LLM 调用是批处理 5 小时的根因
                if extra.get("row_index") is not None or extra.get("parent"):
                    n_rowskip = n_rowskip + 1
                    continue
                text = m.get("text") or ""
                if len(text) < 12:
                    continue
                # enrich_cache 幂等（PK：file_hash=doc 命名空间, page=0, kind=numspan:块哈希）
                key = numspan_cache_key(file_hash_of_doc(dk), cid, text)
                cached = meta.enrich_cache_get(file_hash_of_doc(dk), 0, key)
                spans = (cached or {}).get("spans") if cached else None
                if spans is not None:
                    if not spans:
                        n_empty += 1
                        continue
                    extra["numeric_spans"] = spans
                    meta.upsert_chunk(
                        chunk_id=cid, doc_key=dk,
                        path_key=m.get("path_key") or "", breadcrumb=m.get("breadcrumb") or "",
                        text=m.get("text") or "", text_norm=m.get("text_norm") or "",
                        kind=m.get("kind") or "table", page=m.get("page"),
                        meta=extra)
                    n_done += 1
                    continue
                work.append((dk, cid, m, extra, text, key))

    # LLM 4 路并行（网关延迟主导；sqlite 写入回主线程串行）
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(job):
        dk, cid, m, extra, text, key = job
        spans = annotate_numeric_spans(get_llm(), text, kind=m.get("kind") or "table")
        try:
            meta.enrich_cache_put(file_hash_of_doc(dk), 0, key, {"spans": spans})
        except Exception:  # noqa: BLE001
            pass
        return job, spans

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_one, j) for j in work]
        for fut in as_completed(futs):
            (dk, cid, m, extra, text, key), spans = fut.result()
            if not spans:
                n_empty += 1
                continue
            extra["numeric_spans"] = spans
            meta.upsert_chunk(
                chunk_id=cid, doc_key=dk,
                path_key=m.get("path_key") or "", breadcrumb=m.get("breadcrumb") or "",
                text=m.get("text") or "", text_norm=m.get("text_norm") or "",
                kind=m.get("kind") or "table", page=m.get("page"),
                meta=extra)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  ... annotated {n_done} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: annotated={n_done} cached/skip={n_skip} empty={n_empty} "
          f"row_chunks_skipped={n_rowskip} in {time.time() - t0:.0f}s")
    print("rebuild index...")
    from policyrag.settings import make_llm, service_config
    from policyrag.service.core import CoreService
    svc = CoreService(make_llm(), service_config())
    print("index:", svc.rebuild_index())


if __name__ == "__main__":
    main()
