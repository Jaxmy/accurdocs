"""SQLite 元数据/审计存储：文档注册表、版本链、chunk registry、审计日志。

WAL 模式 + busy_timeout（计划 §五.11）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ...models.ids import desurrogate

__all__ = ["MetaStore"]


def _t(s: Any) -> Any:
    """文本字段边界清理：孤立代理字符无法入 SQLite(UTF-8)。非 str 原样返回。"""
    return desurrogate(s) if isinstance(s, str) else s

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_key     TEXT PRIMARY KEY,          -- doc_id@version
    doc_id      TEXT NOT NULL,
    version     TEXT NOT NULL,
    title       TEXT NOT NULL,
    effective_date TEXT,
    repealed_date  TEXT,
    status      TEXT NOT NULL DEFAULT 'active',  -- active|repealed|temporal_unknown
    meta_json   TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_key     TEXT NOT NULL,
    path_key    TEXT NOT NULL,
    breadcrumb  TEXT NOT NULL,
    text        TEXT NOT NULL,
    text_norm   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    page        INTEGER,
    summary     TEXT NOT NULL DEFAULT '',
    meta_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS supersession (
    old_doc_key TEXT NOT NULL,
    new_doc_key TEXT NOT NULL,
    PRIMARY KEY (old_doc_key, new_doc_key)
);
CREATE TABLE IF NOT EXISTS audit_log (
    request_id  TEXT PRIMARY KEY,
    ts          REAL NOT NULL,
    channel     TEXT NOT NULL,
    tool        TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status      TEXT NOT NULL,
    trace_ref   TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    usage_json  TEXT NOT NULL DEFAULT '{}',
    verification_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS review_tasks (
    task_id     TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    query       TEXT NOT NULL,
    as_of       TEXT NOT NULL DEFAULT '',
    draft_answer TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    envelope_json TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected
    decision_note TEXT NOT NULL DEFAULT '',
    final_answer TEXT NOT NULL DEFAULT '',
    decided_at  REAL
);
CREATE TABLE IF NOT EXISTS enrich_cache (
    file_hash  TEXT NOT NULL,
    page       INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (file_hash, page, kind)
);
CREATE TABLE IF NOT EXISTS rich_jobs (
    job_id     TEXT PRIMARY KEY,
    doc_key    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'queued',
    progress   REAL NOT NULL DEFAULT 0,
    note       TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_key);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_tasks(status);
"""

# v0.3 增量 schema（守卫式迁移；老库由 _ensure_columns 补列）
_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS libraries (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at REAL,
  auto INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS doc_membership (
  doc_key TEXT NOT NULL,
  library_id TEXT NOT NULL,
  source TEXT DEFAULT 'manual',
  PRIMARY KEY (doc_key, library_id)
);
"""


class MetaStore:
    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path, timeout=30.0)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    def _init_db(self) -> None:
        c = self._conn()
        c.executescript(_SCHEMA)
        c.executescript(_SCHEMA_V2)   # 守卫式迁移：v0.3 身份列 + 分库表
        self._ensure_columns()
        c.commit()

    def _ensure_columns(self) -> None:
        """守卫式 ALTER：documents 身份列 / audit 历史列（PRAGMA 检查后补）。"""
        c = self._conn()
        want = {
            "doc_group": "TEXT DEFAULT ''",
            "version_seq": "INTEGER DEFAULT 1",
            "is_latest": "INTEGER DEFAULT 1",
            "content_hash": "TEXT DEFAULT ''",
            "simhash": "TEXT DEFAULT ''",
            "doc_no": "TEXT DEFAULT ''",
        }
        cols = {r["name"] for r in c.execute("PRAGMA table_info(documents)")}
        for col, ddl in want.items():
            if col not in cols:
                c.execute(f"ALTER TABLE documents ADD COLUMN {col} {ddl}")
        audit_cols = {r["name"] for r in c.execute("PRAGMA table_info(audit_log)")}
        for col, ddl in {"query_text": "TEXT DEFAULT ''",
                         "answer_preview": "TEXT DEFAULT ''",
                         "parent_request_id": "TEXT DEFAULT ''"}.items():
            if col not in audit_cols:
                c.execute(f"ALTER TABLE audit_log ADD COLUMN {col} {ddl}")

    # -- documents -----------------------------------------------------------

    def upsert_document(
        self,
        doc_id: str,
        version: str,
        title: str,
        effective_date: str | None = None,
        repealed_date: str | None = None,
        status: str = "active",
        meta: dict[str, Any] | None = None,
        content_hash: str = "",
        simhash: str = "",
        doc_group: str = "",
        version_seq: int = 1,
        is_latest: int = 1,
        doc_no: str = "",
    ) -> None:
        c = self._conn()
        c.execute(
            """INSERT INTO documents(doc_key, doc_id, version, title, effective_date,
                                     repealed_date, status, meta_json, created_at,
                                     content_hash, simhash, doc_group, version_seq,
                                     is_latest, doc_no)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(doc_key) DO UPDATE SET
                 title=excluded.title, effective_date=excluded.effective_date,
                 repealed_date=excluded.repealed_date, status=excluded.status,
                 meta_json=excluded.meta_json, content_hash=excluded.content_hash,
                 simhash=excluded.simhash, doc_group=excluded.doc_group,
                 version_seq=excluded.version_seq, is_latest=excluded.is_latest,
                 doc_no=excluded.doc_no""",
            (_t(f"{doc_id}@{version}"), _t(doc_id), _t(version), _t(title),
             _t(effective_date), _t(repealed_date), _t(status),
             json.dumps(meta or {}, ensure_ascii=True), time.time(),
             _t(content_hash), _t(simhash), _t(doc_group) or _t(doc_id),
             version_seq, is_latest, _t(doc_no)),
        )
        c.commit()

    def set_title(self, doc_key: str, title: str) -> None:
        """只修显示标题（存量修复用）——doc_id/doc_key 冻结，chunk_id 不受影响。"""
        c = self._conn()
        c.execute("UPDATE documents SET title=? WHERE doc_key=?",
                  (_t(title), _t(doc_key)))
        c.commit()

    def set_latest_version(self, doc_group: str, latest_doc_key: str) -> None:
        """组内最新版标记：组内唯一 is_latest=1。"""
        c = self._conn()
        c.execute("UPDATE documents SET is_latest=0 WHERE doc_group=? OR doc_id=?",
                  (_t(doc_group), _t(doc_group)))
        c.execute("UPDATE documents SET is_latest=1 WHERE doc_key=?",
                  (_t(latest_doc_key),))
        c.commit()

    # -- 分库（libraries / doc_membership）--------------------------------------

    def upsert_library(self, lib_id: str, name: str, auto: bool = True) -> None:
        c = self._conn()
        c.execute(
            """INSERT INTO libraries(id, name, created_at, auto) VALUES(?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name""",
            (_t(lib_id), _t(name), time.time(), 1 if auto else 0))
        c.commit()

    def delete_library(self, lib_id: str) -> int:
        c = self._conn()
        n = c.execute("SELECT COUNT(*) FROM doc_membership WHERE library_id=?",
                      (_t(lib_id),)).fetchone()[0]
        c.execute("DELETE FROM doc_membership WHERE library_id=?", (_t(lib_id),))
        c.execute("DELETE FROM libraries WHERE id=?", (_t(lib_id),))
        c.commit()
        return n

    def list_libraries(self) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            """SELECT l.*, COUNT(m.doc_key) AS n_docs
               FROM libraries l LEFT JOIN doc_membership m ON l.id = m.library_id
               GROUP BY l.id ORDER BY l.name""").fetchall()
        return [dict(r) for r in rows]

    def assign_library(self, doc_key: str, library_id: str,
                       source: str = "manual") -> None:
        c = self._conn()
        c.execute(
            """INSERT INTO doc_membership(doc_key, library_id, source) VALUES(?,?,?)
               ON CONFLICT(doc_key, library_id) DO UPDATE SET source=excluded.source""",
            (_t(doc_key), _t(library_id), _t(source)))
        c.commit()

    def unassign_library(self, doc_key: str, library_id: str) -> None:
        c = self._conn()
        c.execute("DELETE FROM doc_membership WHERE doc_key=? AND library_id=?",
                  (_t(doc_key), _t(library_id)))
        c.commit()

    def libraries_of_doc(self, doc_key: str) -> list[str]:
        rows = self._conn().execute(
            "SELECT library_id FROM doc_membership WHERE doc_key=?",
            (_t(doc_key),)).fetchall()
        return [r["library_id"] for r in rows]

    def doc_keys_in_libraries(self, library_ids: list[str]) -> list[str]:
        if not library_ids:
            return []
        ph = ",".join("?" * len(library_ids))
        rows = self._conn().execute(
            f"SELECT DISTINCT doc_key FROM doc_membership WHERE library_id IN ({ph})",
            library_ids).fetchall()
        return [r["doc_key"] for r in rows]

    def get_document(self, doc_key: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM documents WHERE doc_key=?", (doc_key,)
        ).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM documents ORDER BY doc_id, version"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_key: str) -> int:
        """删除文档及其全部 chunk。返回删除的 chunk 数。级联清 registry。"""
        c = self._conn()
        n = c.execute("SELECT COUNT(*) FROM chunks WHERE doc_key=?", (doc_key,)).fetchone()[0]
        c.execute("DELETE FROM chunks WHERE doc_key=?", (doc_key,))
        c.execute("DELETE FROM documents WHERE doc_key=?", (doc_key,))
        c.execute("DELETE FROM supersession WHERE old_doc_key=? OR new_doc_key=?",
                  (doc_key, doc_key))
        c.commit()
        return int(n)

    def temporal_chunk_ids(self, as_of: str) -> set[str] | None:
        """as_of 生效时点过滤：只保留该日已生效且未废止的文档的 chunk。

        无任何文档带生效日期时返回 None（不做过滤，全部视为可用）。
        未标注日期的文档（temporal_unknown）保守纳入（不因缺日期而丢失答案）。
        """
        c = self._conn()
        dated = c.execute(
            "SELECT COUNT(*) FROM documents WHERE effective_date IS NOT NULL"
        ).fetchone()[0]
        if not dated:
            return None
        rows = c.execute(
            """SELECT c.chunk_id FROM chunks c JOIN documents d ON c.doc_key=d.doc_key
               WHERE (d.effective_date IS NULL OR d.effective_date <= ?)
                 AND (d.repealed_date IS NULL OR d.repealed_date > ?)""",
            (as_of, as_of),
        ).fetchall()
        return {r["chunk_id"] for r in rows}

    def filter_chunk_ids(self, scope_chunk_ids: list[str] | None,
                         scope_doc_keys: list[str] | None) -> set[str]:
        """范围过滤：返回允许参与检索的 chunk_id 集合。两者都空 = 全库。"""
        c = self._conn()
        allowed: set[str] = set()
        if scope_chunk_ids:
            ph = ",".join("?" * len(scope_chunk_ids))
            rows = c.execute(
                f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({ph})", scope_chunk_ids
            ).fetchall()
            allowed |= {r["chunk_id"] for r in rows}
        if scope_doc_keys:
            ph = ",".join("?" * len(scope_doc_keys))
            rows = c.execute(
                f"SELECT chunk_id FROM chunks WHERE doc_key IN ({ph})", scope_doc_keys
            ).fetchall()
            allowed |= {r["chunk_id"] for r in rows}
        return allowed

    def list_chunks_filtered(self, doc_key: str | None = None, q: str | None = None,
                             limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        """分块清单（GUI 勾选浏览用）。q 做 LIKE 模糊过滤。"""
        sql = ("SELECT c.chunk_id, c.doc_key, c.breadcrumb, c.kind, c.page, c.summary, "
               "d.title AS doc_title, substr(c.text,1,120) AS preview "
               "FROM chunks c JOIN documents d ON c.doc_key=d.doc_key WHERE 1=1")
        args: list[Any] = []
        if doc_key:
            sql += " AND c.doc_key=?"
            args.append(doc_key)
        if q:
            sql += " AND (c.text LIKE ? OR c.breadcrumb LIKE ?)"
            args += [f"%{q}%", f"%{q}%"]
        sql += " ORDER BY c.doc_key, c.path_key LIMIT ? OFFSET ?"
        args += [limit, offset]
        rows = self._conn().execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def add_supersession(self, old_doc_key: str, new_doc_key: str) -> None:
        c = self._conn()
        c.execute(
            "INSERT OR IGNORE INTO supersession(old_doc_key, new_doc_key) VALUES(?,?)",
            (old_doc_key, new_doc_key),
        )
        c.commit()

    def supersession_closure(self, doc_key: str) -> list[str]:
        """废止链传递闭包：从 doc_key 出发能到达的所有后续版本。"""
        seen: list[str] = []
        frontier = [doc_key]
        while frontier:
            cur = frontier.pop()
            rows = self._conn().execute(
                "SELECT new_doc_key FROM supersession WHERE old_doc_key=?", (cur,)
            ).fetchall()
            for r in rows:
                nk = r["new_doc_key"]
                if nk not in seen:
                    seen.append(nk)
                    frontier.append(nk)
        return seen

    # -- chunks ---------------------------------------------------------------

    def upsert_chunk(self, chunk_id: str, doc_key: str, path_key: str, breadcrumb: str,
                     text: str, text_norm: str, kind: str, page: int | None = None,
                     summary: str = "", meta: dict[str, Any] | None = None) -> None:
        c = self._conn()
        c.execute(
            """INSERT INTO chunks(chunk_id, doc_key, path_key, breadcrumb, text,
                                  text_norm, kind, page, summary, meta_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(chunk_id) DO UPDATE SET
                 text=excluded.text, text_norm=excluded.text_norm,
                 summary=excluded.summary, meta_json=excluded.meta_json""",
            (_t(chunk_id), _t(doc_key), _t(path_key), _t(breadcrumb), _t(text),
             _t(text_norm), _t(kind), page, _t(summary),
             json.dumps(meta or {}, ensure_ascii=True)),
        )

    def commit(self) -> None:
        self._conn().commit()

    def replace_document_chunks(self, doc_key: str, keep_ids: set | list) -> int:
        """替换语义：删除该 doc_key 下不在 keep_ids 中的 chunk（防重入库孤儿块）。

        必须在本次全部 upsert_chunk 之后、rebuild_index 之前调用。
        返回删除数。
        """
        keep = set(keep_ids)
        rows = self._conn().execute(
            "SELECT chunk_id FROM chunks WHERE doc_key=?", (doc_key,)
        ).fetchall()
        stale = [r["chunk_id"] for r in rows if r["chunk_id"] not in keep]
        if stale:
            self._conn().executemany(
                "DELETE FROM chunks WHERE chunk_id=?", [(s,) for s in stale])
        return len(stale)

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def chunk_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """registry 查询：chunk_id -> 原文（引用校验用）。"""
        if not chunk_ids:
            return {}
        ph = ",".join("?" * len(chunk_ids))
        rows = self._conn().execute(
            f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({ph})", chunk_ids
        ).fetchall()
        return {r["chunk_id"]: r["text"] for r in rows}

    def chunks_of_doc(self, doc_key: str) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM chunks WHERE doc_key=? ORDER BY path_key", (doc_key,)
        ).fetchall()
        return [dict(r) for r in rows]

    def rich_blocks_of_page(self, doc_key: str, page: int | None) -> list[dict[str, Any]]:
        """某页的公式/图表块（实时读——含本轮问答中途回填新建的占位块）。"""
        if page is None:
            return []
        rows = self._conn().execute(
            "SELECT * FROM chunks WHERE doc_key=? AND page=? AND kind IN ('formula','chart')",
            (doc_key, page)).fetchall()
        return [dict(r) for r in rows]

    def all_chunk_metas(self) -> dict[str, dict[str, Any]]:
        """HybridIndex.metas 来源。"""
        rows = self._conn().execute(
            """SELECT c.chunk_id, c.text, c.text_norm, c.breadcrumb, c.kind,
                      c.page, c.meta_json, c.doc_key,
                      d.title AS doc_title, d.version AS version
               FROM chunks c JOIN documents d ON c.doc_key = d.doc_key"""
        ).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    # -- review tasks（人工复核队列）--------------------------------------------

    def create_review_task(self, task_id: str, query: str, as_of: str,
                           draft_answer: str, citations: list[dict[str, Any]],
                           envelope: dict[str, Any]) -> None:
        c = self._conn()
        c.execute(
            """INSERT OR REPLACE INTO review_tasks
               (task_id, created_at, query, as_of, draft_answer, citations_json,
                envelope_json, status, decision_note, final_answer, decided_at)
               VALUES(?,?,?,?,?,?,?,'pending','','',NULL)""",
            (task_id, time.time(), query, as_of, draft_answer,
             json.dumps(citations, ensure_ascii=False),
             json.dumps(envelope, ensure_ascii=False)),
        )
        c.commit()

    def get_review_task(self, task_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM review_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        return self._review_row(row) if row else None

    def list_review_tasks(self, status: str | None = None,
                          limit: int = 100) -> list[dict[str, Any]]:
        c = self._conn()
        if status:
            rows = c.execute(
                "SELECT * FROM review_tasks WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM review_tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._review_row(r) for r in rows]

    def decide_review_task(self, task_id: str, status: str, decision_note: str = "",
                           final_answer: str = "") -> bool:
        c = self._conn()
        # 只允许 pending → decided 一次（DB 层防御）
        cur = c.execute(
            """UPDATE review_tasks SET status=?, decision_note=?, final_answer=?,
               decided_at=? WHERE task_id=? AND status='pending'""",
            (status, decision_note, final_answer, time.time(), task_id),
        )
        c.commit()
        return cur.rowcount > 0

    @staticmethod
    def _review_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["citations"] = json.loads(d.pop("citations_json", "[]"))
        d["envelope"] = json.loads(d.pop("envelope_json", "{}"))
        return d


    # -- 富化缓存与任务 ---------------------------------------------------------

    def enrich_cache_get(self, file_hash: str, page: int, kind: str) -> dict | None:
        row = self._conn().execute(
            "SELECT payload_json FROM enrich_cache WHERE file_hash=? AND page=? AND kind=?",
            (file_hash, page, kind)).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def enrich_cache_put(self, file_hash: str, page: int, kind: str,
                         payload: dict) -> None:
        c = self._conn()
        c.execute(
            """INSERT OR REPLACE INTO enrich_cache
               (file_hash, page, kind, payload_json, updated_at)
               VALUES(?,?,?,?,?)""",
            (file_hash, page, kind, json.dumps(payload, ensure_ascii=True), time.time()))
        c.commit()

    def job_upsert(self, job_id: str, doc_key: str, kind: str, payload: dict,
                   status: str = "queued", progress: float = 0.0, note: str = "") -> None:
        c = self._conn()
        now = time.time()
        c.execute(
            """INSERT INTO rich_jobs (job_id, doc_key, kind, payload_json, status,
                                      progress, note, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(job_id) DO UPDATE SET
                 status=excluded.status, progress=excluded.progress,
                 note=excluded.note, updated_at=excluded.updated_at,
                 payload_json=excluded.payload_json""",
            (job_id, doc_key, kind, json.dumps(payload, ensure_ascii=True),
             status, progress, note, now, now))
        c.commit()

    def job_update(self, job_id: str, status: str, progress: float, note: str = "") -> None:
        c = self._conn()
        c.execute(
            "UPDATE rich_jobs SET status=?, progress=?, note=?, updated_at=? WHERE job_id=?",
            (status, progress, note, time.time(), job_id))
        c.commit()

    def job_get(self, job_id: str) -> dict | None:
        row = self._conn().execute(
            "SELECT * FROM rich_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json", "{}"))
        return d

    def job_list(self, doc_key: str | None = None,
                 status: str | None = None) -> list[dict[str, Any]]:
        """富化任务列表（进度页/重启续跑用）。"""
        sql = "SELECT * FROM rich_jobs WHERE 1=1"
        args: list[Any] = []
        if doc_key:
            sql += " AND doc_key=?"
            args.append(doc_key)
        if status:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY created_at"
        out = []
        for row in self._conn().execute(sql, args).fetchall():
            d = dict(row)
            d["payload"] = json.loads(d.pop("payload_json", "{}"))
            out.append(d)
        return out

    # -- audit -----------------------------------------------------------------

    def write_audit(self, request_id: str, channel: str, tool: str, request_hash: str,
                    status: str, trace_ref: str, corpus_version: str,
                    usage: dict[str, Any], verification: dict[str, Any],
                    query_text: str = "", answer_preview: str = "",
                    parent_request_id: str = "") -> None:
        c = self._conn()
        c.execute(
            """INSERT OR REPLACE INTO audit_log
               (request_id, ts, channel, tool, request_hash, status, trace_ref,
                corpus_version, usage_json, verification_json,
                query_text, answer_preview, parent_request_id)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (request_id, time.time(), channel, tool, request_hash, status,
             trace_ref, corpus_version, json.dumps(usage, ensure_ascii=False),
             json.dumps(verification, ensure_ascii=False),
             _t(query_text)[:2000], _t(answer_preview)[:500], _t(parent_request_id)),
        )
        c.commit()

    def list_audit(self, limit: int = 20, offset: int = 0,
                   kind: str = "all", q: str = "") -> list[dict[str, Any]]:
        """历史时间线（WS-F）。kind: all|answered|refused|batch。"""
        sql = "SELECT * FROM audit_log WHERE 1=1"
        args: list[Any] = []
        if kind == "answered":
            sql += " AND status='answered'"
        elif kind == "refused":
            sql += " AND status='refused'"
        elif kind == "batch":
            sql += " AND parent_request_id != ''"
        if q:
            sql += " AND query_text LIKE ?"
            args.append(f"%{q}%")
        sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [dict(r) for r in self._conn().execute(sql, args).fetchall()]

    def count_audit(self, kind: str = "all", q: str = "") -> int:
        sql = "SELECT COUNT(*) FROM audit_log WHERE 1=1"
        args: list[Any] = []
        if kind == "answered":
            sql += " AND status='answered'"
        elif kind == "refused":
            sql += " AND status='refused'"
        elif kind == "batch":
            sql += " AND parent_request_id != ''"
        if q:
            sql += " AND query_text LIKE ?"
            args.append(f"%{q}%")
        return self._conn().execute(sql, args).fetchone()[0]

    def get_audit(self, request_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM audit_log WHERE request_id=? OR parent_request_id=?",
            (request_id, request_id)).fetchone()
        return dict(row) if row else None
