"""BM25 稀疏检索（rank-bm25）+ 中文 tokenizer。"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from .tokenizer_zh import TOKENIZER_VERSION, tokenize

__all__ = ["BM25Index"]


class BM25Index:
    """corpus.json + 内存 BM25Okapi。数百份文档规模完全够用。

    corpus.json 携带 tokenizer_version 与原文 texts：分词规则升级（如条款
    别名词元）后 `_open` 自动用缓存文本重分词——否则旧语料词元永远打不中
    查询侧新别名（重启后静默失配的坑）。
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._corpus_path = self.root / "corpus.json"
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self.stale: bool = False   # 无 texts 可重分词 → 需调用方 rebuild

    def build(self, ids: list[str], texts: list[str]) -> None:
        tokenized = [tokenize(t) for t in texts]
        self._corpus_path.write_text(
            json.dumps(
                {"ids": ids, "texts": texts, "tokenized": tokenized,
                 "tokenizer_version": TOKENIZER_VERSION},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._ids = ids
        self.stale = False
        # rank-bm25 对空语料抛 ZeroDivisionError——空库单独标记
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _open(self) -> BM25Okapi | None:
        if self._bm25 is None and self._corpus_path.exists():
            data = json.loads(self._corpus_path.read_text(encoding="utf-8"))
            self._ids = data["ids"]
            tokenized = data.get("tokenized") or []
            if data.get("tokenizer_version") != TOKENIZER_VERSION:
                texts = data.get("texts")
                if texts is not None and len(texts) == len(self._ids):
                    tokenized = [tokenize(t) for t in texts]  # 重分词重建
                    self._corpus_path.write_text(
                        json.dumps(
                            {"ids": self._ids, "texts": texts,
                             "tokenized": tokenized,
                             "tokenizer_version": TOKENIZER_VERSION},
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                else:
                    self.stale = True   # 旧版无 texts——调用方 rebuild_index 兜底
            if tokenized:
                self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    @property
    def size(self) -> int:
        bm25 = self._open()
        return len(self._ids) if bm25 is not None else 0

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        bm25 = self._open()
        if bm25 is None or not self._ids:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = bm25.get_scores(q)
        k = min(k, len(scores))
        if k <= 0:
            return []
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._ids[i], float(scores[i])) for i in idx if scores[i] > 0]
