"""原生库子进程隔离回归（#15）：OCR 段错误/超时只回空文本，服务不陪葬。"""

import json
import subprocess
from pathlib import Path

from policyrag.ingest.native_worker import ocr_isolated


def _img(tmp_path):
    p = tmp_path / "g.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


def test_crash_returns_empty(tmp_path, monkeypatch):
    """子进程段错误（result 未写出）→ []，不抛异常。"""
    def fake_run(cmd, capture_output=True, timeout=None):
        return subprocess.CompletedProcess(cmd, returncode=-11)
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ocr_isolated(_img(tmp_path)) == []


def test_timeout_returns_empty(tmp_path, monkeypatch):
    def fake_run(cmd, capture_output=True, timeout=None):
        raise subprocess.TimeoutExpired(cmd, timeout or 1)
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ocr_isolated(_img(tmp_path)) == []


def test_success_parses_result(tmp_path, monkeypatch):
    def fake_run(cmd, capture_output=True, timeout=None):
        rp = Path(cmd[-1])
        rp.write_text(json.dumps({"ok": True, "data": ["38877.25", "标题"]}),
                      encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ocr_isolated(_img(tmp_path)) == ["38877.25", "标题"]


def test_child_error_result_returns_empty(tmp_path, monkeypatch):
    def fake_run(cmd, capture_output=True, timeout=None):
        rp = Path(cmd[-1])
        rp.write_text(json.dumps({"ok": False, "error": "boom"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ocr_isolated(_img(tmp_path)) == []


def test_missing_file_returns_empty(tmp_path):
    assert ocr_isolated(tmp_path / "nope.png") == []
