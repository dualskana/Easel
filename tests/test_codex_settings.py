"""Codex 设置面板（runtime=codex）的配置层与 web 端点回归。

重点：状态来自 codex doctor（不含凭据）；模型只写项目 .env 的 EASEL_CODEX_MODEL；
非 codex runtime 拒绝保存；面板保存后下一轮 codex exec 命令实际带 -m（端到端生效）。

运行：pytest tests/test_codex_settings.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web"))

import app as web  # noqa: E402
from easel.runtimes import RunRequest, codex, codex_config, common  # noqa: E402

DOCTOR_REPORT = {
    "codexVersion": "0.155.1",
    "checks": {
        "auth.credentials": {"status": "ok", "details": {
            "stored auth mode": "chatgpt", "stored API key": "false",
        }},
        "config.load": {"status": "ok", "details": {
            "model": "gpt-6-astra", "config.toml": "/Users/u/.codex/config.toml",
        }},
    },
}


class FakeProc:
    def __init__(self):
        self.returncode = 0
        self.stdout = iter([])

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


@pytest.fixture()
def sandbox_env(tmp_path, monkeypatch):
    """沙箱 .env：codex_config 与 common 都指到它，绝不碰用户真配置。"""
    monkeypatch.delenv(codex_config.MODEL_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET_MARKER=keep-me\n", encoding="utf-8")
    monkeypatch.setattr(codex_config, "ENV_FILE", env_file)
    monkeypatch.setattr(common, "ENV_FILE", env_file)
    return env_file


@pytest.fixture()
def doctor(monkeypatch):
    """假 doctor 报告 + 假目录 + 假 codex 可执行文件；CI 上不依赖真实安装。"""
    monkeypatch.setattr(codex_config.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(codex_config, "doctor_report", lambda: json.loads(json.dumps(DOCTOR_REPORT)))
    monkeypatch.setattr(codex_config, "catalog_models", lambda: [])
    monkeypatch.setattr(codex_config, "_catalog", lambda: ([], {}))
    return DOCTOR_REPORT


@pytest.fixture()
def client(sandbox_env, doctor, monkeypatch):
    monkeypatch.setenv("EASEL_AGENT_RUNTIME", "codex")
    with TestClient(web.app) as c:
        yield c


# ---- 快照：状态与降级 ----

def test_snapshot_reports_version_login_and_model(sandbox_env, doctor):
    snap = codex_config.snapshot()
    assert snap["installed"] is True
    assert snap["version"] == "0.155.1"
    assert snap["loggedIn"] is True and snap["authMode"] == "chatgpt"
    assert snap["model"] == "gpt-6-astra"
    assert snap["easelModel"] == ""
    assert snap["candidates"] == ["gpt-6-astra"]
    blob = json.dumps(snap, ensure_ascii=False)
    assert "stored API key" not in blob and "/.codex/config.toml" not in blob


def test_snapshot_degrades_without_doctor(sandbox_env, monkeypatch):
    monkeypatch.setattr(codex_config.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(codex_config, "doctor_report", lambda: {})
    monkeypatch.setattr(codex_config, "catalog_models", lambda: [])
    snap = codex_config.snapshot()
    assert snap["loggedIn"] is False and snap["version"] == ""
    assert "doctor" in snap["message"]


def test_snapshot_reports_missing_cli(sandbox_env, monkeypatch):
    monkeypatch.setattr(codex_config.shutil, "which", lambda _name: None)
    monkeypatch.setattr(codex_config, "doctor_report", lambda: {})
    snap = codex_config.snapshot()
    assert snap["installed"] is False
    assert "npm install" in snap["message"]


# ---- 模型目录（codex debug models） ----

CATALOG = {"models": [
    {"slug": "gpt-6-astra", "visibility": "list", "display_name": "GPT-6-Astra",
     "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}]},
    {"slug": "gpt-reserve", "visibility": "hide", "display_name": "GPT-Reserve",
     "supported_reasoning_levels": [{"effort": "low"}]},
    {"slug": "gpt-5.6-sol", "visibility": "list", "display_name": "GPT-5.6-Sol",
     "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}]},
    {"slug": "gpt-5.5", "visibility": "list", "display_name": "GPT-5.5",
     "supported_reasoning_levels": []},
    {"slug": "gpt-5.6-sol", "visibility": "list", "display_name": "GPT-5.6-Sol（重复）",
     "supported_reasoning_levels": [{"effort": "low"}]},
]}

CATALOG_LEVELS = {"gpt-6-astra": ["low", "high"], "gpt-5.6-sol": ["low", "medium"]}


def _run_result(stdout="", returncode=0):
    class Completed:
        pass

    completed = Completed()
    completed.returncode = returncode
    completed.stdout = stdout
    return completed


def test_catalog_models_filters_hidden_and_dedupes(monkeypatch):
    monkeypatch.setattr(codex_config.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(codex, "base_cmd", lambda: ["/bin/codex"])
    monkeypatch.setattr(codex_config.subprocess, "run",
                        lambda cmd, **kwargs: _run_result(json.dumps(CATALOG)))
    assert codex_config.catalog_models() == ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.5"]


def test_catalog_models_falls_back_to_bundled(monkeypatch):
    monkeypatch.setattr(codex_config.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(codex, "base_cmd", lambda: ["/bin/codex"])
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return _run_result("", returncode=1)
        return _run_result(json.dumps(CATALOG))

    monkeypatch.setattr(codex_config.subprocess, "run", run)
    assert codex_config.catalog_models() == ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.5"]
    assert len(calls) == 2 and calls[1][-1] == "--bundled"


def test_catalog_models_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(codex_config.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(codex, "base_cmd", lambda: ["/bin/codex"])

    def boom(cmd, **kwargs):
        raise OSError("no codex")

    monkeypatch.setattr(codex_config.subprocess, "run", boom)
    assert codex_config.catalog_models() == []


def test_snapshot_candidates_merge_catalog(sandbox_env, doctor, monkeypatch):
    monkeypatch.setattr(codex_config, "catalog_models",
                        lambda: ["gpt-6-astra", "gpt-5.6-terra", "gpt-5.5"])
    snap = codex_config.snapshot()
    assert snap["candidates"] == ["gpt-6-astra", "gpt-5.6-terra", "gpt-5.5"]

    codex_config.set_easel_model("gpt-5.5")
    snap = codex_config.snapshot()
    assert snap["candidates"] == ["gpt-5.5", "gpt-6-astra", "gpt-5.6-terra"]


# ---- 思考强度（档位解析、校验、写盘、端到端） ----

def test_debug_catalog_parses_visible_levels(monkeypatch):
    monkeypatch.setattr(codex_config.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(codex, "base_cmd", lambda: ["/bin/codex"])
    monkeypatch.setattr(codex_config.subprocess, "run",
                        lambda cmd, **kwargs: _run_result(json.dumps(CATALOG)))
    models, levels = codex_config._debug_catalog([])
    assert models == ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.5"]
    assert levels == CATALOG_LEVELS


def test_reasoning_levels_from_catalog_and_fallback(monkeypatch):
    monkeypatch.setattr(codex_config, "_catalog", lambda: (list(CATALOG_LEVELS), dict(CATALOG_LEVELS)))
    assert codex_config.reasoning_levels("gpt-6-astra") == ["low", "high"]
    assert codex_config.reasoning_levels("unknown") == codex_config.DEFAULT_REASONING_LEVELS

    monkeypatch.setattr(codex_config, "_catalog", lambda: ([], {}))
    assert codex_config.reasoning_levels("gpt-6-astra") == codex_config.DEFAULT_REASONING_LEVELS


@pytest.mark.parametrize("value", ["", "  "])
def test_validate_reasoning_allows_empty(value, monkeypatch):
    monkeypatch.setattr(codex_config, "_catalog", lambda: ([], {}))
    assert codex_config.validate_reasoning(value) == ""


@pytest.mark.parametrize("value", ["bogus", "HIGH", "bad value", "-x", "a" * 17, "m\nodel"])
def test_validate_reasoning_rejects_bad_input(value, monkeypatch):
    monkeypatch.setattr(codex_config, "_catalog", lambda: (list(CATALOG_LEVELS), dict(CATALOG_LEVELS)))
    with pytest.raises(ValueError):
        codex_config.validate_reasoning(value)


def test_validate_reasoning_uses_target_model_levels(monkeypatch):
    levels = {"gpt-6-astra": ["low", "high"], "gpt-5.5": ["low"]}
    monkeypatch.setattr(codex_config, "_catalog", lambda: (list(levels), dict(levels)))
    assert codex_config.validate_reasoning("high", "gpt-6-astra") == "high"
    with pytest.raises(ValueError, match="gpt-5.5"):
        codex_config.validate_reasoning("high", "gpt-5.5")
    assert codex_config.validate_reasoning("low", "gpt-5.5") == "low"


def test_set_easel_reasoning_writes_and_clears_env(sandbox_env, monkeypatch):
    monkeypatch.setattr(codex_config, "_catalog", lambda: (list(CATALOG_LEVELS), dict(CATALOG_LEVELS)))
    codex_config.set_easel_reasoning("high")
    body = sandbox_env.read_text(encoding="utf-8")
    assert "EASEL_CODEX_REASONING_EFFORT=high" in body
    assert "SECRET_MARKER=keep-me" in body
    assert codex_config.reasoning() == "high"

    codex_config.set_easel_reasoning("")
    assert codex_config.reasoning() == ""
    assert "EASEL_CODEX_REASONING_EFFORT" not in sandbox_env.read_text(encoding="utf-8")


def test_snapshot_includes_reasoning_and_levels(sandbox_env, doctor, monkeypatch):
    monkeypatch.setattr(codex_config, "_catalog", lambda: (list(CATALOG_LEVELS), dict(CATALOG_LEVELS)))
    snap = codex_config.snapshot()
    assert snap["reasoning"] == ""
    assert snap["reasoningLevels"] == ["low", "high"]  # gpt-6-astra

    codex_config.set_easel_reasoning("high")
    snap = codex_config.snapshot()
    assert snap["reasoning"] == "high" and snap["reasoningLevels"] == ["low", "high"]


# ---- 模型校验与写入 ----

@pytest.mark.parametrize("value", ["", "   ", "bad model", "model;rm -rf /", "-x", "a" * 129, "m\nodel"])
def test_validate_model_rejects_bad_input(value):
    with pytest.raises(ValueError):
        codex_config.validate_model(value)


def test_set_easel_model_updates_env_and_preserves_other_lines(sandbox_env):
    codex_config.set_easel_model("gpt-6-astra")
    codex_config.set_easel_model("gpt-6-astra")
    body = sandbox_env.read_text(encoding="utf-8")
    assert body.count("EASEL_CODEX_MODEL=") == 1
    assert "SECRET_MARKER=keep-me" in body
    assert codex_config.model() == "gpt-6-astra"

    codex_config.set_easel_model("gpt-6-mini")
    body = sandbox_env.read_text(encoding="utf-8")
    assert body.count("EASEL_CODEX_MODEL=") == 1
    assert codex_config.model() == "gpt-6-mini"


# ---- web 端点 ----

def test_api_get_returns_status_without_secrets(client):
    resp = client.get("/api/settings/codex")
    assert resp.status_code == 200
    body = resp.json()
    assert body["installed"] is True and body["loggedIn"] is True
    assert body["model"] == "gpt-6-astra"
    assert body["reasoning"] == ""
    assert body["reasoningLevels"] == codex_config.DEFAULT_REASONING_LEVELS
    blob = json.dumps(body, ensure_ascii=False)
    assert "stored API key" not in blob and "/.codex/config.toml" not in blob


def test_api_save_writes_env_and_is_effective_next_turn(client, sandbox_env, tmp_path, monkeypatch):
    resp = client.post("/api/settings/codex/save", json={"model": "gpt-6-mini", "reasoning": "high"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    body = sandbox_env.read_text(encoding="utf-8")
    assert "EASEL_CODEX_MODEL=gpt-6-mini" in body
    assert "EASEL_CODEX_REASONING_EFFORT=high" in body

    seen = {}

    def popen(cmd, **kwargs):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(codex.subprocess, "Popen", popen)
    monkeypatch.setattr(codex, "base_cmd", lambda: ["/bin/codex"])
    handle = codex.CodexAdapter().start(RunRequest("hi", "sk", 5, tmp_path, {}, tmp_path / "sessions"))
    cmd = seen["cmd"]
    assert cmd[cmd.index("-m") + 1] == "gpt-6-mini"
    assert 'model_reasoning_effort="high"' in cmd
    handle.close()


def test_api_save_reasoning_only(client, sandbox_env):
    resp = client.post("/api/settings/codex/save", json={"reasoning": "medium"})
    assert resp.status_code == 200
    assert resp.json()["reasoning"] == "medium"
    assert "EASEL_CODEX_REASONING_EFFORT=medium" in sandbox_env.read_text(encoding="utf-8")
    assert "EASEL_CODEX_MODEL" not in sandbox_env.read_text(encoding="utf-8")


@pytest.mark.parametrize("payload", [{"reasoning": "bogus"}, {"reasoning": "HIGH"}, {"reasoning": "bad value"}])
def test_api_save_rejects_invalid_reasoning_without_side_effects(client, sandbox_env, payload):
    before = sandbox_env.read_text(encoding="utf-8")
    resp = client.post("/api/settings/codex/save", json=payload)
    assert resp.status_code == 400
    assert sandbox_env.read_text(encoding="utf-8") == before


def test_api_save_validates_all_before_writing_anything(client, sandbox_env, monkeypatch):
    """非法强度不能让同请求里的合法模型落到 .env（无半写）。"""
    monkeypatch.setattr(codex_config, "_catalog",
                        lambda: (list(CATALOG_LEVELS), dict(CATALOG_LEVELS)))
    before = sandbox_env.read_text(encoding="utf-8")
    resp = client.post("/api/settings/codex/save",
                       json={"model": "gpt-6-astra", "reasoning": "bogus"})
    assert resp.status_code == 400
    assert sandbox_env.read_text(encoding="utf-8") == before


def test_api_save_reasoning_must_match_target_model(client, sandbox_env, monkeypatch):
    levels = {"gpt-5.5": ["low"]}
    monkeypatch.setattr(codex_config, "_catalog", lambda: (list(levels), dict(levels)))
    before = sandbox_env.read_text(encoding="utf-8")
    resp = client.post("/api/settings/codex/save", json={"model": "gpt-5.5", "reasoning": "high"})
    assert resp.status_code == 400
    assert sandbox_env.read_text(encoding="utf-8") == before

    resp = client.post("/api/settings/codex/save", json={"model": "gpt-5.5", "reasoning": "low"})
    assert resp.status_code == 200
    body = sandbox_env.read_text(encoding="utf-8")
    assert "EASEL_CODEX_MODEL=gpt-5.5" in body and "EASEL_CODEX_REASONING_EFFORT=low" in body


def test_api_save_rejected_for_other_runtime(sandbox_env, doctor, monkeypatch):
    monkeypatch.setenv("EASEL_AGENT_RUNTIME", "openclaw")
    with TestClient(web.app) as c:
        resp = c.post("/api/settings/codex/save", json={"model": "gpt-6-mini"})
    assert resp.status_code == 400
    assert "EASEL_CODEX_MODEL" not in sandbox_env.read_text(encoding="utf-8")


@pytest.mark.parametrize("payload", [{"model": ""}, {"model": "   "}, {"model": "bad model"}, {"model": "-x"}])
def test_api_save_rejects_invalid_without_side_effects(client, sandbox_env, payload):
    before = sandbox_env.read_text(encoding="utf-8")
    resp = client.post("/api/settings/codex/save", json=payload)
    assert resp.status_code == 400
    assert sandbox_env.read_text(encoding="utf-8") == before
