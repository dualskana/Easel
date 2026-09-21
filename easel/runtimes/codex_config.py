"""Codex 配置读取与 Easel 模型/思考强度选择（只读 ~/.codex，写项目 .env）。

对应 opencode_config.py 的角色：适配器负责跑 agent，这里回答「Codex 装没装、登录没登录、
本机默认模型是什么」以及「Easel 选用哪个模型、哪个思考强度」。

- 模型与思考强度只写项目 .env 的 EASEL_CODEX_MODEL / EASEL_CODEX_REASONING_EFFORT，
  由适配器每轮以 `-m` 与 `-c model_reasoning_effort` 传给 codex；不改 ~/.codex/config.toml，
  也不碰任何凭据文件。
- 状态经 `codex doctor --json` 读取（含版本、登录、模型），不解析用户私有配置；
  模型目录与思考强度档位经 `codex debug models` 读取（debug 命名空间，按 best-effort 处理）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

from . import codex
from .common import ENV_FILE, PROJECT_ROOT, env_file_value, runtime_env

MODEL_VAR = "EASEL_CODEX_MODEL"
REASONING_VAR = "EASEL_CODEX_REASONING_EFFORT"
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}\Z")
_REASONING_RE = re.compile(r"[a-z][a-z0-9-]{1,15}\Z")
# 目录不可用时退化的档位；这些是老版本 Codex 也认的常用值
DEFAULT_REASONING_LEVELS = ["low", "medium", "high"]


def model() -> str:
    """Easel 侧选用的 Codex 模型；空表示用 Codex 自身默认。

    项目 .env 优先于进程环境：设置面板写的是 .env，面板改动必须立刻在下一轮生效，
    不能被启动时残留的进程环境变量盖住。
    """
    return (env_file_value(MODEL_VAR) or os.environ.get(MODEL_VAR) or "").strip()


def reasoning() -> str:
    """Easel 侧选用的思考强度；空表示跟随 Codex 自身默认。"""
    return (env_file_value(REASONING_VAR) or os.environ.get(REASONING_VAR) or "").strip()


def validate_model(value: str) -> str:
    model_value = (value or "").strip()
    if not _MODEL_RE.fullmatch(model_value):
        raise ValueError("模型名不合法（1-128 位字母/数字/._:/@+-，且不能以符号开头）")
    return model_value


def validate_reasoning(value: str, model_name: str | None = None) -> str:
    """校验思考强度：必须属于目标模型（默认 Easel 当前生效模型）支持的档位；空串清除覆盖。"""
    reasoning_value = (value or "").strip()
    if not reasoning_value:
        return ""
    if not _REASONING_RE.fullmatch(reasoning_value):
        raise ValueError("思考强度不合法（1-16 位小写字母/数字/连字符）")
    levels = reasoning_levels(model_name if model_name is not None else model())
    if reasoning_value not in levels:
        raise ValueError(
            f"思考强度不在模型 {model_name or model() or '（本机默认）'} 支持范围内：{reasoning_value[:40]}")
    return reasoning_value


def doctor_report() -> dict:
    """`codex doctor --json` 的解析结果；未安装或失败时返回空字典（调用方降级）。"""
    if shutil.which("codex") is None:
        return {}
    try:
        cmd = codex.base_cmd() + ["doctor", "--json"]
    except FileNotFoundError:
        return {}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                cwd=PROJECT_ROOT, env=runtime_env())
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def login_status() -> tuple[bool, str]:
    """`codex login status` 的轻量探测：返回 (已登录, 可读说明)。"""
    try:
        cmd = codex.base_cmd() + ["login", "status"]
    except FileNotFoundError as exc:
        return False, str(exc)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                                cwd=PROJECT_ROOT, env=runtime_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    lines = (result.stdout or result.stderr or "").strip().splitlines()
    message = lines[0][:200] if lines else ""
    if result.returncode == 0:
        return True, message or "已登录"
    return False, message or "未登录；请在终端运行 codex login"


def _check(report: dict, check_id: str) -> dict:
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    entry = checks.get(check_id) if isinstance(checks, dict) else None
    return entry if isinstance(entry, dict) else {}


def _details(report: dict, check_id: str) -> dict:
    details = _check(report, check_id).get("details")
    return details if isinstance(details, dict) else {}


def _debug_catalog(extra_args: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """`codex debug models [--bundled]`：返回 (可见模型 slug, {slug: 支持的思考强度})；失败返回空。"""
    try:
        cmd = codex.base_cmd() + ["debug", "models", *extra_args]
    except FileNotFoundError:
        return [], {}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                                cwd=PROJECT_ROOT, env=runtime_env())
    except (OSError, subprocess.TimeoutExpired):
        return [], {}
    if result.returncode != 0:
        return [], {}
    try:
        data = json.loads(result.stdout)
    except (TypeError, ValueError):
        return [], {}
    items = data.get("models") if isinstance(data, dict) else None
    slugs: list[str] = []
    levels: dict[str, list[str]] = {}
    for item in items or []:
        if not isinstance(item, dict) or str(item.get("visibility") or "list") != "list":
            continue
        slug = str(item.get("slug") or "").strip()
        if not slug:
            continue
        if slug not in slugs:
            slugs.append(slug)
        rows = item.get("supported_reasoning_levels")
        model_levels: list[str] = []
        for row in rows or []:
            effort = str(row.get("effort") or "").strip() if isinstance(row, dict) else ""
            if effort and effort not in model_levels:
                model_levels.append(effort)
        if model_levels and slug not in levels:
            levels[slug] = model_levels
    return slugs, levels


def _catalog() -> tuple[list[str], dict[str, list[str]]]:
    """模型目录：先刷新，失败退回内置目录（离线可用），再失败返回空。"""
    if shutil.which("codex") is None:
        return [], {}
    models, levels = _debug_catalog([])
    if models:
        return models, levels
    return _debug_catalog(["--bundled"])


def catalog_models() -> list[str]:
    return _catalog()[0]


def reasoning_levels(model_name: str) -> list[str]:
    """给定模型支持的思考强度档位；目录不可用或该模型没有记录时退化为默认档位。"""
    _, levels = _catalog()
    return list(levels.get(model_name or "") or DEFAULT_REASONING_LEVELS)


def auth_state(report: dict) -> tuple[bool, str]:
    entry = _check(report, "auth.credentials")
    return entry.get("status") == "ok", str(_details(report, "auth.credentials").get("stored auth mode") or "")


def config_model(report: dict) -> str:
    return str(_details(report, "config.load").get("model") or "")


def snapshot() -> dict:
    """设置面板状态：安装/版本/登录/本机模型/Easel 选择/候选项/思考强度；无凭据、失败降级。"""
    installed = shutil.which("codex") is not None
    base = {"installed": installed, "version": "", "loggedIn": False, "authMode": "",
            "model": "", "easelModel": model(), "candidates": [],
            "reasoning": reasoning(), "reasoningLevels": reasoning_levels(model()),
            "message": ""}
    report = doctor_report()
    if not report:
        base["message"] = ("Codex CLI 未安装；运行 setup 或 npm install -g @openai/codex@latest"
                           if not installed else "codex doctor 暂不可用；请在终端运行 codex doctor 检查")
        base["candidates"] = list(dict.fromkeys(
            m for m in (base["easelModel"], *catalog_models()) if m))
        return base
    base["version"] = str(report.get("codexVersion") or "")
    base["loggedIn"], base["authMode"] = auth_state(report)
    base["model"] = config_model(report)
    # 候选：Easel 选中的 → 本机默认 → 目录里的其余可见模型（保持目录顺序）
    catalog = catalog_models()
    base["candidates"] = list(dict.fromkeys(
        m for m in (base["easelModel"], base["model"], *catalog) if m))
    base["reasoningLevels"] = reasoning_levels(base["easelModel"] or base["model"])
    if not base["loggedIn"]:
        base["message"] = "Codex 未登录；请在终端运行 codex login"
    return base


def _write_env_value(name: str, value: str) -> None:
    """就地更新/追加/删除项目 .env 的一个键（原子写，保留其它行）；空值删除该行。"""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.is_file() else []
    out: list[str] = []
    seen = False
    for line in lines:
        stripped = line.strip()
        if (stripped and not stripped.startswith("#") and "=" in stripped
                and stripped.split("=", 1)[0].strip() == name):
            seen = True
            if value:
                out.append(f"{name}={value}")
            continue
        out.append(line)
    if not seen and value:
        if out and out[-1].strip() != "":
            out.append("")
        out.append("# ---- Easel Codex 配置（Web 设置面板写入）----")
        out.append(f"{name}={value}")
    tmp = ENV_FILE.with_suffix(".env.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(ENV_FILE)


def set_easel_model(value: str) -> None:
    """把 Easel 选用的模型写入项目 .env。"""
    model_value = validate_model(value)
    if env_file_value(MODEL_VAR) != model_value:
        _write_env_value(MODEL_VAR, model_value)


def set_easel_reasoning(value: str) -> None:
    """把 Easel 选用的思考强度写入项目 .env；空串删除该行（跟随 Codex 默认）。"""
    reasoning_value = validate_reasoning(value)
    if env_file_value(REASONING_VAR) != reasoning_value:
        _write_env_value(REASONING_VAR, reasoning_value)
